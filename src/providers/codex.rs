//! Codex / ChatGPT subscription provider.
//!
//! `login` drives the ChatGPT device-authorization flow and persists an
//! access/refresh token pair plus the `chatgpt_account_id` claim (required as
//! the `chatgpt-account-id` header on Responses API calls).
//!
//! Note: OpenAI's ChatGPT flow is *not* standard RFC 8628. It uses JSON-bodied
//! requests to `/api/accounts/deviceauth/usercode` and `/deviceauth/token`,
//! returns an `authorization_code` + PKCE `code_verifier` instead of a token,
//! and finishes with a standard `authorization_code` grant at `/oauth/token`.
//! The shared `auth::device_flow` helper (RFC 8628, form-encoded) is therefore
//! not reused here — it stays for the GitHub Copilot provider. Reference
//! implementations: `codex-rs/login/src/device_code_auth.rs` and opencode's
//! `packages/opencode/src/plugin/codex.ts`.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::Utc;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::oauth_refresh::{
    AuthenticationError, EXPIRY_THRESHOLD_SECS, OauthFreshness, UpstreamHttpError, is_near_expiry,
    oauth_freshness, provider_refresh_lock,
};
use crate::providers::common::responses_api::{
    chat_request_to_responses_body, responses_sse_to_chat_chunks,
};
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider, ReasoningFormat};
use crate::usage::UsageSnapshot;

/// OAuth client identifier published by the Codex CLI. Matches the value in
/// `codex-rs/login/src/auth/manager.rs` and opencode's plugin/codex.ts.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER: &str = "https://auth.openai.com";
/// User-facing verification URL shown when the server omits one in the
/// usercode response (matches the `{base_url}/codex/device` value in
/// `codex-rs/login/src/device_code_auth.rs`).
const DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
/// Default ChatGPT backend for `Provider::chat_completion`. Can be overridden
/// per-request via `ProviderSettings::base_url` (primarily for tests).
const CHATGPT_BACKEND: &str = "https://chatgpt.com";
const RESPONSES_PATH: &str = "/backend-api/codex/responses";
/// ChatGPT rate-limit / plan-consumption endpoint (`codex-rs/backend-client`
/// → `Client::get_rate_limits_many`). Returns `{ rate_limit: { primary_window,
/// secondary_window }, plan_type, ... }`.
const USAGE_PATH: &str = "/backend-api/wham/usage";

pub struct CodexProvider;

#[async_trait]
impl Provider for CodexProvider {
    fn id(&self) -> &'static str {
        "codex"
    }

    fn display_name(&self) -> &'static str {
        "Codex (ChatGPT Plus/Pro)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        // Sourced from codex-rs's bundled `models-manager/models.json` — the
        // compile-time catalogue the Codex CLI itself loads. Codex also hits
        // a live `/models` endpoint when authed via ChatGPT, but this trait
        // method is synchronous and can't block on network I/O; callers that
        // want a live view should use `mixer providers models codex` (which
        // falls through the default `list_remote_models` → not supported,
        // because Codex's live catalogue rides the Responses API auth).
        vec![
            ModelInfo::new("gpt-5.4", "GPT-5.4", true, 400_000)
                .with_reasoning(ReasoningFormat::ResponsesApiSummary),
            ModelInfo::new("gpt-5.4-mini", "GPT-5.4 Mini", true, 400_000)
                .with_reasoning(ReasoningFormat::ResponsesApiSummary),
            ModelInfo::new("gpt-5.3-codex", "GPT-5.3 Codex", true, 400_000)
                .with_reasoning(ReasoningFormat::ResponsesApiSummary),
            ModelInfo::new("gpt-5.2", "GPT-5.2", true, 400_000)
                .with_reasoning(ReasoningFormat::ResponsesApiSummary),
        ]
    }

    fn auth_kind(&self) -> AuthKind {
        AuthKind::DeviceFlow
    }

    fn is_authenticated(&self, store: &CredentialStore, _settings: &ProviderSettings) -> bool {
        let Ok(Some(blob)) = store.load_blob(self.id()) else {
            return false;
        };
        let has_access = blob
            .get("access_token")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());
        if !has_access {
            return false;
        }
        // A non-expired access_token is authenticated; an expired one is still
        // authenticated when we can refresh it.
        match oauth_freshness(&blob, Utc::now().timestamp()) {
            OauthFreshness::Valid | OauthFreshness::ExpiredRefreshable => true,
            OauthFreshness::ExpiredDead => false,
        }
    }

    async fn usage(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        // Best-effort: a broken usage probe must not poison routing. Any
        // failure — missing credentials, refresh denied, non-200, malformed
        // body — degrades to `Ok(None)`, which the router treats as "unknown
        // consumption, weight 0.5".
        Ok(fetch_codex_usage(store, settings).await.ok().flatten())
    }

    async fn login(&self, store: &CredentialStore) -> Result<()> {
        let client = Client::new();
        let tokens = run_codex_device_flow(&client, /* open_browser */ true).await?;

        let id_token = tokens
            .id_token
            .as_deref()
            .ok_or_else(|| anyhow!("codex token response is missing id_token"))?;
        let account_id = extract_chatgpt_account_id(id_token)?;
        let expires_at = compute_expires_at(Utc::now().timestamp(), tokens.expires_in);

        let blob = json!({
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "id_token": tokens.id_token,
            "expires_at": expires_at,
            "chatgpt_account_id": account_id,
        });
        store.save(self.id(), &blob)?;

        tracing::info!(provider = "codex", account_id = %account_id, "signed in");
        Ok(())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let client = build_codex_http_client(settings)?;
        let base = settings.base_url.as_deref().unwrap_or(CHATGPT_BACKEND);
        let url = format!("{}{RESPONSES_PATH}", base.trim_end_matches('/'));
        let issuer_base = auth_issuer(settings);

        let model = req.model.clone();
        let body = chat_request_to_responses_body(&req, &model);
        let now = Utc::now().timestamp();

        let (access_token, account_id) =
            ensure_fresh_codex_token(store, &client, &issuer_base, now).await?;

        let resp = send_codex_chat(&client, &url, &access_token, &account_id, &body).await?;

        let resp = if resp.status() == StatusCode::UNAUTHORIZED {
            drop(resp);
            let (access_token, account_id) =
                force_refresh_codex_token(store, &client, &issuer_base, &access_token, now).await?;
            send_codex_chat(&client, &url, &access_token, &account_id, &body).await?
        } else {
            resp
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(1024).collect();
            return Err(anyhow::Error::new(UpstreamHttpError {
                status: status.as_u16(),
                body_snippet: snippet,
            }));
        }

        let event_stream = resp
            .bytes_stream()
            .eventsource()
            .map(|r| r.map_err(|e| anyhow!("codex SSE stream error: {e}")));
        let chunks = responses_sse_to_chat_chunks(event_stream);
        Ok(Box::pin(chunks))
    }
}

/// Resolve the OAuth issuer base URL for `/oauth/token` calls. In `cfg(test)`
/// a caller-supplied `base_url` doubles as the issuer so the mock server can
/// serve both the chat and refresh endpoints. In release builds the issuer
/// is always the hardcoded production endpoint — a `base_url` override only
/// affects the chat endpoint (users set it for regional mirrors, not to
/// redirect auth traffic).
fn auth_issuer(settings: &ProviderSettings) -> String {
    #[cfg(test)]
    if let Some(base) = settings.base_url.as_deref() {
        return base.trim_end_matches('/').to_string();
    }
    let _ = settings;
    ISSUER.to_string()
}

async fn send_codex_chat(
    client: &Client,
    url: &str,
    access_token: &str,
    account_id: &str,
    body: &Value,
) -> Result<reqwest::Response> {
    client
        .post(url)
        .bearer_auth(access_token)
        .header("chatgpt-account-id", account_id)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(body)
        .send()
        .await
        .context("posting to codex responses endpoint")
}

/// Proactive-refresh entry point: if the stored access token is within
/// [`EXPIRY_THRESHOLD_SECS`] of expiry, refresh under the per-provider mutex
/// before returning. Tasks arriving during an in-flight refresh block on the
/// mutex and, on acquiring it, re-read the freshly written credentials
/// instead of refreshing again.
async fn ensure_fresh_codex_token(
    store: &CredentialStore,
    client: &Client,
    issuer_base: &str,
    now: i64,
) -> Result<(String, String)> {
    let blob = load_codex_blob(store)?;
    if !is_near_expiry(&blob, now, EXPIRY_THRESHOLD_SECS) {
        return extract_access_and_account(&blob);
    }

    let lock = provider_refresh_lock("codex");
    let _guard = lock.lock().await;

    let blob = load_codex_blob(store)?;
    if !is_near_expiry(&blob, now, EXPIRY_THRESHOLD_SECS) {
        return extract_access_and_account(&blob);
    }

    refresh_and_persist(store, client, issuer_base, &blob, now).await
}

/// 401-fallback entry point: a dispatch came back Unauthorized, so refresh
/// and retry. If another task raced us to the refresh (its new token is
/// already on disk), use that token instead of refreshing a second time.
async fn force_refresh_codex_token(
    store: &CredentialStore,
    client: &Client,
    issuer_base: &str,
    stale_access_token: &str,
    now: i64,
) -> Result<(String, String)> {
    let lock = provider_refresh_lock("codex");
    let _guard = lock.lock().await;

    let blob = load_codex_blob(store)?;
    let current = blob
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("codex credentials are missing access_token"))?;
    if current != stale_access_token {
        return extract_access_and_account(&blob);
    }

    refresh_and_persist(store, client, issuer_base, &blob, now).await
}

/// Exchange the stored refresh token for a new access token and persist the
/// merged credential blob. Preserves `chatgpt_account_id` and any other
/// fields the provider has written alongside the OAuth tokens.
async fn refresh_and_persist(
    store: &CredentialStore,
    client: &Client,
    issuer_base: &str,
    blob: &Value,
    now: i64,
) -> Result<(String, String)> {
    let refresh_token = blob
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::Error::new(AuthenticationError {
                message: "codex credentials are missing refresh_token — run \
                          `mixer auth login codex`"
                    .to_string(),
            })
        })?;

    let tokens = refresh_codex_tokens_via_http(client, issuer_base, refresh_token).await?;

    let mut new_blob = blob.clone();
    let map = new_blob
        .as_object_mut()
        .ok_or_else(|| anyhow!("codex credentials blob is not an object"))?;
    map.insert("access_token".to_string(), json!(tokens.access_token));
    if let Some(rt) = &tokens.refresh_token {
        map.insert("refresh_token".to_string(), json!(rt));
    }
    if let Some(idt) = &tokens.id_token {
        map.insert("id_token".to_string(), json!(idt));
    }
    map.insert(
        "expires_at".to_string(),
        json!(compute_expires_at(now, tokens.expires_in)),
    );

    store.save("codex", &new_blob)?;
    extract_access_and_account(&new_blob)
}

async fn refresh_codex_tokens_via_http(
    client: &Client,
    issuer_base: &str,
    refresh_token: &str,
) -> Result<CodexTokens> {
    let url = format!("{}/oauth/token", issuer_base.trim_end_matches('/'));
    let form: [(&str, &str); 3] = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CLIENT_ID),
    ];

    let resp = client
        .post(&url)
        .form(&form)
        .send()
        .await
        .context("posting to codex /oauth/token (refresh)")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        if status.is_client_error() {
            // Revoked / invalid refresh token — credentials are dead and no
            // amount of retrying will help. Surface as authentication_error.
            return Err(anyhow::Error::new(AuthenticationError {
                message: "codex credentials expired — run `mixer auth login codex`".to_string(),
            }));
        }
        bail!("codex /oauth/token (refresh) returned {status}: {body}");
    }

    let parsed: OAuthTokenResp = resp
        .json()
        .await
        .context("parsing codex refresh-token response")?;
    Ok(CodexTokens {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        id_token: parsed.id_token,
        expires_in: parsed.expires_in,
    })
}

/// Fetch `/backend-api/wham/usage` and translate it to a [`UsageSnapshot`].
/// Refreshes the access token first if it's near expiry; a refresh failure
/// propagates so the caller can degrade to `Ok(None)`.
async fn fetch_codex_usage(
    store: &CredentialStore,
    settings: &ProviderSettings,
) -> Result<Option<UsageSnapshot>> {
    let client = build_codex_http_client(settings)?;
    let base = settings.base_url.as_deref().unwrap_or(CHATGPT_BACKEND);
    let url = format!("{}{USAGE_PATH}", base.trim_end_matches('/'));
    let issuer_base = auth_issuer(settings);
    let now = Utc::now().timestamp();

    let (access_token, account_id) =
        ensure_fresh_codex_token(store, &client, &issuer_base, now).await?;

    let resp = client
        .get(&url)
        .bearer_auth(&access_token)
        .header("chatgpt-account-id", &account_id)
        .header("Accept", "application/json")
        .send()
        .await
        .context("requesting codex usage endpoint")?;

    if !resp.status().is_success() {
        // 401/403/5xx here are uninteresting to the router. Swallow.
        return Ok(None);
    }

    let body: Value = resp.json().await.context("parsing codex usage response")?;
    Ok(extract_codex_usage_snapshot(&body))
}

/// Translate the `/wham/usage` response into a [`UsageSnapshot`]. Prefers
/// `secondary_window` (typically the weekly/monthly plan cap) over
/// `primary_window` (short rate-limit window) since plan-consumption routing
/// is the point of the `usage-aware` strategy. Returns `None` when neither
/// window carries a `used_percent`.
fn extract_codex_usage_snapshot(body: &Value) -> Option<UsageSnapshot> {
    let rl = body.get("rate_limit")?;
    let window = rl
        .get("secondary_window")
        .filter(|w| !w.is_null())
        .or_else(|| rl.get("primary_window").filter(|w| !w.is_null()))?;

    let used_percent = window.get("used_percent").and_then(Value::as_f64)?;
    let window_secs = window
        .get("limit_window_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let plan_type = body
        .get("plan_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let label = format!(
        "{:.1}% of {plan_type} plan used",
        used_percent.clamp(0.0, 100.0)
    );

    Some(UsageSnapshot {
        fraction_used: Some((used_percent / 100.0).clamp(0.0, 1.0)),
        window: codex_window_name(window_secs),
        label: Some(label),
    })
}

fn codex_window_name(secs: u64) -> String {
    const HOUR: u64 = 3600;
    const DAY: u64 = 24 * HOUR;
    match secs {
        0 => "window".to_string(),
        s if s < HOUR => "minutes".to_string(),
        s if s < DAY => "hourly".to_string(),
        s if s < 7 * DAY => "daily".to_string(),
        s if s < 30 * DAY => "weekly".to_string(),
        _ => "monthly".to_string(),
    }
}

fn load_codex_blob(store: &CredentialStore) -> Result<Value> {
    store
        .load_blob("codex")
        .with_context(|| "loading codex credentials")?
        .ok_or_else(|| anyhow!("codex is not authenticated; run `mixer auth login codex`"))
}

fn extract_access_and_account(blob: &Value) -> Result<(String, String)> {
    let access_token = blob
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("codex credentials are missing access_token"))?
        .to_string();
    let account_id = blob
        .get("chatgpt_account_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("codex credentials are missing chatgpt_account_id"))?
        .to_string();
    Ok((access_token, account_id))
}

fn build_codex_http_client(settings: &ProviderSettings) -> Result<Client> {
    let mut builder = Client::builder();
    if let Some(secs) = settings.request_timeout_secs {
        builder = builder.timeout(Duration::from_secs(secs));
    }
    builder.build().context("building reqwest client for codex")
}

#[derive(Debug, Deserialize)]
struct UserCodeResp {
    device_auth_id: String,
    user_code: String,
    /// The server sometimes returns `interval` as a string and sometimes as a
    /// number. Parse leniently.
    #[serde(default)]
    interval: Option<Value>,
    #[serde(default)]
    verification_uri: Option<String>,
    #[serde(default)]
    verification_uri_complete: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PollCodeResp {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResp {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

struct CodexTokens {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
}

async fn run_codex_device_flow(client: &Client, open_browser: bool) -> Result<CodexTokens> {
    let user_code = request_user_code(client).await?;

    let verification_uri = user_code
        .verification_uri_complete
        .clone()
        .or_else(|| user_code.verification_uri.clone())
        .unwrap_or_else(|| DEVICE_VERIFY_URL.to_string());

    eprintln!();
    eprintln!("To complete ChatGPT sign-in, open the following URL and enter the code:");
    eprintln!("  URL:  {verification_uri}");
    eprintln!("  Code: {}", user_code.user_code);
    eprintln!();

    if open_browser {
        let _ = open::that_detached(&verification_uri);
    }

    let interval = Duration::from_secs(parse_interval(user_code.interval.as_ref()));
    let poll = poll_for_authorization_code(
        client,
        &user_code.device_auth_id,
        &user_code.user_code,
        interval,
    )
    .await?;

    exchange_code_for_tokens(client, &poll).await
}

async fn request_user_code(client: &Client) -> Result<UserCodeResp> {
    let body = json!({ "client_id": CLIENT_ID });
    let resp = client
        .post(format!("{ISSUER}/api/accounts/deviceauth/usercode"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .context("requesting codex device authorization (usercode)")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("codex deviceauth/usercode returned {status}: {body}");
    }

    resp.json::<UserCodeResp>()
        .await
        .context("parsing codex deviceauth/usercode response")
}

async fn poll_for_authorization_code(
    client: &Client,
    device_auth_id: &str,
    user_code: &str,
    interval: Duration,
) -> Result<PollCodeResp> {
    let url = format!("{ISSUER}/api/accounts/deviceauth/token");
    let body = json!({ "device_auth_id": device_auth_id, "user_code": user_code }).to_string();

    loop {
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body.clone())
            .send()
            .await
            .context("polling codex deviceauth/token")?;

        if resp.status().is_success() {
            return resp
                .json::<PollCodeResp>()
                .await
                .context("parsing codex deviceauth/token success body");
        }

        // The codex usercode endpoint does not use RFC 8628 error codes — any
        // non-200 while the user has not yet approved simply means "try again
        // shortly". The server-side expiry window ends the loop via a distinct
        // error that `serde` will refuse to parse; we don't distinguish here.
        sleep(interval).await;
    }
}

async fn exchange_code_for_tokens(client: &Client, poll: &PollCodeResp) -> Result<CodexTokens> {
    let redirect_uri = format!("{ISSUER}/deviceauth/callback");
    let form: [(&str, &str); 5] = [
        ("grant_type", "authorization_code"),
        ("code", poll.authorization_code.as_str()),
        ("redirect_uri", redirect_uri.as_str()),
        ("client_id", CLIENT_ID),
        ("code_verifier", poll.code_verifier.as_str()),
    ];

    let resp = client
        .post(format!("{ISSUER}/oauth/token"))
        .form(&form)
        .send()
        .await
        .context("exchanging authorization_code for codex tokens")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("codex /oauth/token returned {status}: {body}");
    }

    let parsed: OAuthTokenResp = resp
        .json()
        .await
        .context("parsing codex /oauth/token response")?;
    Ok(CodexTokens {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        id_token: parsed.id_token,
        expires_in: parsed.expires_in,
    })
}

fn parse_interval(raw: Option<&Value>) -> u64 {
    let parsed = raw.and_then(|v| match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    });
    parsed.unwrap_or(5).max(1)
}

/// Seconds-from-now + `expires_in` (falls back to 1h when the server omits
/// the field, matching opencode's plugin/codex.ts).
fn compute_expires_at(now_secs: i64, expires_in: Option<u64>) -> i64 {
    now_secs + expires_in.unwrap_or(3600) as i64
}

/// Extract the `chatgpt_account_id` from an ID token JWT. Tries, in order:
///   1. top-level `chatgpt_account_id`
///   2. `https://api.openai.com/auth` → `chatgpt_account_id`
///   3. first entry of `organizations` → `id`
///
/// Matches the priority used by opencode's `extractAccountIdFromClaims`.
fn extract_chatgpt_account_id(id_token: &str) -> Result<String> {
    let claims = decode_jwt_claims(id_token)?;

    if let Some(id) = claims.get("chatgpt_account_id").and_then(Value::as_str) {
        return Ok(id.to_string());
    }
    if let Some(id) = claims
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
    {
        return Ok(id.to_string());
    }
    if let Some(id) = claims
        .get("organizations")
        .and_then(Value::as_array)
        .and_then(|orgs| orgs.first())
        .and_then(|org| org.get("id"))
        .and_then(Value::as_str)
    {
        return Ok(id.to_string());
    }
    bail!("id_token does not contain a chatgpt_account_id claim")
}

fn decode_jwt_claims(token: &str) -> Result<Value> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.insecure_disable_signature_validation();
    validation.validate_exp = false;
    validation.validate_nbf = false;
    validation.validate_aud = false;
    validation.required_spec_claims = HashSet::new();

    let data = decode::<Value>(token, &DecodingKey::from_secret(b"ignored"), &validation)
        .context("decoding id_token JWT claims")?;
    Ok(data.claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{EncodingKey, Header, encode};

    fn jwt_with(claims: &Value) -> String {
        encode(
            &Header::default(),
            claims,
            &EncodingKey::from_secret(b"test"),
        )
        .unwrap()
    }

    #[test]
    fn extracts_top_level_chatgpt_account_id() {
        let token = jwt_with(&json!({ "chatgpt_account_id": "acc_top" }));
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "acc_top");
    }

    #[test]
    fn extracts_chatgpt_account_id_from_namespaced_auth_claim() {
        let token = jwt_with(&json!({
            "https://api.openai.com/auth": { "chatgpt_account_id": "acc_nested" }
        }));
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "acc_nested");
    }

    #[test]
    fn falls_back_to_first_organization_id() {
        let token = jwt_with(&json!({
            "organizations": [
                { "id": "org_primary" },
                { "id": "org_secondary" }
            ]
        }));
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "org_primary");
    }

    #[test]
    fn top_level_wins_over_namespaced_and_orgs() {
        let token = jwt_with(&json!({
            "chatgpt_account_id": "acc_top",
            "https://api.openai.com/auth": { "chatgpt_account_id": "acc_nested" },
            "organizations": [{ "id": "org_primary" }]
        }));
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "acc_top");
    }

    #[test]
    fn missing_claim_returns_descriptive_error() {
        let token = jwt_with(&json!({ "sub": "user_123" }));
        let err = extract_chatgpt_account_id(&token).unwrap_err();
        assert!(
            err.to_string().contains("chatgpt_account_id"),
            "expected error to mention the missing claim, got: {err}"
        );
    }

    #[test]
    fn malformed_id_token_returns_error_not_panic() {
        let err = extract_chatgpt_account_id("not-a-jwt").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("id_token") || msg.contains("JWT"),
            "expected a JWT-decode error, got: {msg}"
        );
    }

    #[test]
    fn malformed_id_token_two_segments_returns_error_not_panic() {
        let err = extract_chatgpt_account_id("aaa.bbb").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("id_token") || msg.contains("JWT"),
            "expected a JWT-decode error, got: {msg}"
        );
    }

    #[test]
    fn compute_expires_at_uses_expires_in_when_present() {
        assert_eq!(compute_expires_at(1_000, Some(3_600)), 4_600);
    }

    #[test]
    fn compute_expires_at_defaults_to_one_hour() {
        assert_eq!(compute_expires_at(1_000, None), 4_600);
    }

    #[test]
    fn compute_expires_at_handles_zero_expires_in() {
        assert_eq!(compute_expires_at(1_000, Some(0)), 1_000);
    }

    #[test]
    fn parse_interval_accepts_numbers_and_strings() {
        assert_eq!(parse_interval(Some(&json!(7))), 7);
        assert_eq!(parse_interval(Some(&json!("3"))), 3);
        assert_eq!(parse_interval(None), 5);
        assert_eq!(parse_interval(Some(&json!("not a number"))), 5);
        // Zero is clamped to 1 so we don't busy-loop.
        assert_eq!(parse_interval(Some(&json!(0))), 1);
    }

    #[test]
    fn is_authenticated_true_when_blob_has_access_token() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        store
            .save(
                "codex",
                &json!({ "access_token": "at-live", "chatgpt_account_id": "acc" }),
            )
            .unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(CodexProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_false_when_access_token_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        store.save("codex", &json!({ "access_token": "" })).unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(!CodexProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_false_when_blob_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        let settings = ProviderSettings::default_enabled();
        assert!(!CodexProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_false_when_expired_and_no_refresh_token() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        let past = Utc::now().timestamp() - 3_600;
        store
            .save(
                "codex",
                &json!({
                    "access_token": "at-stale",
                    "chatgpt_account_id": "acc",
                    "expires_at": past,
                }),
            )
            .unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(!CodexProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_true_when_expired_but_refresh_token_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        let past = Utc::now().timestamp() - 3_600;
        store
            .save(
                "codex",
                &json!({
                    "access_token": "at-stale",
                    "refresh_token": "rt-live",
                    "chatgpt_account_id": "acc",
                    "expires_at": past,
                }),
            )
            .unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(CodexProvider.is_authenticated(&store, &settings));
    }

    // ── chat_completion integration tests ───────────────────────────────────

    use std::net::SocketAddr;
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    use crate::openai::ChatRequest;

    /// Spawn a minimal one-shot HTTP/1.1 server on 127.0.0.1:0. The server
    /// reads a single request (up to end-of-headers), stashes the captured
    /// `Authorization` and `chatgpt-account-id` headers for assertions, and
    /// writes `response_body_after_headers` as the HTTP response body.
    async fn spawn_mock_server(
        status_line: &'static str,
        content_type: &'static str,
        body: String,
        captured_auth: Arc<std::sync::Mutex<Option<String>>>,
        captured_account: Arc<std::sync::Mutex<Option<String>>>,
        started: oneshot::Sender<SocketAddr>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let _ = started.send(addr);

        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = Vec::with_capacity(8192);
        let mut tmp = [0u8; 2048];
        loop {
            let n = socket.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }

        let request = String::from_utf8_lossy(&buf).to_string();
        for line in request.lines() {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let name_l = name.trim().to_ascii_lowercase();
            let value = value.trim().to_string();
            match name_l.as_str() {
                "authorization" => *captured_auth.lock().unwrap() = Some(value),
                "chatgpt-account-id" => *captured_account.lock().unwrap() = Some(value),
                _ => {}
            }
        }

        let response = format!(
            "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        let _ = socket.shutdown().await;
    }

    fn write_creds(store: &CredentialStore) {
        store
            .save(
                "codex",
                &json!({
                    "access_token": "test-access-token",
                    "chatgpt_account_id": "acc_test",
                }),
            )
            .unwrap();
    }

    fn chat_request_for(model: &str) -> ChatRequest {
        serde_json::from_str(&format!(
            r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#
        ))
        .unwrap()
    }

    fn sse_body(payloads: &[&str]) -> String {
        payloads.iter().map(|p| format!("data: {p}\n\n")).collect()
    }

    #[tokio::test]
    async fn chat_completion_proxies_responses_sse_to_chunks() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        write_creds(&store);

        let auth = Arc::new(std::sync::Mutex::new(None));
        let account = Arc::new(std::sync::Mutex::new(None));
        let body = sse_body(&[
            r#"{"type":"response.created","response":{"id":"resp_mock","model":"gpt-5.2","created_at":100}}"#,
            r#"{"type":"response.output_text.delta","delta":"Hello "}"#,
            r#"{"type":"response.output_text.delta","delta":"from codex"}"#,
            r#"{"type":"response.completed","response":{"id":"resp_mock","usage":{"input_tokens":1,"output_tokens":2}}}"#,
        ]);

        let (tx, rx) = oneshot::channel();
        let auth_c = Arc::clone(&auth);
        let account_c = Arc::clone(&account);
        let server = tokio::spawn(async move {
            spawn_mock_server(
                "HTTP/1.1 200 OK",
                "text/event-stream",
                body,
                auth_c,
                account_c,
                tx,
            )
            .await;
        });
        let addr = rx.await.unwrap();

        let settings = ProviderSettings {
            base_url: Some(format!("http://{addr}")),
            ..ProviderSettings::default_enabled()
        };
        let req = chat_request_for("gpt-5.2");

        let stream = CodexProvider
            .chat_completion(&store, &settings, req)
            .await
            .expect("chat_completion returns a stream");
        let results: Vec<Result<_>> = stream.collect().await;
        server.await.unwrap();

        let chunks: Vec<_> = results
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("no stream errors");

        assert_eq!(
            auth.lock().unwrap().as_deref(),
            Some("Bearer test-access-token"),
            "Authorization header should be forwarded"
        );
        assert_eq!(
            account.lock().unwrap().as_deref(),
            Some("acc_test"),
            "chatgpt-account-id header should be forwarded"
        );

        let text: String = chunks
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        assert_eq!(text, "Hello from codex");

        let last = chunks.last().expect("at least one chunk");
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = last.usage.as_ref().expect("usage on final chunk");
        assert_eq!(usage["input_tokens"], 1);
        assert_eq!(usage["output_tokens"], 2);
    }

    #[tokio::test]
    async fn chat_completion_surfaces_upstream_error_body() {
        // 5xx responses skip the refresh-retry path (only 401 triggers it),
        // so this test exercises raw status + body surfacing.
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        write_creds(&store);

        let auth = Arc::new(std::sync::Mutex::new(None));
        let account = Arc::new(std::sync::Mutex::new(None));

        let (tx, rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            spawn_mock_server(
                "HTTP/1.1 500 Internal Server Error",
                "application/json",
                r#"{"error":{"message":"upstream oops","type":"server_error"}}"#.to_string(),
                auth,
                account,
                tx,
            )
            .await;
        });
        let addr = rx.await.unwrap();

        let settings = ProviderSettings {
            base_url: Some(format!("http://{addr}")),
            ..ProviderSettings::default_enabled()
        };

        let result = CodexProvider
            .chat_completion(&store, &settings, chat_request_for("gpt-5.2"))
            .await;
        server.await.unwrap();
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("5xx should surface as an error, got Ok stream"),
        };

        let msg = format!("{err:#}");
        assert!(msg.contains("500"), "error should include status: {msg}");
        assert!(
            msg.contains("upstream oops"),
            "error should include body snippet: {msg}"
        );
    }

    #[tokio::test]
    async fn chat_completion_errors_when_credentials_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        // no credentials written

        let settings = ProviderSettings::default_enabled();
        let result = CodexProvider
            .chat_completion(&store, &settings, chat_request_for("gpt-5.2"))
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(
            err.to_string().contains("not authenticated"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn chat_completion_errors_when_account_id_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        store
            .save("codex", &json!({ "access_token": "only-token" }))
            .unwrap();

        let settings = ProviderSettings::default_enabled();
        let result = CodexProvider
            .chat_completion(&store, &settings, chat_request_for("gpt-5.2"))
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing account id should error"),
        };
        assert!(
            err.to_string().contains("chatgpt_account_id"),
            "unexpected error: {err:#}"
        );
    }

    // ── OAuth refresh tests ─────────────────────────────────────────────────

    mod refresh {
        use super::*;
        use std::sync::Mutex as StdMutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use axum::{
            Form, Router, body::Body, extract::State, http::HeaderMap, response::Response,
            routing::post,
        };
        use serde::Deserialize;

        enum ChatPlan {
            AlwaysOk,
            UnauthorizedThenOk,
            AlwaysUnauthorized,
        }

        enum RefreshPlan {
            NewAccessToken {
                access_token: String,
                expires_in: u64,
            },
            Unauthorized,
        }

        struct MockState {
            refresh_calls: AtomicUsize,
            chat_calls: AtomicUsize,
            chat_auth_tokens: StdMutex<Vec<String>>,
            chat_plan: StdMutex<ChatPlan>,
            refresh_plan: StdMutex<RefreshPlan>,
            chat_body: String,
        }

        #[derive(Deserialize)]
        struct RefreshForm {
            grant_type: String,
        }

        async fn refresh_handler(
            State(state): State<Arc<MockState>>,
            Form(form): Form<RefreshForm>,
        ) -> Response {
            state.refresh_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(
                form.grant_type, "refresh_token",
                "refresh endpoint expects grant_type=refresh_token"
            );

            let plan = state.refresh_plan.lock().unwrap();
            match &*plan {
                RefreshPlan::NewAccessToken {
                    access_token,
                    expires_in,
                } => {
                    let body = json!({
                        "access_token": access_token,
                        "refresh_token": "new-refresh",
                        "token_type": "Bearer",
                        "expires_in": expires_in,
                    })
                    .to_string();
                    Response::builder()
                        .status(200)
                        .header("Content-Type", "application/json")
                        .body(Body::from(body))
                        .unwrap()
                }
                RefreshPlan::Unauthorized => Response::builder()
                    .status(401)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"invalid_grant"}"#))
                    .unwrap(),
            }
        }

        async fn chat_handler(
            State(state): State<Arc<MockState>>,
            headers: HeaderMap,
            _body: Body,
        ) -> Response {
            let auth = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            state.chat_auth_tokens.lock().unwrap().push(auth);
            let call_n = state.chat_calls.fetch_add(1, Ordering::SeqCst);

            let unauthorized = || {
                Response::builder()
                    .status(401)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"error":"token expired"}"#))
                    .unwrap()
            };
            let ok = |body: String| {
                Response::builder()
                    .status(200)
                    .header("Content-Type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            };

            let plan = state.chat_plan.lock().unwrap();
            match &*plan {
                ChatPlan::AlwaysOk => ok(state.chat_body.clone()),
                ChatPlan::UnauthorizedThenOk if call_n == 0 => unauthorized(),
                ChatPlan::UnauthorizedThenOk => ok(state.chat_body.clone()),
                ChatPlan::AlwaysUnauthorized => unauthorized(),
            }
        }

        async fn start_axum_mock(state: Arc<MockState>) -> SocketAddr {
            let app = Router::new()
                .route("/oauth/token", post(refresh_handler))
                .route("/backend-api/codex/responses", post(chat_handler))
                .with_state(state);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            addr
        }

        fn default_sse_body() -> String {
            sse_body(&[
                r#"{"type":"response.created","response":{"id":"resp_mock","model":"gpt-5.2","created_at":100}}"#,
                r#"{"type":"response.output_text.delta","delta":"hi"}"#,
                r#"{"type":"response.completed","response":{"id":"resp_mock"}}"#,
            ])
        }

        fn write_creds_with_expiry(store: &CredentialStore, expires_at: i64) {
            store
                .save(
                    "codex",
                    &json!({
                        "access_token": "old-access",
                        "refresh_token": "stored-refresh",
                        "chatgpt_account_id": "acc_test",
                        "expires_at": expires_at,
                    }),
                )
                .unwrap();
        }

        fn mock_state(chat: ChatPlan, refresh: RefreshPlan) -> Arc<MockState> {
            Arc::new(MockState {
                refresh_calls: AtomicUsize::new(0),
                chat_calls: AtomicUsize::new(0),
                chat_auth_tokens: StdMutex::new(Vec::new()),
                chat_plan: StdMutex::new(chat),
                refresh_plan: StdMutex::new(refresh),
                chat_body: default_sse_body(),
            })
        }

        fn settings_for(addr: SocketAddr) -> ProviderSettings {
            ProviderSettings {
                base_url: Some(format!("http://{addr}")),
                ..ProviderSettings::default_enabled()
            }
        }

        #[tokio::test]
        async fn near_expiry_triggers_proactive_refresh_before_dispatch() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let near = Utc::now().timestamp() + 30;
            write_creds_with_expiry(&store, near);

            let mock = mock_state(
                ChatPlan::AlwaysOk,
                RefreshPlan::NewAccessToken {
                    access_token: "refreshed-access".to_string(),
                    expires_in: 3600,
                },
            );
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let stream = CodexProvider
                .chat_completion(&store, &settings_for(addr), chat_request_for("gpt-5.2"))
                .await
                .expect("chat_completion should succeed");
            let chunks: Vec<_> = stream.collect().await;
            for r in &chunks {
                assert!(r.is_ok(), "stream should not error: {r:?}");
            }

            assert_eq!(
                mock.refresh_calls.load(Ordering::SeqCst),
                1,
                "refresh endpoint should be called exactly once"
            );
            assert_eq!(
                mock.chat_calls.load(Ordering::SeqCst),
                1,
                "chat endpoint should be called exactly once"
            );
            let tokens = mock.chat_auth_tokens.lock().unwrap();
            assert_eq!(
                tokens[0], "Bearer refreshed-access",
                "chat should be dispatched with the refreshed token"
            );

            let blob = store.load_blob("codex").unwrap().unwrap();
            assert_eq!(blob["access_token"], "refreshed-access");
            assert_eq!(blob["refresh_token"], "new-refresh");
            assert_eq!(
                blob["chatgpt_account_id"], "acc_test",
                "account id should be preserved across refresh"
            );
        }

        #[tokio::test]
        async fn dispatch_401_triggers_single_refresh_and_retry() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let far = Utc::now().timestamp() + 100_000;
            write_creds_with_expiry(&store, far);

            let mock = mock_state(
                ChatPlan::UnauthorizedThenOk,
                RefreshPlan::NewAccessToken {
                    access_token: "post-401-access".to_string(),
                    expires_in: 3600,
                },
            );
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let stream = CodexProvider
                .chat_completion(&store, &settings_for(addr), chat_request_for("gpt-5.2"))
                .await
                .expect("chat_completion should succeed after retry");
            let _: Vec<_> = stream.collect().await;

            assert_eq!(
                mock.refresh_calls.load(Ordering::SeqCst),
                1,
                "exactly one refresh after a 401"
            );
            assert_eq!(
                mock.chat_calls.load(Ordering::SeqCst),
                2,
                "chat called twice: first 401, then success"
            );
            let tokens = mock.chat_auth_tokens.lock().unwrap();
            assert_eq!(tokens[0], "Bearer old-access");
            assert_eq!(tokens[1], "Bearer post-401-access");
        }

        #[tokio::test]
        async fn second_401_after_retry_surfaces_as_upstream_error() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let far = Utc::now().timestamp() + 100_000;
            write_creds_with_expiry(&store, far);

            let mock = mock_state(
                ChatPlan::AlwaysUnauthorized,
                RefreshPlan::NewAccessToken {
                    access_token: "new-access".to_string(),
                    expires_in: 3600,
                },
            );
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let result = CodexProvider
                .chat_completion(&store, &settings_for(addr), chat_request_for("gpt-5.2"))
                .await;

            assert_eq!(
                mock.chat_calls.load(Ordering::SeqCst),
                2,
                "chat should be called exactly twice (no third try)"
            );
            assert_eq!(
                mock.refresh_calls.load(Ordering::SeqCst),
                1,
                "refresh should be called exactly once"
            );

            let err = match result {
                Err(e) => e,
                Ok(_) => panic!("a second 401 should surface as an error"),
            };
            assert!(
                err.downcast_ref::<AuthenticationError>().is_none(),
                "persistent upstream 401 is an upstream error, not an authentication_error"
            );
            let msg = format!("{err:#}");
            assert!(msg.contains("401"), "error should mention status: {msg}");
        }

        #[tokio::test]
        async fn concurrent_requests_issue_single_refresh() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let near = Utc::now().timestamp() + 30;
            write_creds_with_expiry(&store, near);

            let mock = mock_state(
                ChatPlan::AlwaysOk,
                RefreshPlan::NewAccessToken {
                    access_token: "concurrent-access".to_string(),
                    expires_in: 3600,
                },
            );
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let store = Arc::new(store);
            let settings = Arc::new(settings_for(addr));

            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let store = Arc::clone(&store);
                    let settings = Arc::clone(&settings);
                    tokio::spawn(async move {
                        let stream = CodexProvider
                            .chat_completion(&store, &settings, chat_request_for("gpt-5.2"))
                            .await
                            .expect("chat_completion should succeed");
                        let _: Vec<_> = stream.collect().await;
                    })
                })
                .collect();
            for h in handles {
                h.await.unwrap();
            }

            assert_eq!(
                mock.refresh_calls.load(Ordering::SeqCst),
                1,
                "mutex should serialize refresh to a single HTTP call across 8 concurrent requests"
            );
            assert_eq!(
                mock.chat_calls.load(Ordering::SeqCst),
                8,
                "all eight chat requests should complete"
            );
            let tokens = mock.chat_auth_tokens.lock().unwrap();
            for (i, t) in tokens.iter().enumerate() {
                assert_eq!(
                    t, "Bearer concurrent-access",
                    "chat call #{i} should use the refreshed token"
                );
            }
        }

        #[tokio::test]
        async fn refresh_endpoint_rejection_surfaces_as_authentication_error() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let expired = Utc::now().timestamp() - 100;
            write_creds_with_expiry(&store, expired);

            let mock = mock_state(ChatPlan::AlwaysOk, RefreshPlan::Unauthorized);
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let result = CodexProvider
                .chat_completion(&store, &settings_for(addr), chat_request_for("gpt-5.2"))
                .await;

            let err = match result {
                Err(e) => e,
                Ok(_) => panic!("refresh rejection should propagate as error"),
            };
            let auth_err = err
                .downcast_ref::<AuthenticationError>()
                .expect("refresh rejection should downcast to AuthenticationError");
            assert!(
                auth_err.message.contains("mixer auth login codex"),
                "error message should include the login command: {}",
                auth_err.message
            );

            assert_eq!(mock.refresh_calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                mock.chat_calls.load(Ordering::SeqCst),
                0,
                "chat should not be dispatched when the proactive refresh fails"
            );
        }

        #[tokio::test]
        async fn dispatch_401_without_refresh_token_surfaces_as_authentication_error() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            let far = Utc::now().timestamp() + 100_000;
            // Credentials with no refresh_token — a 401 cannot be recovered.
            store
                .save(
                    "codex",
                    &json!({
                        "access_token": "no-refresh-access",
                        "chatgpt_account_id": "acc_test",
                        "expires_at": far,
                    }),
                )
                .unwrap();

            let mock = mock_state(
                ChatPlan::AlwaysUnauthorized,
                RefreshPlan::NewAccessToken {
                    access_token: "unused".to_string(),
                    expires_in: 3600,
                },
            );
            let addr = start_axum_mock(Arc::clone(&mock)).await;

            let result = CodexProvider
                .chat_completion(&store, &settings_for(addr), chat_request_for("gpt-5.2"))
                .await;

            let err = match result {
                Err(e) => e,
                Ok(_) => panic!("401 without refresh_token should error"),
            };
            let auth_err = err
                .downcast_ref::<AuthenticationError>()
                .expect("should be AuthenticationError");
            assert!(auth_err.message.contains("mixer auth login codex"));
            assert_eq!(mock.refresh_calls.load(Ordering::SeqCst), 0);
            assert_eq!(mock.chat_calls.load(Ordering::SeqCst), 1);
        }
    }

    // ── usage() tests ───────────────────────────────────────────────────────

    mod usage {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use axum::{
            Router, body::Body, extract::State, http::HeaderMap, response::Response, routing::get,
        };

        #[test]
        fn extract_snapshot_prefers_secondary_window() {
            let body = json!({
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 42,
                        "limit_window_seconds": 3600,
                    },
                    "secondary_window": {
                        "used_percent": 5.5,
                        "limit_window_seconds": 7 * 86400,
                    }
                }
            });
            let snap = extract_codex_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.055));
            assert_eq!(snap.window, "weekly");
            let label = snap.label.unwrap();
            assert!(label.contains("pro"), "label mentions plan: {label}");
            assert!(label.contains("5.5"), "label mentions percent: {label}");
        }

        #[test]
        fn extract_snapshot_falls_back_to_primary_when_secondary_absent() {
            let body = json!({
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 73,
                        "limit_window_seconds": 3600,
                    }
                }
            });
            let snap = extract_codex_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.73));
            assert_eq!(snap.window, "hourly");
        }

        #[test]
        fn extract_snapshot_falls_back_to_primary_when_secondary_null() {
            let body = json!({
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": { "used_percent": 20 },
                    "secondary_window": null
                }
            });
            let snap = extract_codex_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.2));
        }

        #[test]
        fn extract_snapshot_returns_none_without_rate_limit() {
            assert!(extract_codex_usage_snapshot(&json!({ "plan_type": "pro" })).is_none());
        }

        #[test]
        fn extract_snapshot_returns_none_without_used_percent() {
            let body = json!({
                "rate_limit": {
                    "primary_window": { "limit_window_seconds": 3600 }
                }
            });
            assert!(extract_codex_usage_snapshot(&body).is_none());
        }

        #[test]
        fn extract_snapshot_clamps_out_of_range_percent() {
            let body = json!({
                "rate_limit": {
                    "primary_window": { "used_percent": 150, "limit_window_seconds": 3600 }
                }
            });
            let snap = extract_codex_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(1.0));
        }

        #[test]
        fn window_name_maps_seconds_to_label() {
            assert_eq!(codex_window_name(0), "window");
            assert_eq!(codex_window_name(60), "minutes");
            assert_eq!(codex_window_name(3600), "hourly");
            assert_eq!(codex_window_name(86_400), "daily");
            assert_eq!(codex_window_name(7 * 86_400), "weekly");
            assert_eq!(codex_window_name(60 * 86_400), "monthly");
        }

        // ── mocked /wham/usage ──────────────────────────────────────────────

        struct UsageMock {
            calls: AtomicUsize,
            auth_tokens: StdMutex<Vec<String>>,
            account_ids: StdMutex<Vec<String>>,
            status: u16,
            body: String,
        }

        use std::sync::Mutex as StdMutex;

        async fn usage_handler(
            State(state): State<Arc<UsageMock>>,
            headers: HeaderMap,
        ) -> Response {
            state.calls.fetch_add(1, Ordering::SeqCst);
            let auth = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            let account = headers
                .get("chatgpt-account-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            state.auth_tokens.lock().unwrap().push(auth);
            state.account_ids.lock().unwrap().push(account);
            Response::builder()
                .status(state.status)
                .header("Content-Type", "application/json")
                .body(Body::from(state.body.clone()))
                .unwrap()
        }

        async fn start_usage_mock(state: Arc<UsageMock>) -> SocketAddr {
            let app = Router::new()
                .route("/backend-api/wham/usage", get(usage_handler))
                .with_state(state);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            addr
        }

        fn write_live_creds(store: &CredentialStore) {
            let far = Utc::now().timestamp() + 100_000;
            store
                .save(
                    "codex",
                    &json!({
                        "access_token": "live-access",
                        "refresh_token": "stored-refresh",
                        "chatgpt_account_id": "acc_test",
                        "expires_at": far,
                    }),
                )
                .unwrap();
        }

        #[tokio::test]
        async fn usage_returns_snapshot_with_bearer_and_account_headers() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            write_live_creds(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_tokens: StdMutex::new(Vec::new()),
                account_ids: StdMutex::new(Vec::new()),
                status: 200,
                body: json!({
                    "plan_type": "pro",
                    "rate_limit": {
                        "primary_window": {
                            "used_percent": 10,
                            "limit_window_seconds": 18_000
                        },
                        "secondary_window": {
                            "used_percent": 25,
                            "limit_window_seconds": 7 * 86_400
                        }
                    }
                })
                .to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = ProviderSettings {
                base_url: Some(format!("http://{addr}")),
                ..ProviderSettings::default_enabled()
            };

            let snap = CodexProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds")
                .expect("snapshot present");

            assert_eq!(snap.fraction_used, Some(0.25));
            assert_eq!(snap.window, "weekly");
            assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                mock.auth_tokens.lock().unwrap()[0],
                "Bearer live-access",
                "usage probe must forward the access token"
            );
            assert_eq!(
                mock.account_ids.lock().unwrap()[0],
                "acc_test",
                "usage probe must forward the account id"
            );
        }

        #[tokio::test]
        async fn usage_returns_none_on_upstream_error() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            write_live_creds(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_tokens: StdMutex::new(Vec::new()),
                account_ids: StdMutex::new(Vec::new()),
                status: 500,
                body: r#"{"error":"oops"}"#.to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = ProviderSettings {
                base_url: Some(format!("http://{addr}")),
                ..ProviderSettings::default_enabled()
            };

            let snap = CodexProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "5xx should degrade to Ok(None)");
            assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        }

        #[tokio::test]
        async fn usage_returns_none_when_credentials_missing() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            // no credentials written
            let settings = ProviderSettings::default_enabled();
            let snap = CodexProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none());
        }

        #[tokio::test]
        async fn usage_returns_none_when_body_unparseable() {
            let tmp = tempfile::TempDir::new().unwrap();
            let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
            write_live_creds(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_tokens: StdMutex::new(Vec::new()),
                account_ids: StdMutex::new(Vec::new()),
                status: 200,
                body: "not-json".to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = ProviderSettings {
                base_url: Some(format!("http://{addr}")),
                ..ProviderSettings::default_enabled()
            };

            let snap = CodexProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "malformed body should degrade to Ok(None)");
        }
    }
}
