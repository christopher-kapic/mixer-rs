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
use futures::stream;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::{ChatStream, ModelInfo, Provider};

/// OAuth client identifier published by the Codex CLI. Matches the value in
/// `codex-rs/login/src/auth/manager.rs` and opencode's plugin/codex.ts.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER: &str = "https://auth.openai.com";
/// User-facing verification URL shown when the server omits one in the
/// usercode response (matches the `{base_url}/codex/device` value in
/// `codex-rs/login/src/device_code_auth.rs`).
const DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";

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
        vec![
            ModelInfo {
                id: "gpt-5.2",
                display_name: "GPT-5.2",
                supports_images: true,
            },
            ModelInfo {
                id: "gpt-5.2-mini",
                display_name: "GPT-5.2 mini",
                supports_images: true,
            },
        ]
    }

    fn is_authenticated(&self, store: &CredentialStore) -> bool {
        match store.load_blob(self.id()) {
            Ok(Some(blob)) => blob
                .get("access_token")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty()),
            _ => false,
        }
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

        eprintln!("codex: signed in (account {account_id})");
        Ok(())
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatStream> {
        Ok(Box::pin(stream::once(async {
            Err(anyhow!("codex chat_completion not yet implemented"))
        })))
    }
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
        assert!(CodexProvider.is_authenticated(&store));
    }

    #[test]
    fn is_authenticated_false_when_access_token_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        store.save("codex", &json!({ "access_token": "" })).unwrap();
        assert!(!CodexProvider.is_authenticated(&store));
    }

    #[test]
    fn is_authenticated_false_when_blob_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        assert!(!CodexProvider.is_authenticated(&store));
    }
}
