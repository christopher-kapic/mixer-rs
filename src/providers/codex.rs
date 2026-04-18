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
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::responses_api::{
    chat_request_to_responses_body, responses_sse_to_chat_chunks,
};
use crate::providers::{ChatStream, ModelInfo, Provider};

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
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let (access_token, account_id) = load_codex_auth(store, self.id())?;

        let model = req.model.clone();
        let body = chat_request_to_responses_body(&req, &model);

        let base = settings.base_url.as_deref().unwrap_or(CHATGPT_BACKEND);
        let url = format!("{}{RESPONSES_PATH}", base.trim_end_matches('/'));

        let client = build_codex_http_client(settings)?;

        let resp = client
            .post(&url)
            .bearer_auth(&access_token)
            .header("chatgpt-account-id", &account_id)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .context("posting to codex responses endpoint")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(1024).collect();
            bail!("codex responses API returned {status}: {snippet}");
        }

        let event_stream = resp
            .bytes_stream()
            .eventsource()
            .map(|r| r.map_err(|e| anyhow!("codex SSE stream error: {e}")));
        let chunks = responses_sse_to_chat_chunks(event_stream);
        Ok(Box::pin(chunks))
    }
}

fn load_codex_auth(store: &CredentialStore, provider_id: &str) -> Result<(String, String)> {
    let blob = store
        .load_blob(provider_id)
        .with_context(|| "loading codex credentials")?
        .ok_or_else(|| anyhow!("codex is not authenticated; run `mixer auth login codex`"))?;
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
        let tmp = tempfile::TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        write_creds(&store);

        let auth = Arc::new(std::sync::Mutex::new(None));
        let account = Arc::new(std::sync::Mutex::new(None));

        let (tx, rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            spawn_mock_server(
                "HTTP/1.1 401 Unauthorized",
                "application/json",
                r#"{"error":{"message":"token expired","type":"invalid_request_error"}}"#
                    .to_string(),
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
            Ok(_) => panic!("401 should surface as an error, got Ok stream"),
        };

        let msg = format!("{err:#}");
        assert!(msg.contains("401"), "error should include status: {msg}");
        assert!(
            msg.contains("token expired"),
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
}
