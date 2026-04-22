//! Kimi Code subscription provider.
//!
//! Subscribers authenticate via OAuth 2.0 Device Authorization Grant
//! (RFC 8628) against `auth.kimi.com`. The resulting access token is then
//! used as `Authorization: Bearer <token>` against the OpenAI-compatible
//! Chat Completions endpoint at `api.kimi.com/coding/v1/chat/completions`.
//!
//! Reference implementation: `MoonshotAI/kimi-cli` — specifically
//! `src/kimi_cli/auth/oauth.py` (device flow + refresh) and
//! `src/kimi_cli/auth/platforms.py` (API base URL).
//!
//! The chat wire format is OpenAI-compatible, so this provider reuses
//! `openai_client::chat_completion`. The device flow is standard RFC 8628,
//! so it reuses `auth::device_flow::run_device_flow`. Token refresh is
//! handled inline (parallel to `codex.rs`) because the refresh grant is a
//! plain `POST /api/oauth/token` with `grant_type=refresh_token` — there is
//! no shared HTTP helper for that yet.
//!
//! Usage reporting: Kimi Code enforces a 5-hour quota window but does not
//! expose a public introspection endpoint. `usage()` returns `None`, so the
//! usage-aware router treats this backend at a neutral 0.5 weight. If an
//! endpoint becomes available later, add it without changing the trait.

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::device_flow::{DeviceFlowConfig, run_device_flow};
use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::models_list::fetch_openai_models;
use crate::providers::common::oauth_refresh::{
    AuthenticationError, EXPIRY_THRESHOLD_SECS, OauthFreshness, is_near_expiry, oauth_freshness,
    provider_refresh_lock,
};
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{
    AuthKind, ChatStream, ModelInfo, Provider, ReasoningFormat, RemoteModelEntry,
};
use crate::usage::UsageSnapshot;

/// OAuth client id published by `kimi-cli` (`KIMI_CODE_CLIENT_ID` in
/// `src/kimi_cli/auth/oauth.py`). Identifies this integration as a device-
/// flow client; not a secret.
const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const AUTH_HOST: &str = "https://auth.kimi.com";
const DEVICE_AUTH_PATH: &str = "/api/oauth/device_authorization";
const TOKEN_PATH: &str = "/api/oauth/token";
const API_BASE: &str = "https://api.kimi.com/coding/v1";
const CHAT_PATH: &str = "/chat/completions";
const MODELS_PATH: &str = "/models";

/// Kimi's coding endpoint sniffs `User-Agent` and returns
/// `403 access_terminated_error` unless the header looks like one of their
/// approved Coding Agent clients. Matches the format `kimi-cli` itself sends
/// (`KimiCLI/<version>`, see `src/kimi_cli/constant.py::get_user_agent`).
/// Bump when Kimi starts rejecting this specific version.
const USER_AGENT: &str = "KimiCLI/1.37.0";

pub struct KimiCodeProvider;

#[async_trait]
impl Provider for KimiCodeProvider {
    fn id(&self) -> &'static str {
        "kimi-code"
    }

    fn display_name(&self) -> &'static str {
        "Kimi Code (subscription)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        // The Kimi Code subscription gateway advertises exactly one model id
        // via its live `/v1/models` endpoint: `kimi-for-coding`. The gateway
        // routes internally to whichever underlying K2 variant is current;
        // the specific K2.x ids are only addressable via the Moonshot
        // pay-per-token endpoint (see `kimi_api.rs`).
        vec![
            ModelInfo::new("kimi-for-coding", "Kimi for Coding", false, 256_000)
                .with_reasoning(ReasoningFormat::Structured),
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
        match oauth_freshness(&blob, Utc::now().timestamp()) {
            OauthFreshness::Valid | OauthFreshness::ExpiredRefreshable => true,
            OauthFreshness::ExpiredDead => false,
        }
    }

    async fn login(&self, store: &CredentialStore) -> Result<()> {
        let client = Client::new();
        let cfg = DeviceFlowConfig {
            client_id: CLIENT_ID.to_string(),
            scopes: Vec::new(),
            device_authorization_url: format!("{AUTH_HOST}{DEVICE_AUTH_PATH}"),
            token_url: format!("{AUTH_HOST}{TOKEN_PATH}"),
            audience: None,
            extra_params: Vec::new(),
            open_browser: true,
        };

        let tokens = run_device_flow(&cfg, &client).await?;
        let refresh_token = tokens
            .refresh_token
            .as_deref()
            .ok_or_else(|| anyhow!("kimi-code token response is missing refresh_token"))?;
        let expires_at = compute_expires_at(Utc::now().timestamp(), tokens.expires_in);

        let mut blob = json!({
            "access_token": tokens.access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        });
        if let Some(scope) = tokens.scope.as_deref() {
            blob["scope"] = json!(scope);
        }
        if !tokens.token_type.is_empty() {
            blob["token_type"] = json!(tokens.token_type);
        }
        store.save(self.id(), &blob)?;

        tracing::info!(provider = "kimi-code", "signed in");
        Ok(())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        mut req: ChatRequest,
    ) -> Result<ChatStream> {
        let refresh_client = build_kimi_http_client(settings)?;
        let base = settings.base_url.as_deref().unwrap_or(API_BASE);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let token_url = token_endpoint(settings);
        let now = Utc::now().timestamp();
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);

        inject_omitted_reasoning_on_assistant_tool_calls(&mut req);

        let access_token = ensure_fresh_kimi_token(store, &refresh_client, &token_url, now).await?;

        match openai_client::chat_completion(
            self.id(),
            &url,
            &access_token,
            AuthScheme::Bearer,
            timeout,
            Some(USER_AGENT),
            req.clone(),
        )
        .await
        {
            Ok(stream) => Ok(stream),
            Err(e) => {
                if is_retryable_auth_error(&e) {
                    let fresh = force_refresh_kimi_token(
                        store,
                        &refresh_client,
                        &token_url,
                        &access_token,
                        now,
                    )
                    .await?;
                    openai_client::chat_completion(
                        self.id(),
                        &url,
                        &fresh,
                        AuthScheme::Bearer,
                        timeout,
                        Some(USER_AGENT),
                        req,
                    )
                    .await
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn usage(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(None)
    }

    async fn list_remote_models(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<Vec<RemoteModelEntry>>> {
        // Mirrors chat_completion: refresh the OAuth access token (if near
        // expiry) before hitting the upstream. Unlike chat, the models
        // endpoint is idempotent, so a 401 here is conclusive — we surface
        // it rather than retrying.
        let refresh_client = build_kimi_http_client(settings)?;
        let base = settings.base_url.as_deref().unwrap_or(API_BASE);
        let url = format!("{}{MODELS_PATH}", base.trim_end_matches('/'));
        let token_url = token_endpoint(settings);
        let now = Utc::now().timestamp();
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);

        let access_token = ensure_fresh_kimi_token(store, &refresh_client, &token_url, now).await?;
        let entries = fetch_openai_models(
            self.id(),
            &url,
            &access_token,
            AuthScheme::Bearer,
            timeout,
            Some(USER_AGENT),
        )
        .await?;
        Ok(Some(entries))
    }
}

/// A dispatch error triggers a refresh-and-retry only when the shim
/// surfaced it as [`AuthenticationError`] (upstream 401). Everything else
/// (5xx, network) stays as-is — refreshing the access token wouldn't help.
fn is_retryable_auth_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<AuthenticationError>().is_some()
}

/// Kimi Code's coding gateway runs every request in "thinking" mode and
/// rejects assistant tool-call messages that lack `reasoning_content` with
/// `400 invalid_request_error` ("thinking is enabled but reasoning_content is
/// missing in assistant tool call message at index N"). Mixer routes across
/// heterogeneous backends, so a tool-call turn produced by a different
/// provider — or by Kimi itself via a client that doesn't round-trip the
/// field (e.g. opencode) — can flow back to Kimi without `reasoning_content`.
/// The original thought is not recoverable here. Kimi appears to treat an empty
/// string as missing, so use a short non-empty marker rather than fabricating a
/// plausible-looking rationale.
const OMITTED_REASONING_MARKER: &str = "reasoning omitted";

fn inject_omitted_reasoning_on_assistant_tool_calls(req: &mut ChatRequest) {
    for msg in req.messages.iter_mut() {
        let missing_or_blank_reasoning = msg
            .reasoning_content
            .as_deref()
            .is_none_or(|s| s.trim().is_empty());
        if msg.role == "assistant" && msg.tool_calls.is_some() && missing_or_blank_reasoning {
            msg.reasoning_content = Some(OMITTED_REASONING_MARKER.to_string());
        }
    }
}

fn token_endpoint(settings: &ProviderSettings) -> String {
    #[cfg(test)]
    if let Some(base) = settings.base_url.as_deref() {
        return format!("{}{TOKEN_PATH}", base.trim_end_matches('/'));
    }
    let _ = settings;
    format!("{AUTH_HOST}{TOKEN_PATH}")
}

async fn ensure_fresh_kimi_token(
    store: &CredentialStore,
    client: &Client,
    token_url: &str,
    now: i64,
) -> Result<String> {
    let blob = load_kimi_blob(store)?;
    if !is_near_expiry(&blob, now, EXPIRY_THRESHOLD_SECS) {
        return extract_access_token(&blob);
    }

    let lock = provider_refresh_lock("kimi-code");
    let _guard = lock.lock().await;

    let blob = load_kimi_blob(store)?;
    if !is_near_expiry(&blob, now, EXPIRY_THRESHOLD_SECS) {
        return extract_access_token(&blob);
    }

    refresh_and_persist(store, client, token_url, &blob, now).await
}

async fn force_refresh_kimi_token(
    store: &CredentialStore,
    client: &Client,
    token_url: &str,
    stale_access_token: &str,
    now: i64,
) -> Result<String> {
    let lock = provider_refresh_lock("kimi-code");
    let _guard = lock.lock().await;

    let blob = load_kimi_blob(store)?;
    let current = blob
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("kimi-code credentials are missing access_token"))?;
    if current != stale_access_token {
        return extract_access_token(&blob);
    }

    refresh_and_persist(store, client, token_url, &blob, now).await
}

async fn refresh_and_persist(
    store: &CredentialStore,
    client: &Client,
    token_url: &str,
    blob: &Value,
    now: i64,
) -> Result<String> {
    let refresh_token = blob
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::Error::new(AuthenticationError {
                message: "kimi-code credentials are missing refresh_token — run \
                          `mixer auth login kimi-code`"
                    .to_string(),
            })
        })?;

    let tokens = refresh_kimi_tokens_via_http(client, token_url, refresh_token).await?;

    let mut new_blob = blob.clone();
    let map = new_blob
        .as_object_mut()
        .ok_or_else(|| anyhow!("kimi-code credentials blob is not an object"))?;
    map.insert("access_token".to_string(), json!(tokens.access_token));
    if let Some(rt) = &tokens.refresh_token {
        map.insert("refresh_token".to_string(), json!(rt));
    }
    map.insert(
        "expires_at".to_string(),
        json!(compute_expires_at(now, tokens.expires_in)),
    );
    if let Some(scope) = &tokens.scope {
        map.insert("scope".to_string(), json!(scope));
    }

    store.save("kimi-code", &new_blob)?;
    extract_access_token(&new_blob)
}

async fn refresh_kimi_tokens_via_http(
    client: &Client,
    token_url: &str,
    refresh_token: &str,
) -> Result<KimiTokens> {
    let form: [(&str, &str); 3] = [
        ("client_id", CLIENT_ID),
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
    ];

    let resp = client
        .post(token_url)
        .form(&form)
        .send()
        .await
        .context("posting to kimi-code /api/oauth/token (refresh)")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        if status.is_client_error() {
            return Err(anyhow::Error::new(AuthenticationError {
                message: "kimi-code credentials expired — run `mixer auth login kimi-code`"
                    .to_string(),
            }));
        }
        bail!("kimi-code /api/oauth/token (refresh) returned {status}: {body}");
    }

    let parsed: OAuthTokenResp = resp
        .json()
        .await
        .context("parsing kimi-code refresh-token response")?;
    Ok(KimiTokens {
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token,
        expires_in: parsed.expires_in,
        scope: parsed.scope,
    })
}

fn load_kimi_blob(store: &CredentialStore) -> Result<Value> {
    store
        .load_blob("kimi-code")
        .context("loading kimi-code credentials")?
        .ok_or_else(|| anyhow!("kimi-code is not authenticated; run `mixer auth login kimi-code`"))
}

fn extract_access_token(blob: &Value) -> Result<String> {
    blob.get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("kimi-code credentials are missing access_token"))
}

fn build_kimi_http_client(settings: &ProviderSettings) -> Result<Client> {
    let mut builder = Client::builder();
    if let Some(secs) = settings.request_timeout_secs {
        builder = builder.timeout(Duration::from_secs(secs));
    }
    builder
        .build()
        .context("building reqwest client for kimi-code")
}

/// Seconds-from-now + `expires_in` (falls back to 1h when the server omits
/// the field, matching the codex provider).
fn compute_expires_at(now_secs: i64, expires_in: Option<u64>) -> i64 {
    now_secs + expires_in.unwrap_or(3600) as i64
}

struct KimiTokens {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResp {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    scope: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatRequest;
    use crate::providers::common::openai_client::test_support::{sample_request, start_mock_chat};
    use futures::StreamExt;
    use serde_json::json;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    use axum::{
        Form, Router, body::Body, extract::State, http::HeaderMap, response::Response,
        routing::post,
    };
    use tokio::net::TcpListener;

    fn store_in(tmp: &TempDir) -> CredentialStore {
        CredentialStore::with_dir_for_tests(tmp.path().to_path_buf())
    }

    fn settings_with_base(base_url: &str) -> ProviderSettings {
        ProviderSettings {
            base_url: Some(base_url.to_string()),
            ..ProviderSettings::default_enabled()
        }
    }

    fn sse(payloads: &[&str]) -> String {
        payloads.iter().map(|p| format!("data: {p}\n\n")).collect()
    }

    fn write_live_creds(store: &CredentialStore) {
        let far = Utc::now().timestamp() + 100_000;
        store
            .save(
                "kimi-code",
                &json!({
                    "access_token": "live-access",
                    "refresh_token": "stored-refresh",
                    "expires_at": far,
                }),
            )
            .unwrap();
    }

    // ── is_authenticated tests ──────────────────────────────────────────────

    #[test]
    fn is_authenticated_true_with_valid_token() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        write_live_creds(&store);
        let settings = ProviderSettings::default_enabled();
        assert!(KimiCodeProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_false_when_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        assert!(!KimiCodeProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_true_when_expired_but_refresh_token_present() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let past = Utc::now().timestamp() - 3_600;
        store
            .save(
                "kimi-code",
                &json!({
                    "access_token": "stale",
                    "refresh_token": "rt",
                    "expires_at": past,
                }),
            )
            .unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(KimiCodeProvider.is_authenticated(&store, &settings));
    }

    #[test]
    fn is_authenticated_false_when_expired_and_no_refresh_token() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let past = Utc::now().timestamp() - 3_600;
        store
            .save(
                "kimi-code",
                &json!({ "access_token": "stale", "expires_at": past }),
            )
            .unwrap();
        let settings = ProviderSettings::default_enabled();
        assert!(!KimiCodeProvider.is_authenticated(&store, &settings));
    }

    // ── chat_completion (no refresh) ─────────────────────────────────────────

    #[tokio::test]
    async fn chat_completion_proxies_sse_with_bearer_auth() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        write_live_creds(&store);

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        let stream = KimiCodeProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("no stream errors");
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer live-access"),
        );
    }

    #[tokio::test]
    async fn forwards_omitted_reasoning_on_assistant_tool_call_messages() {
        // Regression: Kimi Code's coding gateway rejects assistant tool-call
        // messages that lack `reasoning_content` with
        // `400: "thinking is enabled but reasoning_content is missing in
        // assistant tool call message at index N"`. Mixer must inject an
        // non-empty `reasoning_content` on those messages before forwarding,
        // regardless of whether the earlier turn was produced by a different
        // provider in the pool or by a client that dropped the field on
        // replay (e.g. opencode).
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        write_live_creds(&store);

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"kimi-for-coding","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        // Transcript shape: [user, assistant(text), assistant(tool_call, no
        // reasoning_content), tool, user].
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "kimi-for-coding",
                "messages": [
                    {"role": "user", "content": "what's the weather?"},
                    {"role": "assistant", "content": "let me check"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{}"}}
                        ]
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                    {"role": "user", "content": "thanks"}
                ]
            }"#,
        )
        .unwrap();

        let _ = KimiCodeProvider
            .chat_completion(&store, &settings, req)
            .await
            .unwrap();

        let forwarded: serde_json::Value = serde_json::from_str(&mock.captured.body()).unwrap();
        let msgs = forwarded["messages"].as_array().unwrap();

        // Index 2 is the assistant tool-call message: reasoning_content must
        // be present and non-empty so Kimi's presence check passes.
        assert_eq!(
            msgs[2]["reasoning_content"], OMITTED_REASONING_MARKER,
            "assistant tool-call message must carry reasoning_content",
        );
        // Messages that already had the field or don't need it must not be
        // touched.
        assert!(msgs[0].get("reasoning_content").is_none());
        assert!(msgs[1].get("reasoning_content").is_none());
        assert!(msgs[3].get("reasoning_content").is_none());
        assert!(msgs[4].get("reasoning_content").is_none());
    }

    #[tokio::test]
    async fn preserves_existing_reasoning_content_on_assistant_tool_calls() {
        // If the client DOES round-trip reasoning_content, the sanitizer
        // must leave it alone — overwriting a real thought with the omitted
        // marker would destroy chain-of-thought the model actually needs.
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        write_live_creds(&store);

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"kimi-for-coding","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "kimi-for-coding",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": null,
                        "reasoning_content": "prior thought",
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}
                        ]
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"}
                ]
            }"#,
        )
        .unwrap();

        let _ = KimiCodeProvider
            .chat_completion(&store, &settings, req)
            .await
            .unwrap();

        let forwarded: serde_json::Value = serde_json::from_str(&mock.captured.body()).unwrap();
        assert_eq!(
            forwarded["messages"][1]["reasoning_content"],
            "prior thought",
        );
    }

    #[test]
    fn inject_omitted_reasoning_only_targets_assistant_tool_calls() {
        let mut req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "kimi-for-coding",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "plain reply"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {"id": "c", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": null,
                        "reasoning_content": "",
                        "tool_calls": [
                            {"id": "blank", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}
                        ]
                    },
                    {"role": "tool", "tool_call_id": "c", "content": "result"}
                ]
            }"#,
        )
        .unwrap();

        inject_omitted_reasoning_on_assistant_tool_calls(&mut req);

        assert!(req.messages[0].reasoning_content.is_none(), "user");
        assert!(
            req.messages[1].reasoning_content.is_none(),
            "assistant without tool_calls",
        );
        assert_eq!(
            req.messages[2].reasoning_content.as_deref(),
            Some(OMITTED_REASONING_MARKER),
            "assistant with tool_calls must be filled",
        );
        assert_eq!(
            req.messages[3].reasoning_content.as_deref(),
            Some(OMITTED_REASONING_MARKER),
            "blank assistant tool-call reasoning must be replaced",
        );
        assert!(req.messages[4].reasoning_content.is_none(), "tool");
    }

    #[tokio::test]
    async fn chat_completion_errors_when_credentials_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let result = KimiCodeProvider
            .chat_completion(&store, &settings, sample_request())
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

    // ── refresh flow (combined chat + token endpoints) ──────────────────────

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
        refresh_form_params: StdMutex<Vec<(String, String)>>,
        chat_plan: StdMutex<ChatPlan>,
        refresh_plan: StdMutex<RefreshPlan>,
        chat_body: String,
    }

    #[derive(serde::Deserialize)]
    struct RefreshForm {
        grant_type: String,
        #[serde(default)]
        client_id: Option<String>,
        #[serde(default)]
        refresh_token: Option<String>,
    }

    async fn refresh_handler(
        State(state): State<Arc<MockState>>,
        Form(form): Form<RefreshForm>,
    ) -> Response {
        state.refresh_calls.fetch_add(1, Ordering::SeqCst);
        let mut params = state.refresh_form_params.lock().unwrap();
        params.push(("grant_type".to_string(), form.grant_type.clone()));
        if let Some(c) = form.client_id.clone() {
            params.push(("client_id".to_string(), c));
        }
        if let Some(r) = form.refresh_token.clone() {
            params.push(("refresh_token".to_string(), r));
        }
        drop(params);

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
                    "scope": "kimi-code",
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

    async fn start_combined_mock(state: Arc<MockState>) -> SocketAddr {
        let app = Router::new()
            .route(TOKEN_PATH, post(refresh_handler))
            .route("/chat/completions", post(chat_handler))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        addr
    }

    fn default_sse_body() -> String {
        sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ])
    }

    fn write_creds_with_expiry(store: &CredentialStore, expires_at: i64) {
        store
            .save(
                "kimi-code",
                &json!({
                    "access_token": "old-access",
                    "refresh_token": "stored-refresh",
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
            refresh_form_params: StdMutex::new(Vec::new()),
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

    fn chat_request_for(model: &str) -> ChatRequest {
        serde_json::from_str(&format!(
            r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#
        ))
        .unwrap()
    }

    #[tokio::test]
    async fn near_expiry_triggers_proactive_refresh_before_dispatch() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let near = Utc::now().timestamp() + 30;
        write_creds_with_expiry(&store, near);

        let mock = mock_state(
            ChatPlan::AlwaysOk,
            RefreshPlan::NewAccessToken {
                access_token: "refreshed-access".to_string(),
                expires_in: 3600,
            },
        );
        let addr = start_combined_mock(Arc::clone(&mock)).await;

        let stream = KimiCodeProvider
            .chat_completion(&store, &settings_for(addr), chat_request_for("kimi-k2.6"))
            .await
            .expect("chat_completion should succeed");
        let chunks: Vec<_> = stream.collect().await;
        for r in &chunks {
            assert!(r.is_ok(), "stream should not error: {r:?}");
        }

        assert_eq!(mock.refresh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(mock.chat_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            mock.chat_auth_tokens.lock().unwrap()[0],
            "Bearer refreshed-access",
        );

        let params = mock.refresh_form_params.lock().unwrap();
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "grant_type" && v == "refresh_token"),
            "refresh must use grant_type=refresh_token",
        );
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "client_id" && v == CLIENT_ID),
            "refresh must send client_id",
        );

        let blob = store.load_blob("kimi-code").unwrap().unwrap();
        assert_eq!(blob["access_token"], "refreshed-access");
        assert_eq!(blob["refresh_token"], "new-refresh");
    }

    #[tokio::test]
    async fn concurrent_requests_issue_single_refresh() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let near = Utc::now().timestamp() + 30;
        write_creds_with_expiry(&store, near);

        let mock = mock_state(
            ChatPlan::AlwaysOk,
            RefreshPlan::NewAccessToken {
                access_token: "concurrent-access".to_string(),
                expires_in: 3600,
            },
        );
        let addr = start_combined_mock(Arc::clone(&mock)).await;

        let store = Arc::new(store);
        let settings = Arc::new(settings_for(addr));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                let settings = Arc::clone(&settings);
                tokio::spawn(async move {
                    let stream = KimiCodeProvider
                        .chat_completion(&store, &settings, chat_request_for("kimi-k2.6"))
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
            "mutex should serialize refresh across 8 concurrent requests"
        );
        assert_eq!(mock.chat_calls.load(Ordering::SeqCst), 8);
    }

    #[tokio::test]
    async fn refresh_rejection_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let expired = Utc::now().timestamp() - 100;
        write_creds_with_expiry(&store, expired);

        let mock = mock_state(ChatPlan::AlwaysOk, RefreshPlan::Unauthorized);
        let addr = start_combined_mock(Arc::clone(&mock)).await;

        let result = KimiCodeProvider
            .chat_completion(&store, &settings_for(addr), chat_request_for("kimi-k2.6"))
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("refresh rejection should propagate as error"),
        };
        let auth_err = err
            .downcast_ref::<AuthenticationError>()
            .expect("refresh rejection should downcast to AuthenticationError");
        assert!(
            auth_err.message.contains("mixer auth login kimi-code"),
            "message should include login command: {}",
            auth_err.message
        );
        assert_eq!(mock.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn dispatch_401_triggers_single_refresh_and_retry() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let far = Utc::now().timestamp() + 100_000;
        write_creds_with_expiry(&store, far);

        let mock = mock_state(
            ChatPlan::UnauthorizedThenOk,
            RefreshPlan::NewAccessToken {
                access_token: "post-401-access".to_string(),
                expires_in: 3600,
            },
        );
        let addr = start_combined_mock(Arc::clone(&mock)).await;

        let stream = KimiCodeProvider
            .chat_completion(&store, &settings_for(addr), chat_request_for("kimi-k2.6"))
            .await
            .expect("chat_completion should succeed after retry");
        let _: Vec<_> = stream.collect().await;

        assert_eq!(
            mock.refresh_calls.load(Ordering::SeqCst),
            1,
            "one refresh after the 401"
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
    async fn second_401_after_retry_surfaces_as_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let far = Utc::now().timestamp() + 100_000;
        write_creds_with_expiry(&store, far);

        let mock = mock_state(
            ChatPlan::AlwaysUnauthorized,
            RefreshPlan::NewAccessToken {
                access_token: "new-access".to_string(),
                expires_in: 3600,
            },
        );
        let addr = start_combined_mock(Arc::clone(&mock)).await;

        let result = KimiCodeProvider
            .chat_completion(&store, &settings_for(addr), chat_request_for("kimi-k2.6"))
            .await;

        assert_eq!(mock.chat_calls.load(Ordering::SeqCst), 2);
        assert_eq!(mock.refresh_calls.load(Ordering::SeqCst), 1);
        assert!(result.is_err());
    }

    // ── list_remote_models ──────────────────────────────────────────────────

    /// Proves the OAuth refresh path fires before the models call, and that
    /// the refreshed bearer — not the stored stale token — is sent to the
    /// upstream /v1/models endpoint.
    #[tokio::test]
    async fn list_remote_models_refreshes_near_expiry_then_queries_upstream() {
        use axum::{
            Form, Router,
            body::Body,
            extract::State,
            http::HeaderMap,
            response::Response,
            routing::{get, post},
        };
        use tokio::net::TcpListener;

        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let near = Utc::now().timestamp() + 30;
        write_creds_with_expiry(&store, near);

        struct State0 {
            refresh_calls: AtomicUsize,
            models_calls: AtomicUsize,
            models_auth: StdMutex<Option<String>>,
        }

        #[derive(serde::Deserialize)]
        struct RefreshForm {
            grant_type: String,
        }

        async fn refresh_handler(
            State(s): State<Arc<State0>>,
            Form(form): Form<RefreshForm>,
        ) -> Response {
            s.refresh_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(form.grant_type, "refresh_token");
            let body = json!({
                "access_token": "refreshed-for-models",
                "refresh_token": "new-rt",
                "token_type": "Bearer",
                "expires_in": 3600u64,
            })
            .to_string();
            Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap()
        }

        async fn models_handler(State(s): State<Arc<State0>>, headers: HeaderMap) -> Response {
            s.models_calls.fetch_add(1, Ordering::SeqCst);
            *s.models_auth.lock().unwrap() = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = json!({
                "object": "list",
                "data": [
                    {"id": "kimi-k2.6"},
                    {"id": "kimi-k2.5"},
                    {"id": "kimi-k2-thinking"},
                    {"id": "kimi-future"}
                ]
            })
            .to_string();
            Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(Body::from(body))
                .unwrap()
        }

        let state = Arc::new(State0 {
            refresh_calls: AtomicUsize::new(0),
            models_calls: AtomicUsize::new(0),
            models_auth: StdMutex::new(None),
        });
        let app = Router::new()
            .route(TOKEN_PATH, post(refresh_handler))
            .route("/models", get(models_handler))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let settings = settings_for(addr);

        let entries = KimiCodeProvider
            .list_remote_models(&store, &settings)
            .await
            .expect("call succeeds")
            .expect("kimi-code supports remote listing");

        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].id, "kimi-k2.6");
        assert_eq!(
            state.refresh_calls.load(Ordering::SeqCst),
            1,
            "near-expiry token must be refreshed before the models call",
        );
        assert_eq!(state.models_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            state.models_auth.lock().unwrap().as_deref(),
            Some("Bearer refreshed-for-models"),
            "models call must use the refreshed token, not the stored stale one",
        );
    }

    // ── compute_expires_at ──────────────────────────────────────────────────

    #[test]
    fn compute_expires_at_uses_expires_in_when_present() {
        assert_eq!(compute_expires_at(1_000, Some(3_600)), 4_600);
    }

    #[test]
    fn compute_expires_at_defaults_to_one_hour() {
        assert_eq!(compute_expires_at(1_000, None), 4_600);
    }
}
