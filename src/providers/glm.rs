//! GLM (z.ai) subscription provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `https://api.z.ai/api/paas/v4` (override via
//! `providers.glm.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`.
//!
//! Login stores the pasted API key at `~/.config/mixer/credentials/glm.json`
//! (0600). At dispatch time the key resolves env-var-first via
//! [`CredentialStore::load_api_key`] so an environment-provided key wins over
//! anything on disk (see plan.md §3.6.2).
//!
//! # Usage probe — UNOFFICIAL
//!
//! z.ai does **not** document a public quota-introspection endpoint. The
//! `/api/monitor/usage/quota/limit` path we query here is the same one the
//! z.ai web dashboard calls, reverse-engineered by community tools (e.g.
//! `guyinwonder168/opencode-glm-quota`). It is therefore subject to
//! undocumented changes at any time — response shape, auth header conventions,
//! even URL — with no upstream compatibility guarantees.
//!
//! Two consequences of that:
//!   * Authentication uses a bare `Authorization: <token>` header (no `Bearer`
//!     prefix), matching what the dashboard actually sends. `Authorization:
//!     Bearer <token>` 401s.
//!   * Any failure — non-2xx, shape drift, missing credentials — degrades
//!     silently to `Ok(None)` so the router falls back to a neutral weight
//!     rather than failing routing entirely.
//!
//! If z.ai publishes an official quota API, migrate to that and remove the
//! caveats here. Tracking anchor: no upstream issue yet; the plugin's README
//! carries the same "undocumented" warning.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::api_key_login::prompt_and_store_api_key;
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider};
use crate::usage::UsageSnapshot;

const DEFAULT_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
const CHAT_PATH: &str = "/chat/completions";

/// Default host for the undocumented quota endpoint. Lives on `api.z.ai`
/// (root, not `/api/paas/v4`), so it can't share `DEFAULT_BASE_URL`.
/// `settings.base_url` overrides this (primarily so tests can mock the
/// endpoint). In production, a user who points `base_url` at the CN mirror
/// (`https://open.bigmodel.cn`) would have their usage probe hit
/// `https://open.bigmodel.cn/api/monitor/usage/quota/limit`, which is the
/// correct mirror path.
const USAGE_BASE_URL: &str = "https://api.z.ai";
const USAGE_QUOTA_PATH: &str = "/api/monitor/usage/quota/limit";

pub struct GlmProvider;

#[async_trait]
impl Provider for GlmProvider {
    fn id(&self) -> &'static str {
        "glm"
    }

    fn display_name(&self) -> &'static str {
        "GLM (z.ai)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("glm-4.6", "GLM 4.6", false, 200_000),
            ModelInfo::new("glm-4.5v", "GLM 4.5V", true, 64_000),
        ]
    }

    fn auth_kind(&self) -> AuthKind {
        AuthKind::ApiKey
    }

    fn is_authenticated(&self, store: &CredentialStore, settings: &ProviderSettings) -> bool {
        store
            .load_api_key(self.id(), settings)
            .is_some_and(|s| !s.is_empty())
    }

    async fn login(&self, store: &CredentialStore) -> Result<()> {
        eprintln!("Generate a z.ai API key at https://z.ai/manage-apikey/apikey-list");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    /// Best-effort Coding Plan consumption via the **undocumented**
    /// `/api/monitor/usage/quota/limit` endpoint (see the module docs). Any
    /// failure — missing API key, non-2xx, response shape drift — degrades to
    /// `Ok(None)` so routing keeps working at a neutral weight.
    async fn usage(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(fetch_glm_usage(store, settings).await.ok().flatten())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let api_key = store
            .load_api_key(self.id(), settings)
            .ok_or_else(|| anyhow!("glm is not authenticated; run `mixer auth login glm`"))?;
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        openai_client::chat_completion(self.id(), &url, &api_key, AuthScheme::Bearer, timeout, req)
            .await
    }
}

/// Fetch the undocumented quota endpoint and translate it to a
/// [`UsageSnapshot`]. The caller wraps any error into `Ok(None)`.
///
/// Note the non-Bearer auth: z.ai's dashboard sends the raw token as the
/// `Authorization` header value, and the endpoint rejects `Bearer <token>`.
async fn fetch_glm_usage(
    store: &CredentialStore,
    settings: &ProviderSettings,
) -> Result<Option<UsageSnapshot>> {
    let Some(api_key) = store.load_api_key("glm", settings) else {
        return Ok(None);
    };
    let base = settings.base_url.as_deref().unwrap_or(USAGE_BASE_URL);
    let url = format!("{}{USAGE_QUOTA_PATH}", base.trim_end_matches('/'));

    let mut builder = Client::builder();
    if let Some(secs) = settings.request_timeout_secs {
        builder = builder.timeout(Duration::from_secs(secs));
    }
    let client = builder.build().context("building reqwest client for glm")?;

    let resp = client
        .get(&url)
        .header("Authorization", api_key.as_str())
        .header("Accept", "application/json")
        .header("Accept-Language", "en-US,en")
        .send()
        .await
        .context("requesting glm usage endpoint")?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let body: Value = resp.json().await.context("parsing glm usage response")?;
    Ok(extract_glm_usage_snapshot(&body))
}

/// Translate a `/api/monitor/usage/quota/limit` body into a [`UsageSnapshot`].
///
/// Response shape observed in the wild:
/// ```json
/// {
///   "code": 200, "msg": "...",
///   "data": {
///     "level": "pro",
///     "limits": [
///       {"type": "TIME_LIMIT",   "percentage": 7},
///       {"type": "TOKENS_LIMIT", "percentage": 53}
///     ]
///   }
/// }
/// ```
///
/// `TIME_LIMIT` is the short 5-hour rolling cap; `TOKENS_LIMIT` entries are
/// the longer-window caps (weekly/plan-lifetime). We pick the entry with the
/// highest `percentage` to drive routing, because the tightest-consumed
/// bucket is what will rate-limit the user next.
///
/// Returns `None` when the body doesn't match this shape at all — that
/// typically means z.ai changed the schema and the feature has drifted;
/// `usage()` will treat the provider as "unknown consumption" until someone
/// updates the parser.
fn extract_glm_usage_snapshot(body: &Value) -> Option<UsageSnapshot> {
    let data = body.get("data")?;
    let limits = data.get("limits")?.as_array()?;
    let level = data
        .get("level")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let mut worst: Option<(f64, &str)> = None;
    for entry in limits {
        let pct = entry.get("percentage").and_then(Value::as_f64)?;
        let kind = entry.get("type").and_then(Value::as_str).unwrap_or("");
        if worst.is_none_or(|(w, _)| pct > w) {
            worst = Some((pct, kind));
        }
    }
    let (pct, kind) = worst?;
    let fraction = (pct / 100.0).clamp(0.0, 1.0);
    let window = match kind {
        "TIME_LIMIT" => "hourly",
        "TOKENS_LIMIT" => "weekly",
        _ => "window",
    }
    .to_string();
    Some(UsageSnapshot {
        fraction_used: Some(fraction),
        window,
        label: Some(format!("{pct:.1}% of {level} plan used")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::common::oauth_refresh::AuthenticationError;
    use crate::providers::common::openai_client::test_support::{sample_request, start_mock_chat};
    use futures::StreamExt;
    use serde_json::json;
    use tempfile::TempDir;

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

    #[tokio::test]
    async fn chat_completion_proxies_through_base_url_with_bearer_auth() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("glm", &json!({ "api_key": "sk-glm" })).unwrap();

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"glm-4.6","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let stream = GlmProvider
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
            Some("Bearer sk-glm"),
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("glm", &json!({ "api_key": "wrong" })).unwrap();
        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"unauthorized"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = GlmProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be AuthenticationError");
        assert!(auth.message.contains("mixer auth login glm"));
    }

    #[tokio::test]
    async fn usage_returns_none_without_credentials() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let snap = GlmProvider.usage(&store, &settings).await.unwrap();
        assert!(snap.is_none(), "no api key → Ok(None)");
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let result = GlmProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }

    // ── usage() tests ───────────────────────────────────────────────────────

    mod usage {
        use super::*;
        use axum::{
            Router, body::Body, extract::State, http::HeaderMap, response::Response, routing::get,
        };
        use std::net::SocketAddr;
        use std::sync::Arc;
        use std::sync::Mutex as StdMutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::net::TcpListener;

        #[test]
        fn extract_snapshot_picks_highest_percentage() {
            let body = json!({
                "code": 200,
                "data": {
                    "level": "pro",
                    "limits": [
                        { "type": "TIME_LIMIT",   "percentage": 7 },
                        { "type": "TOKENS_LIMIT", "percentage": 53 },
                        { "type": "TOKENS_LIMIT", "percentage": 44 },
                    ]
                }
            });
            let snap = extract_glm_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.53));
            assert_eq!(snap.window, "weekly");
            let label = snap.label.unwrap();
            assert!(label.contains("pro"), "label mentions plan: {label}");
            assert!(label.contains("53"), "label mentions percent: {label}");
        }

        #[test]
        fn extract_snapshot_labels_time_limit_hourly() {
            let body = json!({
                "data": {
                    "level": "lite",
                    "limits": [
                        { "type": "TIME_LIMIT", "percentage": 90 }
                    ]
                }
            });
            let snap = extract_glm_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.window, "hourly");
        }

        #[test]
        fn extract_snapshot_clamps_percentage_above_100() {
            let body = json!({
                "data": {
                    "limits": [
                        { "type": "TOKENS_LIMIT", "percentage": 150 }
                    ]
                }
            });
            let snap = extract_glm_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(1.0));
        }

        #[test]
        fn extract_snapshot_returns_none_without_data() {
            assert!(extract_glm_usage_snapshot(&json!({ "code": 200 })).is_none());
        }

        #[test]
        fn extract_snapshot_returns_none_without_limits_array() {
            assert!(extract_glm_usage_snapshot(&json!({ "data": { "level": "pro" } })).is_none());
        }

        #[test]
        fn extract_snapshot_returns_none_when_percentage_missing() {
            let body = json!({
                "data": { "limits": [ { "type": "TIME_LIMIT" } ] }
            });
            assert!(extract_glm_usage_snapshot(&body).is_none());
        }

        // ── mocked /api/monitor/usage/quota/limit ──────────────────────────

        struct UsageMock {
            calls: AtomicUsize,
            auth_headers: StdMutex<Vec<String>>,
            status: u16,
            body: String,
        }

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
            state.auth_headers.lock().unwrap().push(auth);
            Response::builder()
                .status(state.status)
                .header("Content-Type", "application/json")
                .body(Body::from(state.body.clone()))
                .unwrap()
        }

        async fn start_usage_mock(state: Arc<UsageMock>) -> SocketAddr {
            let app = Router::new()
                .route(USAGE_QUOTA_PATH, get(usage_handler))
                .with_state(state);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            addr
        }

        fn write_key(store: &CredentialStore) {
            store
                .save("glm", &json!({ "api_key": "zai-test-token" }))
                .unwrap();
        }

        #[tokio::test]
        async fn usage_sends_raw_authorization_header_not_bearer() {
            // This is the critical wire-format invariant. The z.ai dashboard
            // sends the token raw, and the endpoint 401s on `Bearer <token>`.
            // Regressing this to `bearer_auth` would silently break the probe.
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_headers: StdMutex::new(Vec::new()),
                status: 200,
                body: json!({
                    "data": {
                        "level": "pro",
                        "limits": [{ "type": "TOKENS_LIMIT", "percentage": 40 }]
                    }
                })
                .to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = GlmProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds")
                .expect("snapshot present");

            assert_eq!(snap.fraction_used, Some(0.4));
            assert_eq!(
                mock.auth_headers.lock().unwrap()[0],
                "zai-test-token",
                "z.ai dashboard sends token raw, not `Bearer <token>`",
            );
        }

        #[tokio::test]
        async fn usage_returns_none_on_upstream_error() {
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_headers: StdMutex::new(Vec::new()),
                status: 500,
                body: r#"{"error":"oops"}"#.to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = GlmProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "5xx should degrade to Ok(None)");
            assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        }

        #[tokio::test]
        async fn usage_returns_none_when_body_unparseable() {
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_headers: StdMutex::new(Vec::new()),
                status: 200,
                body: "not-json".to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = GlmProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "malformed body should degrade to Ok(None)");
        }

        #[tokio::test]
        async fn usage_returns_none_when_body_shape_drifts() {
            // Protects the "silent fallback on shape drift" contract: if z.ai
            // renames `limits` or nests it differently, we must land on
            // Ok(None), not panic or error.
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_headers: StdMutex::new(Vec::new()),
                status: 200,
                body: json!({
                    "data": { "level": "pro", "usage": { "percent": 42 } }
                })
                .to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = GlmProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "shape drift must degrade to Ok(None)");
        }
    }
}
