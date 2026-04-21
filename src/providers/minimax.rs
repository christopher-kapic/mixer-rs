//! Minimax subscription provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `https://api.minimax.chat/v1` (override via
//! `providers.minimax.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`.
//!
//! Login stores the pasted API key at `~/.config/mixer/credentials/minimax.json`
//! with permissions 0600. At dispatch time the key resolves env-var-first via
//! [`CredentialStore::load_api_key`] so an environment-provided key wins over
//! anything on disk (see plan.md §3.6.2).
//!
//! Usage reporting: the Coding Plan `remains` endpoint at
//! `https://api.minimax.io/v1/api/openplatform/coding_plan/remains` is public
//! and Bearer-authenticated. See `fetch_minimax_usage` for details. Users on a
//! Minimax Token Plan (not the Coding Plan) will fall through to `Ok(None)`;
//! that endpoint's response shape is not documented, so we don't parse it.

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

const DEFAULT_BASE_URL: &str = "https://api.minimax.chat/v1";
const CHAT_PATH: &str = "/chat/completions";

/// Default host for the Coding Plan `remains` endpoint. Minimax serves plan
/// introspection from `api.minimax.io` rather than the chat host, so we keep
/// it separate from `DEFAULT_BASE_URL`. An override in
/// `settings.base_url` wins (primarily so tests can point everything at a
/// mock); when a user overrides this in production to a host that doesn't
/// expose `/v1/api/openplatform/coding_plan/remains`, the probe 404s and
/// `usage()` degrades to `Ok(None)` — routing keeps working at a neutral
/// weight instead of failing.
const USAGE_BASE_URL: &str = "https://api.minimax.io/v1";
const USAGE_CODING_PLAN_PATH: &str = "/api/openplatform/coding_plan/remains";

pub struct MinimaxProvider;

#[async_trait]
impl Provider for MinimaxProvider {
    fn id(&self) -> &'static str {
        "minimax"
    }

    fn display_name(&self) -> &'static str {
        "Minimax"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("MiniMax-M2", "MiniMax M2", false, 192_000),
            ModelInfo::new("MiniMax-M2-vl", "MiniMax M2 (vision)", true, 192_000),
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
        eprintln!("Generate a Minimax API key at https://platform.minimaxi.com/");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    /// Best-effort Coding Plan consumption, via
    /// `GET /v1/api/openplatform/coding_plan/remains`. Any failure — missing
    /// API key, non-2xx response, malformed body, or simply a Token-Plan
    /// account (which uses a different, undocumented endpoint) — degrades to
    /// `Ok(None)` so routing treats this provider as "unknown consumption"
    /// rather than erroring.
    async fn usage(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(fetch_minimax_usage(store, settings).await.ok().flatten())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let api_key = store.load_api_key(self.id(), settings).ok_or_else(|| {
            anyhow!("minimax is not authenticated; run `mixer auth login minimax`")
        })?;
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        openai_client::chat_completion(self.id(), &url, &api_key, AuthScheme::Bearer, timeout, req)
            .await
    }
}

/// Fetch the Coding Plan `remains` endpoint and translate it to a
/// [`UsageSnapshot`]. The caller wraps any error into `Ok(None)`.
async fn fetch_minimax_usage(
    store: &CredentialStore,
    settings: &ProviderSettings,
) -> Result<Option<UsageSnapshot>> {
    let Some(api_key) = store.load_api_key("minimax", settings) else {
        return Ok(None);
    };
    let base = settings.base_url.as_deref().unwrap_or(USAGE_BASE_URL);
    let url = format!("{}{USAGE_CODING_PLAN_PATH}", base.trim_end_matches('/'));

    let mut builder = Client::builder();
    if let Some(secs) = settings.request_timeout_secs {
        builder = builder.timeout(Duration::from_secs(secs));
    }
    let client = builder
        .build()
        .context("building reqwest client for minimax")?;

    let resp = client
        .get(&url)
        .bearer_auth(&api_key)
        .header("Accept", "application/json")
        .send()
        .await
        .context("requesting minimax usage endpoint")?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let body: Value = resp
        .json()
        .await
        .context("parsing minimax usage response")?;
    Ok(extract_minimax_usage_snapshot(&body))
}

/// Translate a Coding Plan `remains` body into a [`UsageSnapshot`]. Picks the
/// most-consumed bucket across `modelRemains[]` since that's what will rate-
/// limit the user first; for routing we want the provider's tightest
/// constraint, not an average. Returns `None` when the body shape is
/// unrecognisable (e.g. Token-Plan response) or no entry carries both counts.
fn extract_minimax_usage_snapshot(body: &Value) -> Option<UsageSnapshot> {
    let models = body.get("modelRemains")?.as_array()?;
    let mut worst: Option<(f64, u64)> = None;
    for m in models {
        let total = m.get("currentIntervalTotalCount").and_then(Value::as_u64)?;
        if total == 0 {
            continue;
        }
        let remaining = m
            .get("currentIntervalRemainingCount")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(total);
        let used = (total - remaining) as f64 / total as f64;
        let window_secs = match (
            m.get("startTime").and_then(Value::as_i64),
            m.get("endTime").and_then(Value::as_i64),
        ) {
            (Some(s), Some(e)) if e > s => (e - s) as u64,
            _ => 0,
        };
        if worst.is_none_or(|(w, _)| used > w) {
            worst = Some((used, window_secs));
        }
    }
    let (used, secs) = worst?;
    let used = used.clamp(0.0, 1.0);
    Some(UsageSnapshot {
        fraction_used: Some(used),
        window: minimax_window_name(secs),
        label: Some(format!("{:.1}% of coding plan used", used * 100.0)),
    })
}

fn minimax_window_name(secs: u64) -> String {
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
        store
            .save("minimax", &json!({ "api_key": "sk-minimax-test" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        let stream = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("no stream errors");
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-minimax-test")
        );
        let body = mock.captured.body();
        assert!(
            body.contains("\"stream\":true"),
            "stream flag should be forced: {body}"
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("minimax", &json!({ "api_key": "wrong" }))
            .unwrap();

        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"invalid api key"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be an AuthenticationError");
        assert!(auth.message.contains("mixer auth login minimax"));
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        // Neither env nor file has a key.
        let settings = ProviderSettings::default_enabled();
        let result = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }

    #[tokio::test]
    async fn usage_returns_none_without_credentials() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let snap = MinimaxProvider.usage(&store, &settings).await.unwrap();
        assert!(snap.is_none(), "no api key → Ok(None)");
    }

    #[tokio::test]
    async fn chat_completion_prefers_env_var_over_stored_file() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("minimax", &json!({ "api_key": "sk-from-file" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"resp-1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let var = "MIXER_TEST_MINIMAX_KEY";
        // SAFETY: tests share no global state with this unique var name.
        unsafe { std::env::set_var(var, "sk-from-env") };
        let settings = ProviderSettings {
            base_url: Some(format!("http://{}", mock.addr)),
            api_key_env: Some(var.to_string()),
            ..ProviderSettings::default_enabled()
        };
        let stream = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let _: Vec<_> = stream.collect().await;
        unsafe { std::env::remove_var(var) };

        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-from-env"),
            "env var should shadow the stored file key",
        );
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
        fn extract_snapshot_picks_highest_used_across_models() {
            let body = json!({
                "modelRemains": [
                    {
                        "modelName": "MiniMax-M2",
                        "startTime": 1_000_000,
                        "endTime": 1_018_000, // 18_000s = 5h
                        "currentIntervalTotalCount": 1000,
                        "currentIntervalRemainingCount": 900 // 10% used
                    },
                    {
                        "modelName": "MiniMax-M2-vl",
                        "startTime": 1_000_000,
                        "endTime": 1_018_000,
                        "currentIntervalTotalCount": 200,
                        "currentIntervalRemainingCount": 50 // 75% used — worst
                    }
                ]
            });
            let snap = extract_minimax_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.75));
            assert_eq!(snap.window, "hourly");
            let label = snap.label.unwrap();
            assert!(label.contains("75.0"), "label mentions percent: {label}");
        }

        #[test]
        fn extract_snapshot_returns_none_without_model_remains() {
            assert!(extract_minimax_usage_snapshot(&json!({ "other": "field" })).is_none());
        }

        #[test]
        fn extract_snapshot_returns_none_when_all_entries_missing_counts() {
            let body = json!({
                "modelRemains": [
                    { "modelName": "m" }
                ]
            });
            assert!(extract_minimax_usage_snapshot(&body).is_none());
        }

        #[test]
        fn extract_snapshot_clamps_remaining_above_total() {
            // Defensive: a server glitch could emit remaining > total; clamp
            // rather than underflow the unsigned subtraction.
            let body = json!({
                "modelRemains": [
                    {
                        "currentIntervalTotalCount": 100,
                        "currentIntervalRemainingCount": 999
                    }
                ]
            });
            let snap = extract_minimax_usage_snapshot(&body).expect("snapshot");
            assert_eq!(snap.fraction_used, Some(0.0));
        }

        #[test]
        fn extract_snapshot_zero_total_is_skipped() {
            let body = json!({
                "modelRemains": [
                    { "currentIntervalTotalCount": 0, "currentIntervalRemainingCount": 0 }
                ]
            });
            assert!(extract_minimax_usage_snapshot(&body).is_none());
        }

        #[test]
        fn window_name_maps_seconds_to_label() {
            assert_eq!(minimax_window_name(0), "window");
            assert_eq!(minimax_window_name(60), "minutes");
            assert_eq!(minimax_window_name(3600), "hourly");
            assert_eq!(minimax_window_name(86_400), "daily");
            assert_eq!(minimax_window_name(7 * 86_400), "weekly");
            assert_eq!(minimax_window_name(60 * 86_400), "monthly");
        }

        // ── mocked /v1/api/openplatform/coding_plan/remains ────────────────

        struct UsageMock {
            calls: AtomicUsize,
            auth_tokens: StdMutex<Vec<String>>,
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
            state.auth_tokens.lock().unwrap().push(auth);
            Response::builder()
                .status(state.status)
                .header("Content-Type", "application/json")
                .body(Body::from(state.body.clone()))
                .unwrap()
        }

        async fn start_usage_mock(state: Arc<UsageMock>) -> SocketAddr {
            let app = Router::new()
                .route(USAGE_CODING_PLAN_PATH, get(usage_handler))
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
                .save("minimax", &json!({ "api_key": "sk-minimax-test" }))
                .unwrap();
        }

        #[tokio::test]
        async fn usage_returns_snapshot_with_bearer_auth() {
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_tokens: StdMutex::new(Vec::new()),
                status: 200,
                body: json!({
                    "modelRemains": [
                        {
                            "modelName": "MiniMax-M2",
                            "startTime": 1_000_000,
                            "endTime": 1_018_000,
                            "currentIntervalTotalCount": 1000,
                            "currentIntervalRemainingCount": 600
                        }
                    ]
                })
                .to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = MinimaxProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds")
                .expect("snapshot present");

            assert_eq!(snap.fraction_used, Some(0.4));
            assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                mock.auth_tokens.lock().unwrap()[0],
                "Bearer sk-minimax-test",
                "usage probe must forward bearer token"
            );
        }

        #[tokio::test]
        async fn usage_returns_none_on_upstream_error() {
            let tmp = TempDir::new().unwrap();
            let store = store_in(&tmp);
            write_key(&store);

            let mock = Arc::new(UsageMock {
                calls: AtomicUsize::new(0),
                auth_tokens: StdMutex::new(Vec::new()),
                status: 500,
                body: r#"{"error":"oops"}"#.to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = MinimaxProvider
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
                auth_tokens: StdMutex::new(Vec::new()),
                status: 200,
                body: "not-json".to_string(),
            });
            let addr = start_usage_mock(Arc::clone(&mock)).await;
            let settings = settings_with_base(&format!("http://{addr}"));

            let snap = MinimaxProvider
                .usage(&store, &settings)
                .await
                .expect("usage call succeeds");
            assert!(snap.is_none(), "malformed body should degrade to Ok(None)");
        }
    }
}
