//! Ollama (self-hosted) provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `http://localhost:11434/v1` (override via
//! `providers.ollama.base_url`).
//! Auth: none — intended for a local ollama server.
//!
//! This provider is disabled by default in `Config::default`. Users must opt
//! in by setting `providers.ollama.enabled = true`. `max_concurrent_requests`
//! is the main reason to run ollama through mixer: GPU-constrained hosts can
//! cap in-flight requests so the server is never asked to service more than
//! it can handle.
//!
//! **Model catalogue:** fetched on the first call to `models()` via
//! `GET <DEFAULT_BASE_URL>/models`. After that, results are cached for
//! [`CATALOG_TTL`] (60s); when the cache goes stale we serve the previous
//! snapshot immediately and kick off a background refresh so newly-installed
//! ollama models become visible without restarting mixer. The refresh requires
//! a tokio runtime to be active (CLI commands like `mixer providers list` run
//! inside one); if not, we skip the refresh and serve stale data — the
//! catalogue updates next time mixer is invoked.
//!
//! The very first fetch runs on a dedicated OS thread with its own
//! current-thread tokio runtime so it works regardless of the caller's async
//! context (nested-runtime panics are avoided). If the ollama server is
//! unreachable we cache an empty catalogue and the router filters this provider
//! out just like an unauthenticated one — but the TTL still applies, so the
//! next stale-driven background refresh will pick up the server when it comes
//! back.
//!
//! The catalogue fetch targets the default URL even if the user overrides
//! `base_url` in config, because `models()` is a synchronous trait method
//! that has no access to `ProviderSettings`. Chat dispatch respects the
//! configured `base_url` regardless.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::models_list::fetch_openai_models;
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider, RemoteModelEntry};
use crate::usage::UsageSnapshot;

const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";
const CHAT_PATH: &str = "/chat/completions";
const MODELS_PATH: &str = "/models";
/// Short timeout so a stopped ollama server doesn't stall every CLI command
/// that transitively calls `models()` (e.g. `mixer providers list`).
const CATALOG_FETCH_TIMEOUT: Duration = Duration::from_millis(500);
/// How long a fetched catalogue stays fresh before the next call schedules a
/// background refresh. Picked to be short enough that newly-installed ollama
/// models surface within a typical iteration cycle, long enough that a normal
/// request burst doesn't stampede the ollama `/models` endpoint.
const CATALOG_TTL: Duration = Duration::from_secs(60);

pub struct OllamaProvider;

#[derive(Default)]
struct CatalogCache {
    models: Vec<ModelInfo>,
    fetched_at: Option<Instant>,
}

static MODELS_CACHE: OnceLock<Arc<Mutex<CatalogCache>>> = OnceLock::new();
/// Coalesces concurrent stale-driven background refreshes — only the first
/// caller wins the CAS and actually fires the fetch.
static REFRESH_INFLIGHT: AtomicBool = AtomicBool::new(false);

fn catalog_cache() -> &'static Arc<Mutex<CatalogCache>> {
    MODELS_CACHE.get_or_init(|| Arc::new(Mutex::new(CatalogCache::default())))
}

#[derive(Deserialize)]
struct OpenAiModelsResponse {
    data: Vec<OpenAiModelEntry>,
}

#[derive(Deserialize)]
struct OpenAiModelEntry {
    id: String,
}

/// Fetch the catalogue over async reqwest. Separated so tests can exercise
/// the fetch directly against a mocked endpoint, bypassing the global cache.
async fn fetch_catalog_async(base_url: &str) -> Result<Vec<ModelInfo>> {
    let client = reqwest::Client::builder()
        .timeout(CATALOG_FETCH_TIMEOUT)
        .build()
        .context("building reqwest client for ollama catalog")?;
    let url = format!("{}{MODELS_PATH}", base_url.trim_end_matches('/'));
    let resp = client.get(&url).send().await?.error_for_status()?;
    let parsed: OpenAiModelsResponse = resp.json().await?;
    Ok(parsed.data.into_iter().map(entry_to_model_info).collect())
}

/// Convert an OpenAI-models entry into a [`ModelInfo`]. The catalogue can
/// change at runtime (users install new ollama models), so we hold owned
/// strings rather than leaking — the cache itself bounds memory.
fn entry_to_model_info(entry: OpenAiModelEntry) -> ModelInfo {
    ModelInfo::new(
        entry.id.clone(),
        entry.id,
        // Ollama's OpenAI-compat API doesn't report vision capability in the
        // catalogue; conservatively assume text-only. Users who need vision
        // routing against ollama can configure a vision-capable mixer model
        // that targets other providers.
        false,
        // Ollama's OpenAI-compat catalogue doesn't surface num_ctx either; a
        // conservative 8192 default excludes self-hosted models from oversize
        // prompts rather than routing them and timing out.
        8192,
    )
}

/// Blocking entry point used by the very first `models()` call. Runs the fetch
/// on a fresh OS thread with its own single-thread tokio runtime so we don't
/// panic when called from either inside a current-thread runtime (like
/// `#[tokio::test]`) or outside any runtime at all. Subsequent stale-driven
/// refreshes use the async path directly via the existing tokio runtime.
fn fetch_catalog_blocking(base_url: &str) -> Vec<ModelInfo> {
    let base = base_url.to_string();
    std::thread::spawn(move || -> Vec<ModelInfo> {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return Vec::new();
        };
        rt.block_on(async { fetch_catalog_async(&base).await.unwrap_or_default() })
    })
    .join()
    .unwrap_or_default()
}

/// Snapshot the cached catalogue, returning `(models, age)` where `age` is
/// `None` until the first fetch has populated the cache.
fn snapshot_catalog() -> (Vec<ModelInfo>, Option<Duration>) {
    let cache = catalog_cache().lock().expect("catalog cache poisoned");
    (cache.models.clone(), cache.fetched_at.map(|t| t.elapsed()))
}

fn store_catalog(fresh: Vec<ModelInfo>) {
    let mut cache = catalog_cache().lock().expect("catalog cache poisoned");
    cache.models = fresh;
    cache.fetched_at = Some(Instant::now());
}

/// If we're inside a tokio runtime and no other refresh is already in flight,
/// kick off a background fetch and update the cache when it returns. The CAS
/// ensures concurrent stale callers don't stampede ollama's `/models`.
fn try_spawn_background_refresh() {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    if REFRESH_INFLIGHT
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    handle.spawn(async {
        let fresh = fetch_catalog_async(DEFAULT_BASE_URL)
            .await
            .unwrap_or_default();
        store_catalog(fresh);
        REFRESH_INFLIGHT.store(false, Ordering::Release);
    });
}

#[async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn display_name(&self) -> &'static str {
        "Ollama (self-hosted)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        let (cached, age) = snapshot_catalog();
        match age {
            // First call ever: block on the fetch so the catalogue is populated
            // before we return — even an empty result counts (it primes the TTL
            // so subsequent calls stop blocking).
            None => {
                let fresh = fetch_catalog_blocking(DEFAULT_BASE_URL);
                store_catalog(fresh.clone());
                fresh
            }
            // Fresh enough — serve cached.
            Some(elapsed) if elapsed < CATALOG_TTL => cached,
            // Stale — return the cached snapshot now and refresh in the
            // background so the next call sees the update. If we're not in a
            // runtime (CLI path), the refresh is silently skipped and the next
            // invocation will retry.
            Some(_) => {
                try_spawn_background_refresh();
                cached
            }
        }
    }

    fn auth_kind(&self) -> AuthKind {
        AuthKind::None
    }

    fn is_authenticated(&self, _store: &CredentialStore, _settings: &ProviderSettings) -> bool {
        true
    }

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        eprintln!("ollama doesn't require authentication — just run a local ollama server.");
        Ok(())
    }

    async fn logout(&self, _store: &CredentialStore) -> Result<()> {
        Ok(())
    }

    /// Self-hosted has no plan quota to report.
    async fn usage(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(None)
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        openai_client::chat_completion(self.id(), &url, "", AuthScheme::None, timeout, None, req)
            .await
    }

    /// Direct, uncached read of `GET <base_url>/models`. Deliberately bypasses
    /// the TTL'd in-memory cache that `models()` uses: the CLI path wants a
    /// live snapshot, not whatever was last served to a routing decision.
    async fn list_remote_models(
        &self,
        _store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<Vec<RemoteModelEntry>>> {
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{MODELS_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        let entries =
            fetch_openai_models(self.id(), &url, "", AuthScheme::None, timeout, None).await?;
        Ok(Some(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatRequest;
    use axum::{
        Router,
        body::Body,
        extract::State,
        http::StatusCode,
        response::Response,
        routing::{get, post},
    };
    use futures::StreamExt;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration as StdDuration;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    fn sample_request() -> ChatRequest {
        serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap()
    }

    fn store_in(tmp: &TempDir) -> CredentialStore {
        CredentialStore::with_dir_for_tests(tmp.path().to_path_buf())
    }

    fn sse(payloads: &[&str]) -> String {
        payloads.iter().map(|p| format!("data: {p}\n\n")).collect()
    }

    // ── /v1/models catalogue fetch ──────────────────────────────────────────

    /// Serve a canned JSON body at GET /models. Used by the catalogue tests.
    async fn start_models_mock(body: String) -> SocketAddr {
        async fn handler(State(body): State<Arc<String>>) -> Response {
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Body::from((*body).clone()))
                .unwrap()
        }
        let app = Router::new()
            .route("/models", get(handler))
            .with_state(Arc::new(body));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        addr
    }

    #[tokio::test]
    async fn fetch_catalog_parses_openai_models_response() {
        let body = r#"{
            "object": "list",
            "data": [
                {"id": "llama3.1:8b", "object": "model"},
                {"id": "qwen2.5-coder:7b", "object": "model"}
            ]
        }"#;
        let addr = start_models_mock(body.to_string()).await;
        let models = fetch_catalog_async(&format!("http://{addr}"))
            .await
            .unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama3.1:8b");
        assert_eq!(models[1].id, "qwen2.5-coder:7b");
        // All ollama entries report non-vision — we don't trust the catalogue
        // for image capability.
        assert!(models.iter().all(|m| !m.supports_images));
    }

    #[tokio::test]
    async fn fetch_catalog_returns_error_when_unreachable() {
        // 127.0.0.1:1 is reserved; nothing will answer.
        let result = fetch_catalog_async("http://127.0.0.1:1").await;
        assert!(
            result.is_err(),
            "unreachable server should surface an error"
        );
    }

    // ── chat_completion streaming ───────────────────────────────────────────

    /// Mock that serves POST /chat/completions with the configured SSE body
    /// and records the peak number of concurrent in-flight requests (for the
    /// cap test).
    struct ChatMock {
        max_concurrent: Arc<AtomicUsize>,
    }

    async fn start_chat_mock(sse_body: String, hold: StdDuration) -> (SocketAddr, ChatMock) {
        let concurrent = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        #[derive(Clone)]
        struct HState {
            sse: Arc<String>,
            hold: StdDuration,
            concurrent: Arc<AtomicUsize>,
            max: Arc<AtomicUsize>,
        }

        async fn handler(State(state): State<HState>, _body: String) -> Response {
            let n = state.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
            state.max.fetch_max(n, Ordering::SeqCst);
            tokio::time::sleep(state.hold).await;
            state.concurrent.fetch_sub(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .body(Body::from((*state.sse).clone()))
                .unwrap()
        }

        let state = HState {
            sse: Arc::new(sse_body),
            hold,
            concurrent,
            max: max_concurrent.clone(),
        };
        let app = Router::new()
            .route("/chat/completions", post(handler))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (addr, ChatMock { max_concurrent })
    }

    #[tokio::test]
    async fn chat_completion_proxies_streaming_response() {
        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"llama3.1:8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"llama3.1:8b","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"llama3.1:8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let (addr, _mock) = start_chat_mock(body, StdDuration::from_millis(0)).await;
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings {
            base_url: Some(format!("http://{addr}")),
            ..ProviderSettings::default_enabled()
        };

        let stream = OllamaProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(chunks.len(), 3, "DONE sentinel should be filtered out");
        let text: String = chunks
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        assert_eq!(text, "hi");
    }

    #[tokio::test]
    async fn chat_completion_surfaces_connection_error_when_server_unreachable() {
        // Nothing listening on 127.0.0.1:1 → reqwest fails to connect.
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings {
            base_url: Some("http://127.0.0.1:1".to_string()),
            // Keep the test fast even if something does answer.
            request_timeout_secs: Some(1),
            ..ProviderSettings::default_enabled()
        };

        let result = OllamaProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        assert!(
            result.is_err(),
            "unreachable ollama server must error, not return empty stream"
        );
    }

    // ── Concurrency cap ─────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrency_cap_limits_in_flight_requests() {
        use crate::concurrency::ConcurrencyLimits;
        use crate::config::Config;

        // Hold each request for 100ms so concurrency is observable.
        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let (addr, mock) = start_chat_mock(body, StdDuration::from_millis(100)).await;

        let mut config = Config::default();
        // Override the default-disabled ollama entry with a cap of 2.
        config.providers.insert(
            "ollama".to_string(),
            ProviderSettings {
                enabled: true,
                max_concurrent_requests: Some(2),
                base_url: Some(format!("http://{addr}")),
                ..ProviderSettings::default_enabled()
            },
        );
        let limits = ConcurrencyLimits::from_config(&config);
        let settings = config.providers["ollama"].clone();

        let tmp = TempDir::new().unwrap();
        let store = Arc::new(store_in(&tmp));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let limits = limits.clone();
            let settings = settings.clone();
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                let _permit = limits.acquire("ollama").await;
                let stream = OllamaProvider
                    .chat_completion(&store, &settings, sample_request())
                    .await?;
                let chunks: Vec<_> = stream.collect().await;
                let _ = chunks.into_iter().collect::<Result<Vec<_>>>()?;
                Ok::<(), anyhow::Error>(())
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        let peak = mock.max_concurrent.load(Ordering::SeqCst);
        assert!(
            peak <= 2,
            "cap=2 should gate at most 2 concurrent upstream requests, saw {peak}",
        );
        assert!(
            peak >= 2,
            "with 5 tasks and hold=100ms we should hit the cap"
        );
    }
}
