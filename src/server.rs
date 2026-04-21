//! Axum-based OpenAI-compatible HTTP server.
//!
//! Endpoints:
//!   - `GET  /v1/models`             → lists configured mixer models
//!   - `POST /v1/chat/completions`   → routes + dispatches to a provider
//!   - `GET  /healthz`               → always 200 while the server is up
//!
//! Request flow:
//!   1. Parse the OpenAI-compatible body.
//!   2. Resolve `model` to a [`MixerModel`] (falling back to the default).
//!   3. Inspect the request for image parts to decide capability filter.
//!   4. `router::pick` chooses a `(provider, provider_model)`.
//!   5. Acquire the provider's concurrency permit.
//!   6. Rewrite `req.model` and dispatch via `Provider::chat_completion`.
//!   7. Providers always return a stream of `ChatCompletionChunk`s. If the
//!      client asked for `stream: true` we forward as SSE with a trailing
//!      `data: [DONE]` frame; otherwise we accumulate into a JSON response.
//!
//! **Per-request failover** (plan.md §5.2.1): a single retryable failure on
//! the first chosen backend causes the server to re-pick from the same mixer
//! model's pool *excluding* the failed backend. Retry budget is hardcoded to
//! one. We can only retry while no chunks have been emitted to the client —
//! once the SSE stream has flushed a chunk a mid-stream failure is terminal,
//! and we emit a final `data: {"error":...}` frame and close (no `[DONE]`).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{StreamExt, stream};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use crate::concurrency::ConcurrencyLimits;
use crate::config::{Backend, Config, ProviderSettings};
use crate::credentials::CredentialStore;
use crate::openai::{
    self, ChatRequest, ChatResponse, ModelListEntry, ModelListResponse, request_has_images,
};
use crate::providers::ChatStream;
use crate::providers::ProviderRegistry;
use crate::providers::common::oauth_refresh::{AuthenticationError, UpstreamHttpError};
use crate::router;
use crate::usage::UsageCache;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub registry: Arc<ProviderRegistry>,
    pub credentials: Arc<CredentialStore>,
    pub concurrency: ConcurrencyLimits,
    /// Short-TTL cache for `Provider::usage` lookups, consulted by the
    /// usage-aware router on each pick. Lives on `AppState` so the cache is
    /// shared across all in-flight requests for a given `mixer serve` process.
    pub usage_cache: UsageCache,
    /// When `Some`, only this mixer model name is served; all others return
    /// 404. Used by `mixer serve --model <name>`.
    pub pinned_model: Option<String>,
}

pub async fn serve(state: AppState, listen_addr: &str) -> Result<()> {
    let app = build_router(state.clone());
    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("binding to {listen_addr}"))?;
    let actual = listener.local_addr()?;
    tracing::info!(addr = %actual, "mixer listening");
    maybe_warn_unprotected_bind(&state.config, &actual);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")?;
    Ok(())
}

fn build_router(state: AppState) -> Router {
    // The bearer-token gate only guards `/v1/*`; `/healthz` always answers so
    // liveness probes keep working regardless of the auth posture.
    let v1 = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route_layer(middleware::from_fn_with_state(state.clone(), bearer_auth));

    Router::new()
        .route("/healthz", get(healthz))
        .merge(v1)
        .with_state(state)
}

/// Log a warning at startup when the server is about to accept traffic on a
/// non-loopback interface without a bearer-token gate configured. This is the
/// one place mixer has "first contact" with the network posture the operator
/// picked.
fn maybe_warn_unprotected_bind(config: &Config, bound: &std::net::SocketAddr) {
    if bound.ip().is_loopback() {
        return;
    }
    if resolve_bearer_token(config).is_some() {
        return;
    }
    tracing::warn!(
        addr = %bound,
        "mixer is bound to a non-loopback address without a bearer-token gate; \
         set `listen_bearer_token_env` in config and export the named env var to require \
         `Authorization: Bearer <token>` on every /v1/* request",
    );
}

/// Resolve the configured bearer token from the environment, treating an unset
/// or empty value as "gate disabled". Returning `Option<String>` (rather than
/// referencing the env value) sidesteps borrowing the process env across await
/// points.
fn resolve_bearer_token(config: &Config) -> Option<String> {
    let env_name = config.listen_bearer_token_env.as_deref()?;
    let value = std::env::var(env_name).ok()?;
    if value.is_empty() { None } else { Some(value) }
}

/// Outcome of checking a request's Authorization header against the configured
/// expected bearer. Kept separate from the middleware wrapper so the logic can
/// be unit-tested without touching the process environment or axum internals.
#[derive(Debug, PartialEq, Eq)]
enum BearerOutcome {
    /// Auth is not configured — pass the request through.
    Disabled,
    /// Auth matches — pass the request through.
    Allowed,
    /// Auth missing or wrong — respond 401.
    Denied,
}

fn verify_bearer(expected: Option<&str>, auth_header: Option<&str>) -> BearerOutcome {
    let Some(expected) = expected else {
        return BearerOutcome::Disabled;
    };
    let provided = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .unwrap_or("");
    if constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
        BearerOutcome::Allowed
    } else {
        BearerOutcome::Denied
    }
}

/// Constant-time byte comparison that does not short-circuit on length
/// mismatch: both length and byte differences fold into the same accumulator,
/// so run time depends only on `max(a.len(), b.len())`. Hand-rolled to avoid a
/// `subtle` dep for a single comparison site.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let len = a.len().max(b.len());
    // Fold the length difference into the accumulator as a single non-zero
    // bit when lengths disagree. Using u32 avoids wrap on `usize -> u8` casts.
    let mut diff: u32 = if a.len() == b.len() { 0 } else { 1 };
    for i in 0..len {
        let av = *a.get(i).unwrap_or(&0);
        let bv = *b.get(i).unwrap_or(&0);
        diff |= (av ^ bv) as u32;
    }
    diff == 0
}

async fn bearer_auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let expected = resolve_bearer_token(&state.config);
    let header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    match verify_bearer(expected.as_deref(), header.as_deref()) {
        BearerOutcome::Disabled | BearerOutcome::Allowed => next.run(req).await,
        BearerOutcome::Denied => {
            AppError::unauthorized("missing or invalid bearer token").into_response()
        }
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received, draining");
}

async fn healthz() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let data = state
        .config
        .models
        .keys()
        .filter(|k| state.pinned_model.as_ref().map(|p| p == *k).unwrap_or(true))
        .map(|id| ModelListEntry {
            id: id.clone(),
            object: "model",
            owned_by: "mixer",
        })
        .collect();
    Json(ModelListResponse {
        object: "list",
        data,
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    let start = Instant::now();
    let wants_stream = req.stream.unwrap_or(false);
    let requires_images = request_has_images(&req);

    // Per-request span (plan.md §10): fields are declared up-front and filled
    // in as routing decisions are made. The subscriber is configured with
    // FmtSpan::CLOSE so one final log line fires when the span exits with
    // every field populated.
    let span = tracing::info_span!(
        "chat_completion",
        mixer_model = %req.model,
        provider = tracing::field::Empty,
        provider_model = tracing::field::Empty,
        stream = wants_stream,
        has_images = requires_images,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
        status_code = tracing::field::Empty,
    );

    async move {
        let result = dispatch_chat(state, req, headers, wants_stream, requires_images).await;
        let current = tracing::Span::current();
        current.record("duration_ms", start.elapsed().as_millis() as u64);
        let status = match &result {
            Ok(resp) => resp.status().as_u16(),
            Err(e) => e.status.as_u16(),
        };
        current.record("status_code", status);
        result
    }
    .instrument(span)
    .await
}

async fn dispatch_chat(
    state: AppState,
    req: ChatRequest,
    headers: HeaderMap,
    wants_stream: bool,
    requires_images: bool,
) -> Result<Response, AppError> {
    if let Some(pinned) = &state.pinned_model
        && &req.model != pinned
    {
        return Err(AppError::not_found(format!(
            "mixer was started with --model {pinned}; requested `{}` is not served",
            req.model
        )));
    }

    let (mixer_model_name, mixer_model) =
        state.config.resolve_model(&req.model).ok_or_else(|| {
            AppError::not_found(format!(
                "no mixer model named `{}` and no default configured",
                req.model
            ))
        })?;

    // Sticky key is computed once per request — evaluating it after a retry
    // would reuse the just-failed backend, which is exactly what
    // `pick_excluding` is there to prevent.
    let sticky_hash = router::compute_sticky_hash(mixer_model, &req, &headers);

    let est_in = openai::estimate_input_tokens(&req);
    let max_out = req.resolved_max_tokens().unwrap_or(0);

    // Retry budget = 1 (plan.md §5.2.1). On a retryable failure before any
    // chunk reaches the client, we re-pick from the pool excluding the failed
    // backend. A second failure surfaces. A failure after a chunk has been
    // emitted (mid-stream) is terminal and is signalled in-band on the SSE
    // stream (see `sse_response`).
    let mut excluded: Vec<Backend> = Vec::new();
    let mut last_err: Option<AppError> = None;

    let ctx = router::RoutingContext {
        config: &state.config,
        registry: &state.registry,
        credentials: &state.credentials,
        usage_cache: &state.usage_cache,
    };

    for attempt in 1..=2u8 {
        let pick_result = if excluded.is_empty() {
            router::pick(
                &ctx,
                mixer_model,
                requires_images,
                est_in,
                max_out,
                sticky_hash,
            )
            .await
        } else {
            router::pick_excluding(
                &ctx,
                mixer_model,
                requires_images,
                est_in,
                max_out,
                &excluded,
                sticky_hash,
            )
            .await
        };

        let decision = match pick_result {
            Ok(d) => d,
            Err(e) => {
                // No alternate backend available. Prefer the first attempt's
                // error if we already have one, since it's a more useful
                // diagnostic than "no eligible backends remain".
                return Err(last_err.unwrap_or_else(|| AppError::from_provider(e)));
            }
        };

        let span = tracing::Span::current();
        span.record("provider", decision.provider_id.as_str());
        span.record("provider_model", decision.provider_model.as_str());
        tracing::info!(
            mixer_model = %mixer_model_name,
            provider = %decision.provider_id,
            provider_model = %decision.provider_model,
            attempt,
            has_images = requires_images,
            "route decision",
        );

        let provider = state
            .registry
            .get(&decision.provider_id)
            .map_err(AppError::internal)?;
        let settings = state
            .config
            .providers
            .get(&decision.provider_id)
            .cloned()
            .unwrap_or_else(ProviderSettings::default_enabled);

        let permit = state.concurrency.acquire(&decision.provider_id).await;

        let mut req_attempt = req.clone();
        req_attempt.model = decision.provider_model.clone();

        let dispatch_result = provider
            .chat_completion(&state.credentials, &settings, req_attempt)
            .await;

        let stream = match dispatch_result {
            Ok(s) => s,
            Err(e) => {
                drop(permit);
                if attempt == 1 && is_retryable(&e) {
                    log_retry(&decision.provider_id, &e, "dispatch");
                    excluded.push(Backend {
                        provider: decision.provider_id.clone(),
                        model: decision.provider_model.clone(),
                    });
                    last_err = Some(AppError::from_provider(e));
                    continue;
                }
                return Err(AppError::from_provider(e));
            }
        };

        // Peek at the first item on the stream so a retryable failure that
        // happens before any data reaches the client can still be retried.
        let mut stream = stream;
        let first = stream.next().await;
        match first {
            Some(Err(e)) => {
                drop(permit);
                if attempt == 1 && is_retryable(&e) {
                    log_retry(&decision.provider_id, &e, "first-chunk");
                    excluded.push(Backend {
                        provider: decision.provider_id.clone(),
                        model: decision.provider_model.clone(),
                    });
                    last_err = Some(AppError::from_provider(e));
                    continue;
                }
                return Err(AppError::from_provider(e));
            }
            None => {
                // Empty stream from upstream — surface an empty success.
                let combined: ChatStream = Box::pin(stream::empty());
                return finalize_response(combined, permit, wants_stream).await;
            }
            Some(Ok(first_chunk)) => {
                let head = stream::once(async move { Ok(first_chunk) });
                let combined: ChatStream = Box::pin(head.chain(stream));
                return finalize_response(combined, permit, wants_stream).await;
            }
        }
    }

    // The loop returns or `continue`s on every iteration; falling through
    // would mean we exhausted attempts without setting `last_err`, which is
    // unreachable in practice.
    Err(last_err.unwrap_or_else(|| {
        AppError::internal(anyhow::anyhow!("retry loop exited without a result"))
    }))
}

/// Either accumulate the (already first-chunk-resolved) stream into a single
/// JSON body or pipe it as SSE, based on `wants_stream`.
async fn finalize_response(
    chunks: ChatStream,
    permit: Option<OwnedSemaphorePermit>,
    wants_stream: bool,
) -> Result<Response, AppError> {
    if wants_stream {
        Ok(sse_response(chunks, permit).into_response())
    } else {
        let mut collected = Vec::new();
        let mut s = chunks;
        while let Some(next) = s.next().await {
            // A mid-buffer error in non-streaming mode surfaces as a single
            // 502 — by the time we have at least one accumulated chunk we
            // can no longer redo work cheaply, and the client never saw
            // partial output, so a single error response is the right shape.
            collected.push(next.map_err(AppError::from_provider)?);
        }
        drop(permit);
        let response = ChatResponse::from_chunks(collected);
        Ok(Json(response).into_response())
    }
}

/// Classify a provider error for the per-request failover loop (plan.md
/// §5.2.1). Retry on transient upstream conditions (429 / 5xx / connect
/// errors / read-timeouts); never retry on caller-broken requests (4xx other
/// than 429) or on a post-refresh `AuthenticationError` (which already means
/// "user must re-login").
fn is_retryable(err: &anyhow::Error) -> bool {
    if err.downcast_ref::<AuthenticationError>().is_some() {
        return false;
    }
    if let Some(http) = err.downcast_ref::<UpstreamHttpError>() {
        return matches!(http.status, 429 | 500 | 502 | 503 | 504);
    }
    err.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|re| re.is_connect() || re.is_timeout() || re.is_request())
    })
}

fn log_retry(provider_id: &str, err: &anyhow::Error, where_: &str) {
    let reason = if let Some(http) = err.downcast_ref::<UpstreamHttpError>() {
        format!("HTTP {}", http.status)
    } else if err.chain().any(|c| {
        c.downcast_ref::<reqwest::Error>()
            .is_some_and(|re| re.is_timeout())
    }) {
        "timeout".to_string()
    } else if err.chain().any(|c| {
        c.downcast_ref::<reqwest::Error>()
            .is_some_and(|re| re.is_connect())
    }) {
        "connect".to_string()
    } else {
        format!("{err:#}")
    };
    tracing::info!(
        retry = true,
        from_provider = provider_id,
        stage = where_,
        reason = %reason,
        "retrying on a different backend",
    );
}

/// Convert a provider chunk stream into an SSE response. Each chunk becomes a
/// `data: {json}` frame.
///
/// On a successful stream a final `data: [DONE]` frame closes the response.
/// Per plan.md §5.2.1, a mid-stream error emits a single
/// `data: {"error":{...}}` frame and then closes the connection *without* a
/// trailing `[DONE]` — most OpenAI SDKs treat the absence of `[DONE]` as the
/// signal that the stream did not complete cleanly.
fn sse_response(
    chunks: ChatStream,
    permit: Option<OwnedSemaphorePermit>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let body = async_stream::stream! {
        let mut chunks = chunks;
        let mut errored = false;
        while let Some(item) = chunks.next().await {
            match item {
                Ok(chunk) => {
                    let payload = serde_json::to_string(&chunk).unwrap_or_else(|e| {
                        json!({
                            "error": {
                                "message": format!("chunk serialize failed: {e}"),
                                "type": "mixer_internal_error",
                            }
                        })
                        .to_string()
                    });
                    yield Ok::<_, Infallible>(Event::default().data(payload));
                }
                Err(e) => {
                    let payload = json!({
                        "error": {
                            "message": format!("{e:#}"),
                            "type": "upstream_error",
                        }
                    })
                    .to_string();
                    yield Ok(Event::default().data(payload));
                    errored = true;
                    break;
                }
            }
        }
        // Hold the concurrency permit until the very end of the stream so a
        // long-running response keeps a slot in the per-provider semaphore.
        drop(permit);
        if !errored {
            yield Ok(Event::default().data("[DONE]"));
        }
    };
    Sse::new(body).keep_alive(KeepAlive::default())
}

/// Converts anyhow errors into OpenAI-style JSON error bodies so downstream
/// OpenAI SDKs surface them naturally.
#[derive(Debug)]
pub struct AppError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl AppError {
    fn internal(e: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "mixer_internal_error",
            message: format!("{e:#}"),
        }
    }

    fn bad_gateway(e: anyhow::Error) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            kind: "mixer_upstream_error",
            message: format!("{e:#}"),
        }
    }

    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "mixer_not_found",
            message: msg.into(),
        }
    }

    fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            kind: "authentication_error",
            message: msg.into(),
        }
    }

    /// Classify a provider error: downcast to [`AuthenticationError`] for a
    /// 401 with `type: "authentication_error"` (so OpenAI SDKs surface the
    /// actionable "run `mixer auth login`" message as a real auth error),
    /// otherwise fall through to `bad_gateway`.
    fn from_provider(e: anyhow::Error) -> Self {
        if let Some(auth) =
            e.downcast_ref::<crate::providers::common::oauth_refresh::AuthenticationError>()
        {
            return Self {
                status: StatusCode::UNAUTHORIZED,
                kind: "authentication_error",
                message: auth.message.clone(),
            };
        }
        Self::bad_gateway(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.kind,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, MixerModel, RoutingStrategy};
    use crate::openai::{ChatCompletionChunk, ChatDelta, ChatMessage, ChunkChoice, MessageContent};
    use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider, ProviderRegistry};
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use axum::extract::State;
    use futures::stream;
    use std::sync::Mutex;
    use tempfile::TempDir;

    struct ChunkedStubProvider {
        chunks: Mutex<Option<Vec<ChatCompletionChunk>>>,
    }

    impl ChunkedStubProvider {
        fn new(chunks: Vec<ChatCompletionChunk>) -> Self {
            Self {
                chunks: Mutex::new(Some(chunks)),
            }
        }
    }

    #[async_trait]
    impl Provider for ChunkedStubProvider {
        fn id(&self) -> &'static str {
            "stub"
        }
        fn display_name(&self) -> &'static str {
            "Stub"
        }
        fn models(&self) -> Vec<ModelInfo> {
            vec![ModelInfo::new("m", "M", false, 100_000)]
        }
        fn auth_kind(&self) -> AuthKind {
            AuthKind::ApiKey
        }
        fn is_authenticated(&self, _store: &CredentialStore, _settings: &ProviderSettings) -> bool {
            true
        }
        async fn login(&self, _store: &CredentialStore) -> anyhow::Result<()> {
            Ok(())
        }
        async fn chat_completion(
            &self,
            _store: &CredentialStore,
            _settings: &ProviderSettings,
            _req: ChatRequest,
        ) -> anyhow::Result<ChatStream> {
            let chunks = self.chunks.lock().unwrap().take().unwrap_or_default();
            Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
        }
    }

    fn test_state(chunks: Vec<ChatCompletionChunk>) -> AppState {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(ChunkedStubProvider::new(chunks)));

        let mut config = Config::default();
        config.providers.clear();
        config
            .providers
            .insert("stub".to_string(), ProviderSettings::default_enabled());
        config.models.clear();
        config.models.insert(
            "mixer".to_string(),
            MixerModel {
                description: String::new(),
                backends: vec![Backend {
                    provider: "stub".to_string(),
                    model: "m".to_string(),
                }],
                strategy: RoutingStrategy::Random,
                weights: Default::default(),
                sticky: None,
            },
        );

        let tmp = TempDir::new().unwrap();
        // Leak the tempdir so the credential store path stays valid for the
        // test's lifetime; the OS will clean up when the process exits.
        let path = tmp.path().to_path_buf();
        std::mem::forget(tmp);
        let credentials = CredentialStore::with_dir_for_tests(path);

        let concurrency = ConcurrencyLimits::from_config(&config);
        AppState {
            config: Arc::new(config),
            registry: Arc::new(registry),
            credentials: Arc::new(credentials),
            concurrency,
            usage_cache: UsageCache::default(),
            pinned_model: None,
        }
    }

    fn sample_chunks() -> Vec<ChatCompletionChunk> {
        vec![
            ChatCompletionChunk {
                id: "id-xyz".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1,
                model: "m".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: Some("hello ".to_string()),
                        tool_calls: None,
                        extra: Default::default(),
                    },
                    finish_reason: None,
                }],
                usage: None,
                extra: Default::default(),
            },
            ChatCompletionChunk {
                id: "id-xyz".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1,
                model: "m".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some("world".to_string()),
                        tool_calls: None,
                        extra: Default::default(),
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
                extra: Default::default(),
            },
        ]
    }

    fn chat_request(stream: bool) -> ChatRequest {
        ChatRequest {
            model: "mixer".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: Some(stream),
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            tools: None,
            tool_choice: None,
            extra: Default::default(),
        }
    }

    #[tokio::test]
    async fn non_streaming_returns_accumulated_json() {
        let state = test_state(sample_chunks());
        let resp = chat_completions(State(state), HeaderMap::new(), Json(chat_request(false)))
            .await
            .unwrap();
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::OK);
        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        let parsed: ChatResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.id, "id-xyz");
        assert_eq!(parsed.model, "m");
        assert_eq!(parsed.choices.len(), 1);
        match &parsed.choices[0].message.content {
            Some(MessageContent::Text(s)) => assert_eq!(s, "hello world"),
            _ => panic!("expected text content"),
        }
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn streaming_returns_sse_with_done_marker() {
        let state = test_state(sample_chunks());
        let resp = chat_completions(State(state), HeaderMap::new(), Json(chat_request(true)))
            .await
            .unwrap();
        let (parts, body) = resp.into_parts();
        assert_eq!(parts.status, StatusCode::OK);
        let ct = parts
            .headers
            .get(axum::http::header::CONTENT_TYPE)
            .expect("content-type")
            .to_str()
            .unwrap();
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got {ct}"
        );
        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("\"content\":\"hello \""));
        assert!(text.contains("\"content\":\"world\""));
        assert!(text.contains("\"finish_reason\":\"stop\""));
        assert!(text.contains("data: [DONE]"));
    }

    // ── Failover (per-request retry on a different backend) ─────────────────

    mod failover {
        use super::*;
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Programmable per-call behavior the scripted provider walks through.
        enum Behavior {
            /// Succeed: return a stream that yields these chunks.
            OkChunks(Vec<ChatCompletionChunk>),
            /// Fail before any stream is produced (e.g. upstream returned 5xx).
            DispatchErr(anyhow::Error),
            /// Hand back a stream whose first item is an error.
            FirstChunkErr(anyhow::Error),
            /// Yield the chunks, then surface an error on the next poll. Used
            /// to exercise the mid-stream failure path.
            ChunksThenErr(Vec<ChatCompletionChunk>, anyhow::Error),
        }

        struct ScriptedProvider {
            id: &'static str,
            calls: AtomicUsize,
            behaviors: Mutex<VecDeque<Behavior>>,
        }

        impl ScriptedProvider {
            fn new(id: &'static str, behaviors: Vec<Behavior>) -> Arc<Self> {
                Arc::new(Self {
                    id,
                    calls: AtomicUsize::new(0),
                    behaviors: Mutex::new(behaviors.into()),
                })
            }
            fn calls(&self) -> usize {
                self.calls.load(Ordering::SeqCst)
            }
        }

        #[async_trait]
        impl Provider for ScriptedProvider {
            fn id(&self) -> &'static str {
                self.id
            }
            fn display_name(&self) -> &'static str {
                self.id
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![ModelInfo::new("m", "M", false, 100_000)]
            }
            fn auth_kind(&self) -> AuthKind {
                AuthKind::ApiKey
            }
            fn is_authenticated(
                &self,
                _store: &CredentialStore,
                _settings: &ProviderSettings,
            ) -> bool {
                true
            }
            async fn login(&self, _store: &CredentialStore) -> anyhow::Result<()> {
                Ok(())
            }
            async fn chat_completion(
                &self,
                _store: &CredentialStore,
                _settings: &ProviderSettings,
                _req: ChatRequest,
            ) -> anyhow::Result<ChatStream> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let next = self
                    .behaviors
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("scripted provider script exhausted");
                match next {
                    Behavior::OkChunks(cs) => Ok(Box::pin(stream::iter(cs.into_iter().map(Ok)))),
                    Behavior::DispatchErr(e) => Err(e),
                    Behavior::FirstChunkErr(e) => Ok(Box::pin(stream::once(async move { Err(e) }))),
                    Behavior::ChunksThenErr(cs, e) => {
                        let head = stream::iter(cs.into_iter().map(Ok));
                        let tail = stream::once(async move { Err(e) });
                        Ok(Box::pin(head.chain(tail)))
                    }
                }
            }
        }

        /// Build an `AppState` whose `mixer` model lists exactly the supplied
        /// providers as backends (in order). All providers are auto-registered.
        fn pool_state(providers: &[Arc<ScriptedProvider>]) -> AppState {
            let mut registry = ProviderRegistry::new();
            for p in providers {
                let dyn_p: Arc<dyn Provider> = p.clone();
                registry.register(dyn_p);
            }
            let mut config = Config::default();
            config.providers.clear();
            for p in providers {
                config
                    .providers
                    .insert(p.id().to_string(), ProviderSettings::default_enabled());
            }
            config.models.clear();
            config.models.insert(
                "mixer".to_string(),
                MixerModel {
                    description: String::new(),
                    backends: providers
                        .iter()
                        .map(|p| Backend {
                            provider: p.id().to_string(),
                            model: "m".to_string(),
                        })
                        .collect(),
                    strategy: RoutingStrategy::Random,
                    weights: Default::default(),
                    sticky: None,
                },
            );

            let tmp = TempDir::new().unwrap();
            let path = tmp.path().to_path_buf();
            std::mem::forget(tmp);
            let credentials = CredentialStore::with_dir_for_tests(path);
            let concurrency = ConcurrencyLimits::from_config(&config);
            AppState {
                config: Arc::new(config),
                registry: Arc::new(registry),
                credentials: Arc::new(credentials),
                concurrency,
                usage_cache: UsageCache::default(),
                pinned_model: None,
            }
        }

        fn http_err(status: u16) -> anyhow::Error {
            anyhow::Error::new(UpstreamHttpError {
                status,
                body_snippet: format!("upstream {status}"),
            })
        }

        // ── is_retryable classification ─────────────────────────────────────

        #[test]
        fn classification_retries_on_429_and_5xx() {
            for status in [429u16, 500, 502, 503, 504] {
                assert!(
                    is_retryable(&http_err(status)),
                    "status {status} should be retryable"
                );
            }
        }

        #[test]
        fn classification_does_not_retry_on_request_error_4xx() {
            for status in [400u16, 401, 403, 404, 422] {
                assert!(
                    !is_retryable(&http_err(status)),
                    "status {status} should NOT be retryable"
                );
            }
        }

        #[test]
        fn classification_does_not_retry_on_authentication_error() {
            let err = anyhow::Error::new(AuthenticationError {
                message: "session dead".to_string(),
            });
            assert!(!is_retryable(&err));
        }

        #[tokio::test]
        async fn classification_retries_on_reqwest_connect_error() {
            // Force a real reqwest connection error by pointing at an
            // unroutable port. The error will downcast to reqwest::Error
            // somewhere in the source chain.
            let client = reqwest::Client::new();
            let err = client
                .get("http://127.0.0.1:1") // port 1 is reserved → connection refused
                .timeout(std::time::Duration::from_millis(500))
                .send()
                .await
                .expect_err("connect should fail");
            let wrapped = anyhow::Error::new(err).context("posting to upstream");
            assert!(
                is_retryable(&wrapped),
                "reqwest connect/timeout/request errors should be retryable: {wrapped:#}"
            );
        }

        // ── End-to-end retry behavior ───────────────────────────────────────

        fn ok_chunk(content: &str, finish: Option<&str>) -> ChatCompletionChunk {
            ChatCompletionChunk {
                id: "id-1".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1,
                model: "m".to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: Some("assistant".to_string()),
                        content: Some(content.to_string()),
                        tool_calls: None,
                        extra: Default::default(),
                    },
                    finish_reason: finish.map(|s| s.to_string()),
                }],
                usage: None,
                extra: Default::default(),
            }
        }

        #[tokio::test]
        async fn dispatch_503_retries_on_other_backend_and_succeeds() {
            let bad = ScriptedProvider::new("bad", vec![Behavior::DispatchErr(http_err(503))]);
            let good = ScriptedProvider::new(
                "good",
                vec![Behavior::OkChunks(vec![ok_chunk("hi", Some("stop"))])],
            );
            let state = pool_state(&[bad.clone(), good.clone()]);
            // Run multiple iterations so we exercise both initial-pick
            // orderings under the random strategy.
            for _ in 0..16 {
                let resp = chat_completions(
                    State(state.clone()),
                    HeaderMap::new(),
                    Json(chat_request(false)),
                )
                .await
                .expect("retry should produce success");
                let (parts, body) = resp.into_parts();
                assert_eq!(parts.status, StatusCode::OK);
                let bytes = to_bytes(body, usize::MAX).await.unwrap();
                let parsed: ChatResponse = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(parsed.choices.len(), 1);
                match &parsed.choices[0].message.content {
                    Some(MessageContent::Text(s)) => assert_eq!(s, "hi"),
                    _ => panic!("expected text content"),
                }
                // Reset stubs for the next iteration.
                bad.behaviors
                    .lock()
                    .unwrap()
                    .push_back(Behavior::DispatchErr(http_err(503)));
                good.behaviors
                    .lock()
                    .unwrap()
                    .push_back(Behavior::OkChunks(vec![ok_chunk("hi", Some("stop"))]));
            }
            assert!(
                bad.calls() >= 1,
                "the failing backend should have been tried at least once"
            );
            assert!(
                good.calls() >= 1,
                "the good backend should have served the eventual success"
            );
        }

        #[tokio::test]
        async fn first_chunk_503_retries_and_succeeds() {
            // The dispatch returns a stream that immediately errors. Because
            // no chunk has reached the client yet, retry should still work.
            let bad = ScriptedProvider::new("bad", vec![Behavior::FirstChunkErr(http_err(503))]);
            let good = ScriptedProvider::new(
                "good",
                vec![Behavior::OkChunks(vec![ok_chunk("ok", Some("stop"))])],
            );
            let state = pool_state(&[bad, good]);
            let resp = chat_completions(State(state), HeaderMap::new(), Json(chat_request(false)))
                .await
                .expect("retry should succeed");
            let (parts, _) = resp.into_parts();
            assert_eq!(parts.status, StatusCode::OK);
        }

        #[tokio::test]
        async fn dispatch_400_does_not_retry_and_surfaces_error() {
            // 400 means the request itself is broken — retrying on a different
            // backend would just produce another 400.
            let only = ScriptedProvider::new("only", vec![Behavior::DispatchErr(http_err(400))]);
            let other = ScriptedProvider::new(
                "other",
                vec![Behavior::OkChunks(vec![ok_chunk("nope", Some("stop"))])],
            );
            let state = pool_state(&[only.clone(), other.clone()]);

            // We can't deterministically force pick order, so iterate enough
            // times to land on `only` first at least once and verify that
            // the request fails *without* falling through to `other`.
            let mut saw_only_failure = false;
            for _ in 0..32 {
                let result = chat_completions(
                    State(state.clone()),
                    HeaderMap::new(),
                    Json(chat_request(false)),
                )
                .await;
                match result {
                    Err(e) => {
                        assert_eq!(
                            e.status,
                            StatusCode::BAD_GATEWAY,
                            "non-retryable upstream errors render as 502"
                        );
                        assert!(e.message.contains("400"));
                        saw_only_failure = true;
                    }
                    Ok(_) => {
                        // Random pick chose `other` first; that's fine, just
                        // re-arm the bad provider for the next iteration.
                    }
                }
                only.behaviors
                    .lock()
                    .unwrap()
                    .push_back(Behavior::DispatchErr(http_err(400)));
                other
                    .behaviors
                    .lock()
                    .unwrap()
                    .push_back(Behavior::OkChunks(vec![ok_chunk("nope", Some("stop"))]));
            }
            assert!(
                saw_only_failure,
                "expected at least one iteration to surface the 400 from `only` with no retry"
            );
            assert_eq!(
                other.calls(),
                32 - bad_failures(&only) as usize,
                "the good backend should never be called as a fallback for a non-retryable 400"
            );
        }

        /// Helper: count how many times the provider was poked.
        fn bad_failures(p: &ScriptedProvider) -> u32 {
            p.calls() as u32
        }

        #[tokio::test]
        async fn single_backend_retryable_failure_surfaces_original_error() {
            // Only one backend. The retry pick has nothing to fall back to,
            // so the original 503 surfaces to the client.
            let only = ScriptedProvider::new("only", vec![Behavior::DispatchErr(http_err(503))]);
            let state = pool_state(std::slice::from_ref(&only));
            let result =
                chat_completions(State(state), HeaderMap::new(), Json(chat_request(false))).await;
            let err = result.expect_err("503 with no fallback should error");
            assert_eq!(err.status, StatusCode::BAD_GATEWAY);
            assert!(
                err.message.contains("503"),
                "should surface the original status: {}",
                err.message
            );
            assert_eq!(
                only.calls(),
                1,
                "no retry attempt should fire when there's no other backend",
            );
        }

        #[tokio::test]
        async fn both_backends_503_surfaces_error() {
            let a = ScriptedProvider::new("a", vec![Behavior::DispatchErr(http_err(503))]);
            let b = ScriptedProvider::new("b", vec![Behavior::DispatchErr(http_err(503))]);
            let state = pool_state(&[a.clone(), b.clone()]);
            let result =
                chat_completions(State(state), HeaderMap::new(), Json(chat_request(false))).await;
            let err = result.expect_err("both backends failing should error");
            assert_eq!(err.status, StatusCode::BAD_GATEWAY);
            assert!(err.message.contains("503"));
            assert_eq!(
                a.calls() + b.calls(),
                2,
                "exactly two attempts (initial + one retry); not three or more",
            );
        }

        #[tokio::test]
        async fn mid_stream_failure_emits_error_frame_and_no_done() {
            // Stream yields one good chunk, then errors. Because data has
            // already been flushed to the client we cannot retry — instead
            // we must emit a `data: {"error":...}` frame and close without
            // a trailing `[DONE]` per plan.md §5.2.1.
            let provider = ScriptedProvider::new(
                "mid",
                vec![Behavior::ChunksThenErr(
                    vec![ok_chunk("partial ", None)],
                    http_err(500),
                )],
            );
            let state = pool_state(&[provider]);
            let resp = chat_completions(State(state), HeaderMap::new(), Json(chat_request(true)))
                .await
                .expect("the headers and first frame succeed even if the stream then fails");
            let (parts, body) = resp.into_parts();
            assert_eq!(parts.status, StatusCode::OK);
            let bytes = to_bytes(body, usize::MAX).await.unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(
                text.contains("\"content\":\"partial \""),
                "first chunk must be flushed: {text}"
            );
            assert!(
                text.contains("\"type\":\"upstream_error\""),
                "mid-stream error must be emitted as a data frame: {text}"
            );
            assert!(
                !text.contains("[DONE]"),
                "no [DONE] marker on a stream that errored: {text}"
            );
        }

        #[tokio::test]
        async fn streaming_first_chunk_503_retries_and_serves_alternate() {
            let bad = ScriptedProvider::new("bad", vec![Behavior::FirstChunkErr(http_err(503))]);
            let good = ScriptedProvider::new(
                "good",
                vec![Behavior::OkChunks(vec![ok_chunk("ok", Some("stop"))])],
            );
            let state = pool_state(&[bad, good]);
            let resp = chat_completions(State(state), HeaderMap::new(), Json(chat_request(true)))
                .await
                .expect("retry should produce a streaming response");
            let (parts, body) = resp.into_parts();
            assert_eq!(parts.status, StatusCode::OK);
            let bytes = to_bytes(body, usize::MAX).await.unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(
                text.contains("\"content\":\"ok\""),
                "served from `good`: {text}"
            );
            assert!(
                text.contains("[DONE]"),
                "successful stream must close with [DONE]: {text}"
            );
        }
    }

    // ── Structured logging ─────────────────────────────────────────────────

    mod logs {
        use super::*;
        use crate::logging::test_support::{CapturedWriter, json_subscriber};

        /// Drive a single non-streaming request through `chat_completions` with
        /// the JSON subscriber installed, and return the captured stderr
        /// output. We use `with_default` + a current-thread tokio runtime so
        /// the subscriber remains the default across await points.
        fn run_with_json_subscriber<F>(f: F) -> String
        where
            F: FnOnce() + Send,
        {
            let writer = CapturedWriter::new();
            let sub = json_subscriber(writer.clone());
            tracing::subscriber::with_default(sub, f);
            writer.contents()
        }

        #[test]
        fn request_emits_close_event_with_routing_fields() {
            let output = run_with_json_subscriber(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let state = test_state(sample_chunks());
                    let _ =
                        chat_completions(State(state), HeaderMap::new(), Json(chat_request(false)))
                            .await
                            .expect("non-streaming request succeeds");
                });
            });

            // The subscriber is configured with FmtSpan::CLOSE, which emits a
            // synthetic event when the `chat_completion` span exits. That is
            // the one line we guarantee carries every routing field.
            // Span close events emit with the span's fields nested under
            // `span` (per `tracing-subscriber`'s JSON format) and
            // `message: "close"`. That's the one request-scoped line we
            // guarantee carries every routing field.
            let close_line = output
                .lines()
                .find(|l| {
                    l.contains("\"message\":\"close\"")
                        && l.contains("\"name\":\"chat_completion\"")
                })
                .unwrap_or_else(|| panic!("no chat_completion close event in log:\n{output}"));

            let parsed: serde_json::Value = serde_json::from_str(close_line).unwrap();
            let span = parsed
                .get("span")
                .and_then(|v| v.as_object())
                .unwrap_or_else(|| panic!("span object missing: {close_line}"));

            assert_eq!(
                span.get("mixer_model").and_then(|v| v.as_str()),
                Some("mixer"),
            );
            assert_eq!(span.get("provider").and_then(|v| v.as_str()), Some("stub"),);
            assert_eq!(
                span.get("provider_model").and_then(|v| v.as_str()),
                Some("m"),
            );
            assert_eq!(span.get("stream").and_then(|v| v.as_bool()), Some(false));
            assert_eq!(
                span.get("has_images").and_then(|v| v.as_bool()),
                Some(false),
            );
            assert_eq!(span.get("status_code").and_then(|v| v.as_u64()), Some(200),);
            assert!(
                span.get("duration_ms").and_then(|v| v.as_u64()).is_some(),
                "duration_ms must be a number: {close_line}",
            );
        }

        #[test]
        fn every_emitted_json_line_is_parseable() {
            // Ensures the JSON format never produces partial/malformed lines
            // regardless of which code path fired (route decisions, span
            // close, child events).
            let output = run_with_json_subscriber(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let state = test_state(sample_chunks());
                    let _ =
                        chat_completions(State(state), HeaderMap::new(), Json(chat_request(true)))
                            .await
                            .expect("streaming request succeeds");
                });
            });
            let mut saw_any = false;
            for line in output.lines().filter(|l| !l.is_empty()) {
                let _: serde_json::Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("invalid JSON `{line}`: {e}"));
                saw_any = true;
            }
            assert!(
                saw_any,
                "expected at least one JSON log line, got:\n{output}",
            );
        }
    }

    // ── Bearer-token gate on /v1/* ─────────────────────────────────────────

    mod bearer {
        use super::*;
        use axum::body::Body;
        use axum::http::{self, Request};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tower::ServiceExt;

        // Hand out a unique env var name per test so parallel tests don't
        // stomp on each other's process environment.
        static ENV_SEQ: AtomicUsize = AtomicUsize::new(0);
        fn unique_env_name() -> String {
            let n = ENV_SEQ.fetch_add(1, Ordering::SeqCst);
            format!("MIXER_TEST_BEARER_{n}")
        }

        fn state_with_bearer(env_name: Option<String>) -> AppState {
            let mut state = test_state(sample_chunks());
            let mut cfg: Config = (*state.config).clone();
            cfg.listen_bearer_token_env = env_name;
            state.config = Arc::new(cfg);
            state
        }

        fn chat_body_json() -> String {
            serde_json::to_string(&chat_request(false)).unwrap()
        }

        // ── Pure unit tests ─────────────────────────────────────────────────

        #[test]
        fn constant_time_eq_matches_equal_bytes() {
            assert!(constant_time_eq(b"hello", b"hello"));
            assert!(constant_time_eq(b"", b""));
        }

        #[test]
        fn constant_time_eq_rejects_different_bytes_of_same_length() {
            assert!(!constant_time_eq(b"hello", b"hellz"));
        }

        #[test]
        fn constant_time_eq_rejects_different_lengths() {
            assert!(!constant_time_eq(b"hello", b"helloworld"));
            assert!(!constant_time_eq(b"", b"x"));
        }

        #[test]
        fn verify_bearer_passes_when_gate_disabled() {
            assert_eq!(verify_bearer(None, None), BearerOutcome::Disabled);
            assert_eq!(
                verify_bearer(None, Some("Bearer whatever")),
                BearerOutcome::Disabled,
            );
        }

        #[test]
        fn verify_bearer_allows_matching_token() {
            assert_eq!(
                verify_bearer(Some("secret"), Some("Bearer secret")),
                BearerOutcome::Allowed,
            );
        }

        #[test]
        fn verify_bearer_denies_wrong_token() {
            assert_eq!(
                verify_bearer(Some("secret"), Some("Bearer wrong")),
                BearerOutcome::Denied,
            );
        }

        #[test]
        fn verify_bearer_denies_missing_header() {
            assert_eq!(verify_bearer(Some("secret"), None), BearerOutcome::Denied,);
        }

        #[test]
        fn verify_bearer_denies_non_bearer_scheme() {
            assert_eq!(
                verify_bearer(Some("secret"), Some("Basic secret")),
                BearerOutcome::Denied,
            );
        }

        // ── Router-level tests through the full middleware stack ───────────

        async fn send(
            state: AppState,
            auth: Option<&str>,
            uri: &str,
            method: &str,
        ) -> (StatusCode, Vec<u8>) {
            let app = build_router(state);
            let builder = Request::builder().method(method).uri(uri);
            let builder = match auth {
                Some(v) => builder.header(http::header::AUTHORIZATION, v),
                None => builder,
            };
            let body = if method == "POST" {
                Body::from(chat_body_json())
            } else {
                Body::empty()
            };
            let req = builder
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(body)
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            let (parts, body) = resp.into_parts();
            let bytes = to_bytes(body, usize::MAX).await.unwrap().to_vec();
            (parts.status, bytes)
        }

        #[tokio::test]
        async fn valid_bearer_token_returns_200() {
            let name = unique_env_name();
            // SAFETY: tests run in-process and share the env, but each test
            // picks a unique var name (see ENV_SEQ) so parallel runs don't
            // observe each other's writes.
            unsafe {
                std::env::set_var(&name, "s3cret");
            }
            let state = state_with_bearer(Some(name.clone()));
            let (status, _body) =
                send(state, Some("Bearer s3cret"), "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::OK);
            unsafe {
                std::env::remove_var(&name);
            }
        }

        #[tokio::test]
        async fn missing_authorization_header_returns_401_openai_body() {
            let name = unique_env_name();
            unsafe {
                std::env::set_var(&name, "s3cret");
            }
            let state = state_with_bearer(Some(name.clone()));
            let (status, body) = send(state, None, "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(parsed["error"]["type"], "authentication_error");
            assert_eq!(
                parsed["error"]["message"],
                "missing or invalid bearer token"
            );
            unsafe {
                std::env::remove_var(&name);
            }
        }

        #[tokio::test]
        async fn wrong_bearer_token_returns_401() {
            let name = unique_env_name();
            unsafe {
                std::env::set_var(&name, "s3cret");
            }
            let state = state_with_bearer(Some(name.clone()));
            let (status, _body) =
                send(state, Some("Bearer nope"), "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            unsafe {
                std::env::remove_var(&name);
            }
        }

        #[tokio::test]
        async fn env_var_unset_disables_the_gate() {
            let name = unique_env_name();
            // Name the env var in config but deliberately leave it unset — the
            // middleware must pass through, matching today's unauthenticated
            // default.
            unsafe {
                std::env::remove_var(&name);
            }
            let state = state_with_bearer(Some(name));
            let (status, _body) = send(state, None, "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::OK);
        }

        #[tokio::test]
        async fn empty_env_var_disables_the_gate() {
            // Treat an empty string as "not set" — mirrors api_key_env
            // semantics so users can `unset MIXER_BEARER` or
            // `MIXER_BEARER=` interchangeably.
            let name = unique_env_name();
            unsafe {
                std::env::set_var(&name, "");
            }
            let state = state_with_bearer(Some(name.clone()));
            let (status, _body) = send(state, None, "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::OK);
            unsafe {
                std::env::remove_var(&name);
            }
        }

        #[tokio::test]
        async fn config_unset_means_no_gate() {
            let state = state_with_bearer(None);
            let (status, _body) = send(state, None, "/v1/chat/completions", "POST").await;
            assert_eq!(status, StatusCode::OK);
        }

        #[tokio::test]
        async fn healthz_bypasses_gate_when_enabled() {
            let name = unique_env_name();
            unsafe {
                std::env::set_var(&name, "s3cret");
            }
            let state = state_with_bearer(Some(name.clone()));
            let (status, body) = send(state, None, "/healthz", "GET").await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, b"ok");
            unsafe {
                std::env::remove_var(&name);
            }
        }

        #[tokio::test]
        async fn healthz_bypasses_gate_when_disabled() {
            let state = state_with_bearer(None);
            let (status, body) = send(state, None, "/healthz", "GET").await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, b"ok");
        }

        #[tokio::test]
        async fn non_loopback_bind_without_token_logs_warning() {
            use crate::logging::test_support::{CapturedWriter, text_subscriber};

            let writer = CapturedWriter::new();
            let sub = text_subscriber(writer.clone());
            let bound: std::net::SocketAddr = "0.0.0.0:4141".parse().unwrap();
            let cfg = Config {
                listen_bearer_token_env: None,
                ..Config::default()
            };
            tracing::subscriber::with_default(sub, || {
                maybe_warn_unprotected_bind(&cfg, &bound);
            });
            let out = writer.contents();
            assert!(
                out.contains("non-loopback"),
                "expected warning about non-loopback bind without a token, got:\n{out}",
            );
        }

        #[tokio::test]
        async fn loopback_bind_without_token_is_silent() {
            use crate::logging::test_support::{CapturedWriter, text_subscriber};

            let writer = CapturedWriter::new();
            let sub = text_subscriber(writer.clone());
            let bound: std::net::SocketAddr = "127.0.0.1:4141".parse().unwrap();
            let cfg = Config::default();
            tracing::subscriber::with_default(sub, || {
                maybe_warn_unprotected_bind(&cfg, &bound);
            });
            let out = writer.contents();
            assert!(
                !out.contains("non-loopback"),
                "loopback bind should not warn, got:\n{out}",
            );
        }

        #[tokio::test]
        async fn non_loopback_bind_with_token_is_silent() {
            use crate::logging::test_support::{CapturedWriter, text_subscriber};

            let name = unique_env_name();
            unsafe {
                std::env::set_var(&name, "s3cret");
            }
            let writer = CapturedWriter::new();
            let sub = text_subscriber(writer.clone());
            let bound: std::net::SocketAddr = "0.0.0.0:4141".parse().unwrap();
            let cfg = Config {
                listen_bearer_token_env: Some(name.clone()),
                ..Config::default()
            };
            tracing::subscriber::with_default(sub, || {
                maybe_warn_unprotected_bind(&cfg, &bound);
            });
            let out = writer.contents();
            assert!(
                !out.contains("non-loopback"),
                "non-loopback bind with a token should not warn, got:\n{out}",
            );
            unsafe {
                std::env::remove_var(&name);
            }
        }
    }
}
