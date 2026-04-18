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

use std::convert::Infallible;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::{StreamExt, stream};
use serde_json::json;
use tokio::net::TcpListener;

use crate::concurrency::ConcurrencyLimits;
use crate::config::{Config, ProviderSettings};
use crate::credentials::CredentialStore;
use crate::openai::{
    ChatRequest, ChatResponse, ModelListEntry, ModelListResponse, request_has_images,
};
use crate::providers::ProviderRegistry;
use crate::router;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub registry: Arc<ProviderRegistry>,
    pub credentials: Arc<CredentialStore>,
    pub concurrency: ConcurrencyLimits,
    /// When `Some`, only this mixer model name is served; all others return
    /// 404. Used by `mixer serve --model <name>`.
    pub pinned_model: Option<String>,
}

pub async fn serve(state: AppState, listen_addr: &str) -> Result<()> {
    let app = build_router(state);
    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("binding to {listen_addr}"))?;
    let actual = listener.local_addr()?;
    eprintln!("mixer listening on http://{actual}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve")?;
    Ok(())
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("shutdown signal received, draining");
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
    Json(mut req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    let wants_stream = req.stream.unwrap_or(false);

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

    let requires_images = request_has_images(&req);

    let decision = router::pick(
        &state.config,
        &state.registry,
        &state.credentials,
        mixer_model,
        requires_images,
    )
    .await
    .map_err(AppError::from_provider)?;

    eprintln!(
        "[route] mixer_model={mixer_model_name} -> provider={} model={} (images={requires_images})",
        decision.provider_id, decision.provider_model
    );

    // Rewrite the user-facing mixer model name with the provider-native one.
    req.model = decision.provider_model.clone();

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

    let chunks = provider
        .chat_completion(&state.credentials, &settings, req)
        .await
        .map_err(AppError::from_provider)?;

    if wants_stream {
        Ok(sse_response(chunks, permit).into_response())
    } else {
        let mut collected = Vec::new();
        let mut s = chunks;
        while let Some(next) = s.next().await {
            collected.push(next.map_err(AppError::from_provider)?);
        }
        drop(permit);
        let response = ChatResponse::from_chunks(collected);
        Ok(Json(response).into_response())
    }
}

/// Convert a provider chunk stream into an SSE response. Each chunk becomes a
/// `data: {json}` frame; errors mid-stream are emitted as `data: {"error":...}`
/// (matching OpenAI's on-the-wire error convention for streams) and then end
/// the stream. A final `data: [DONE]` frame closes a successful stream.
fn sse_response(
    chunks: crate::providers::ChatStream,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let data_stream = chunks.flat_map(|item| match item {
        Ok(chunk) => {
            let payload = serde_json::to_string(&chunk).unwrap_or_else(|e| {
                json!({"error": {"message": format!("chunk serialize failed: {e}"), "type": "mixer_internal_error"}}).to_string()
            });
            stream::iter(vec![Ok(Event::default().data(payload))])
        }
        Err(e) => {
            let payload = json!({
                "error": {
                    "message": format!("{e:#}"),
                    "type": "mixer_upstream_error",
                }
            })
            .to_string();
            stream::iter(vec![Ok(Event::default().data(payload))])
        }
    });

    // Drop the permit when the stream ends (either normally or after DONE).
    let done = stream::once(async move {
        drop(permit);
        Ok::<_, Infallible>(Event::default().data("[DONE]"))
    });

    Sse::new(data_stream.chain(done)).keep_alive(KeepAlive::default())
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
            vec![ModelInfo {
                id: "m",
                display_name: "M",
                supports_images: false,
            }]
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
                content: MessageContent::Text("hi".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: Some(stream),
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
            extra: Default::default(),
        }
    }

    #[tokio::test]
    async fn non_streaming_returns_accumulated_json() {
        let state = test_state(sample_chunks());
        let resp = chat_completions(State(state), Json(chat_request(false)))
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
            MessageContent::Text(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected text content"),
        }
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn streaming_returns_sse_with_done_marker() {
        let state = test_state(sample_chunks());
        let resp = chat_completions(State(state), Json(chat_request(true)))
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
}
