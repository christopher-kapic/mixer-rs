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
//!   7. Return the provider's response verbatim (currently non-streaming only;
//!      streaming returns 501 until we wire up SSE).

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
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
) -> Result<Json<ChatResponse>, AppError> {
    if req.stream.unwrap_or(false) {
        return Err(AppError::not_implemented(
            "streaming responses are not yet supported — set `stream: false`",
        ));
    }

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
    .map_err(AppError::bad_gateway)?;

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

    let _permit = state.concurrency.acquire(&decision.provider_id).await;

    let response = provider
        .chat_completion(&state.credentials, &settings, req)
        .await
        .map_err(AppError::bad_gateway)?;

    Ok(Json(response))
}

/// Converts anyhow errors into OpenAI-style JSON error bodies so downstream
/// OpenAI SDKs surface them naturally.
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

    fn not_implemented(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "mixer_not_implemented",
            message: msg.into(),
        }
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
