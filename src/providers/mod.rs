//! Provider registry + the [`Provider`] trait every subscription backend
//! implements.
//!
//! **Adding a provider:** drop a new file in `src/providers/`, implement
//! [`Provider`], and register it in [`builtin_registry`]. Everything else —
//! routing, auth storage, usage-weighted picks, per-provider concurrency
//! caps, image-capability filtering — is handled generically.

pub mod codex;
pub mod common;
pub mod glm;
pub mod kimi_api;
pub mod kimi_code;
pub mod minimax;
pub mod ollama;
pub mod opencode;

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::Stream;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::{ChatCompletionChunk, ChatRequest};
use crate::usage::UsageSnapshot;

/// Streaming output of [`Provider::chat_completion`]. The server adapts this
/// to either an SSE response (when the client asked for `stream: true`) or a
/// single accumulated JSON response.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;

/// What a provider's model can process. A model is omitted from the pool for
/// image-bearing requests when `supports_images == false`.
///
/// `id` and `display_name` are `Cow<'static, str>` so static built-in catalogues
/// stay zero-alloc (`Cow::Borrowed`) while providers that fetch their catalogue
/// at runtime — e.g. Ollama — can supply owned strings without leaking memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: Cow<'static, str>,
    pub display_name: Cow<'static, str>,
    pub supports_images: bool,
    /// Total context window in tokens (input + output), as the provider
    /// advertises it. Consumed by the router's context-window exclusion filter.
    pub context_window: u32,
    /// Which chain-of-thought dialect this model emits on the wire. Consumed by
    /// [`crate::reasoning`] to normalize every model's output into the canonical
    /// `delta.reasoning_content` field before the server renders it to the client.
    pub reasoning_format: ReasoningFormat,
}

/// Per-model declaration of the upstream chain-of-thought dialect. Each
/// provider's `models()` sets this once; `crate::reasoning` does the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// Model doesn't produce chain-of-thought output, or does but it isn't
    /// separable from the primary content stream. Passthrough.
    #[default]
    None,
    /// DeepSeek-style convention adopted by Kimi, GLM, MiniMax M2, etc.: the
    /// upstream emits reasoning on `delta.reasoning_content`. Passthrough —
    /// serde deserializes it into the typed field automatically.
    Structured,
    /// Qwen-style: reasoning arrives inline inside `delta.content`, wrapped in
    /// `<think>…</think>` tags. The normalizer strips the tags and routes the
    /// enclosed bytes to `delta.reasoning_content`.
    InlineThinkTags,
    /// OpenAI Responses API reasoning-summary events
    /// (`response.reasoning_summary_text.delta` / `response.reasoning_text.delta`).
    /// Translation happens inside `responses_api.rs`; the normalization stage is
    /// a passthrough here.
    ResponsesApiSummary,
}

impl ModelInfo {
    pub fn new(
        id: impl Into<Cow<'static, str>>,
        display_name: impl Into<Cow<'static, str>>,
        supports_images: bool,
        context_window: u32,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            supports_images,
            context_window,
            reasoning_format: ReasoningFormat::None,
        }
    }

    /// Declare the upstream reasoning dialect for this model. Chains on top of
    /// [`ModelInfo::new`] so existing call sites that don't care about reasoning
    /// need no change.
    pub fn with_reasoning(mut self, format: ReasoningFormat) -> Self {
        self.reasoning_format = format;
        self
    }
}

/// A single entry from a provider's live `/v1/models` listing. Returned by
/// [`Provider::list_remote_models`]. Kept deliberately thin: `id` is the only
/// field the OpenAI `/v1/models` contract guarantees; `raw` preserves the
/// unparsed JSON so the CLI can surface whatever additional metadata a
/// provider chooses to publish (`context_length`, `owned_by`, etc.) without
/// forcing a schema decision here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteModelEntry {
    pub id: String,
    pub raw: serde_json::Value,
}

/// How a provider authenticates. Drives `mixer auth status` output and lets
/// the CLI know whether `api_key_env` is applicable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthKind {
    /// Static API key — may be sourced from `settings.api_key_env` or the
    /// stored credentials file. See `CredentialStore::load_api_key`.
    ApiKey,
    /// OAuth device-authorization flow with refreshable access tokens.
    DeviceFlow,
    /// No authentication — e.g. a self-hosted ollama server reached over the
    /// local network. `mixer auth` treats these as always authenticated and
    /// skips credential bookkeeping.
    None,
}

#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable identifier used in config (`backends[].provider`) and on the
    /// command line (`mixer login <id>`).
    fn id(&self) -> &'static str;

    /// Human-readable name shown in `mixer providers list`.
    fn display_name(&self) -> &'static str;

    /// The complete catalogue of models this provider can serve, regardless
    /// of whether the user has enabled all of them in config.
    fn models(&self) -> Vec<ModelInfo>;

    /// How this provider authenticates. Informs `mixer auth status` about
    /// whether env-var sources and OAuth freshness details apply.
    fn auth_kind(&self) -> AuthKind;

    /// Whether the user has credentials that look currently usable. Providers
    /// should lean toward "yes" here — a real probe happens on dispatch.
    ///
    /// Default behavior inspects the stored file only, which is enough for
    /// device-flow providers to override meaningfully and for API-key
    /// providers to consult `settings.api_key_env`.
    fn is_authenticated(&self, store: &CredentialStore, _settings: &ProviderSettings) -> bool {
        store.exists(self.id())
    }

    /// Interactive login flow. Prompts on stdin/stderr, writes credentials
    /// via `store.save(self.id(), ...)`.
    async fn login(&self, store: &CredentialStore) -> Result<()>;

    /// Remove stored credentials (and any server-side session if applicable).
    async fn logout(&self, store: &CredentialStore) -> Result<()> {
        store.remove(self.id())
    }

    /// Best-effort snapshot of current subscription consumption. `None` means
    /// the provider does not (yet) expose this.
    ///
    /// Called lazily by the usage-aware router on each pick (no background
    /// polling, no caching). Transient endpoint failures should degrade to
    /// `Ok(None)` rather than propagate, so a broken telemetry endpoint
    /// doesn't cascade into a routing failure.
    async fn usage(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(None)
    }

    /// Execute a chat completion. `req.model` has already been rewritten by
    /// the router to the provider-native model id; `settings` carries the
    /// user's overrides (base URL, timeout, etc.). Providers always return a
    /// stream of [`ChatCompletionChunk`]s; the server collapses them into a
    /// single response for non-streaming clients.
    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream>;

    /// Ask the upstream to list the models it currently serves, by hitting
    /// its OpenAI-compatible `GET /v1/models` endpoint (or equivalent).
    ///
    /// Returning `Ok(None)` means the provider does not offer a live catalogue
    /// — either the wire protocol is different (codex's Responses API), or the
    /// vendor simply hasn't shipped one (opencode Zen, z.ai). The CLI treats
    /// `None` as "unsupported" and prints the hardcoded catalogue with a note.
    ///
    /// `Err` is reserved for "the call itself failed" — network errors,
    /// non-2xx upstream, unparseable response. Those should surface so the
    /// user can see what went wrong rather than silently falling through.
    async fn list_remote_models(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
    ) -> Result<Option<Vec<RemoteModelEntry>>> {
        Ok(None)
    }
}

/// A registry that owns each [`Provider`] via `Arc<dyn Provider>` and offers
/// name-based lookup. Cheap to clone.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, provider: Arc<dyn Provider>) {
        self.providers.insert(provider.id().to_string(), provider);
    }

    pub fn get(&self, id: &str) -> Result<Arc<dyn Provider>> {
        self.providers
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown provider `{id}`"))
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<dyn Provider>)> {
        self.providers.iter()
    }

    pub fn ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.providers.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }
}

/// The built-in set of subscription-based providers mixer ships with.
pub fn builtin_registry() -> ProviderRegistry {
    let mut r = ProviderRegistry::new();
    r.register(Arc::new(codex::CodexProvider));
    r.register(Arc::new(minimax::MinimaxProvider));
    r.register(Arc::new(glm::GlmProvider));
    r.register(Arc::new(opencode::OpencodeProvider));
    r.register(Arc::new(kimi_code::KimiCodeProvider));
    r.register(Arc::new(kimi_api::KimiApiProvider));
    r.register(Arc::new(ollama::OllamaProvider));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_has_all_providers() {
        let r = builtin_registry();
        let ids = r.ids();
        assert_eq!(
            ids,
            vec![
                "codex",
                "glm",
                "kimi-api",
                "kimi-code",
                "minimax",
                "ollama",
                "opencode",
            ],
        );
    }

    #[test]
    fn get_unknown_provider_errors() {
        let r = builtin_registry();
        let err = r.get("not-a-real-provider").err().expect("expected error");
        assert!(err.to_string().contains("unknown provider"));
    }

    #[test]
    fn model_info_new_stores_context_window() {
        let info = ModelInfo::new("x", "X", false, 1024);
        assert_eq!(info.context_window, 1024);
    }
}
