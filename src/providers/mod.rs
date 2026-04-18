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
pub mod minimax;
pub mod opencode;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub supports_images: bool,
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
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_has_four_providers() {
        let r = builtin_registry();
        let ids = r.ids();
        assert_eq!(ids, vec!["codex", "glm", "minimax", "opencode"]);
    }

    #[test]
    fn get_unknown_provider_errors() {
        let r = builtin_registry();
        let err = r.get("not-a-real-provider").err().expect("expected error");
        assert!(err.to_string().contains("unknown provider"));
    }
}
