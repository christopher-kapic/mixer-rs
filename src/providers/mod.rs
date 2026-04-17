//! Provider registry + the [`Provider`] trait every subscription backend
//! implements.
//!
//! **Adding a provider:** drop a new file in `src/providers/`, implement
//! [`Provider`], and register it in [`builtin_registry`]. Everything else —
//! routing, auth storage, usage-weighted picks, per-provider concurrency
//! caps, image-capability filtering — is handled generically.

pub mod codex;
pub mod glm;
pub mod minimax;
pub mod opencode;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::{ChatRequest, ChatResponse};
use crate::usage::UsageSnapshot;

/// What a provider's model can process. A model is omitted from the pool for
/// image-bearing requests when `supports_images == false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub supports_images: bool,
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

    /// Whether the user has stored credentials that look usable. Providers
    /// should lean toward "yes" here — a real probe happens on dispatch.
    fn is_authenticated(&self, store: &CredentialStore) -> bool {
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
    async fn usage(&self, _store: &CredentialStore) -> Result<Option<UsageSnapshot>> {
        Ok(None)
    }

    /// Execute a chat completion. `req.model` has already been rewritten by
    /// the router to the provider-native model id; `settings` carries the
    /// user's overrides (base URL, timeout, etc.).
    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatResponse>;
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
