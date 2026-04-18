//! Minimax subscription provider.
//!
//! Status: scaffold.

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use futures::stream;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider};

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
            ModelInfo {
                id: "MiniMax-M2",
                display_name: "MiniMax M2",
                supports_images: false,
            },
            ModelInfo {
                id: "MiniMax-M2-vl",
                display_name: "MiniMax M2 (vision)",
                supports_images: true,
            },
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

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        // TODO: prompt for the subscription API key and persist it.
        bail!("minimax login not yet implemented")
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatStream> {
        // TODO: POST to the Minimax /v1/chat/completions endpoint with the
        //       stored key and map the response body into ChatCompletionChunks.
        Ok(Box::pin(stream::once(async {
            Err(anyhow!("minimax chat_completion not yet implemented"))
        })))
    }
}
