//! Minimax subscription provider.
//!
//! Status: scaffold.

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::{ChatRequest, ChatResponse};
use crate::providers::{ModelInfo, Provider};

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

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        // TODO: prompt for the subscription API key and persist it.
        bail!("minimax login not yet implemented")
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatResponse> {
        // TODO: POST to the Minimax /v1/chat/completions endpoint with the
        //       stored key and map the response body.
        bail!("minimax chat_completion not yet implemented")
    }
}
