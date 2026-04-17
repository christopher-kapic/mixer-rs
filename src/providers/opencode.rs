//! opencode.ai (opencode-go / opencode Pro) subscription provider.
//!
//! Status: scaffold.

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::{ChatRequest, ChatResponse};
use crate::providers::{ModelInfo, Provider};

pub struct OpencodeProvider;

#[async_trait]
impl Provider for OpencodeProvider {
    fn id(&self) -> &'static str {
        "opencode"
    }

    fn display_name(&self) -> &'static str {
        "opencode"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "anthropic/claude-sonnet-4-6",
                display_name: "Claude Sonnet 4.6 (via opencode)",
                supports_images: true,
            },
            ModelInfo {
                id: "openai/gpt-5.2",
                display_name: "GPT-5.2 (via opencode)",
                supports_images: true,
            },
        ]
    }

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        // TODO: run the opencode subscription auth flow (OAuth or token
        //       exchange) and persist the resulting credentials.
        bail!("opencode login not yet implemented")
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatResponse> {
        // TODO: POST to the opencode gateway and map the response.
        bail!("opencode chat_completion not yet implemented")
    }
}
