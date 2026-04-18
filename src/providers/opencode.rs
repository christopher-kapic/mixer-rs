//! opencode.ai (opencode-go / opencode Pro) subscription provider.
//!
//! Status: scaffold.

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use futures::stream;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider};

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

    fn auth_kind(&self) -> AuthKind {
        AuthKind::ApiKey
    }

    fn is_authenticated(&self, store: &CredentialStore, settings: &ProviderSettings) -> bool {
        store
            .load_api_key(self.id(), settings)
            .is_some_and(|s| !s.is_empty())
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
    ) -> Result<ChatStream> {
        // TODO: POST to the opencode gateway and map the response into
        //       ChatCompletionChunks.
        Ok(Box::pin(stream::once(async {
            Err(anyhow!("opencode chat_completion not yet implemented"))
        })))
    }
}
