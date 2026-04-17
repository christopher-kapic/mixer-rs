//! Codex / ChatGPT subscription provider.
//!
//! Authenticates against an existing ChatGPT Plus/Pro session and forwards
//! chat completion requests through the Codex backend.
//!
//! Status: scaffold. `login` and `chat_completion` return `not yet
//! implemented` until the session-handshake and request-forwarding logic is
//! ported in.

use anyhow::{Result, bail};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::{ChatRequest, ChatResponse};
use crate::providers::{ModelInfo, Provider};

pub struct CodexProvider;

#[async_trait]
impl Provider for CodexProvider {
    fn id(&self) -> &'static str {
        "codex"
    }

    fn display_name(&self) -> &'static str {
        "Codex (ChatGPT Plus/Pro)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "gpt-5.2",
                display_name: "GPT-5.2",
                supports_images: true,
            },
            ModelInfo {
                id: "gpt-5.2-mini",
                display_name: "GPT-5.2 mini",
                supports_images: true,
            },
        ]
    }

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        // TODO: initiate the ChatGPT device-code / browser login flow, then
        //       store the resulting session token + refresh token via
        //       `store.save(self.id(), &value)`.
        bail!("codex login not yet implemented")
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatResponse> {
        // TODO: refresh the session if needed, POST to the Codex chat
        //       completions endpoint, and map the response into the
        //       OpenAI-compatible shape.
        bail!("codex chat_completion not yet implemented")
    }
}
