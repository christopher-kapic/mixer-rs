//! GLM (z.ai) subscription provider.
//!
//! Status: scaffold.

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use futures::stream;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::{ChatStream, ModelInfo, Provider};

pub struct GlmProvider;

#[async_trait]
impl Provider for GlmProvider {
    fn id(&self) -> &'static str {
        "glm"
    }

    fn display_name(&self) -> &'static str {
        "GLM (z.ai)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "glm-4.6",
                display_name: "GLM 4.6",
                supports_images: false,
            },
            ModelInfo {
                id: "glm-4.5v",
                display_name: "GLM 4.5V",
                supports_images: true,
            },
        ]
    }

    async fn login(&self, _store: &CredentialStore) -> Result<()> {
        // TODO: prompt for z.ai subscription key and persist it.
        bail!("glm login not yet implemented")
    }

    async fn chat_completion(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
        _req: ChatRequest,
    ) -> Result<ChatStream> {
        // TODO: POST to the z.ai Anthropic-compatible or OpenAI-compatible
        //       endpoint and map the response into ChatCompletionChunks.
        Ok(Box::pin(stream::once(async {
            Err(anyhow!("glm chat_completion not yet implemented"))
        })))
    }
}
