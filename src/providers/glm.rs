//! GLM (z.ai) subscription provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `https://api.z.ai/api/paas/v4` (override via
//! `providers.glm.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`.
//!
//! Login stores the pasted API key at `~/.config/mixer/credentials/glm.json`
//! (0600). At dispatch time the key resolves env-var-first via
//! [`CredentialStore::load_api_key`] so an environment-provided key wins over
//! anything on disk (see plan.md §3.6.2).

use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::api_key_login::prompt_and_store_api_key;
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider};

const DEFAULT_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
const CHAT_PATH: &str = "/chat/completions";

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

    fn auth_kind(&self) -> AuthKind {
        AuthKind::ApiKey
    }

    fn is_authenticated(&self, store: &CredentialStore, settings: &ProviderSettings) -> bool {
        store
            .load_api_key(self.id(), settings)
            .is_some_and(|s| !s.is_empty())
    }

    async fn login(&self, store: &CredentialStore) -> Result<()> {
        eprintln!("Generate a z.ai API key at https://z.ai/manage-apikey/apikey-list");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let api_key = store
            .load_api_key(self.id(), settings)
            .ok_or_else(|| anyhow!("glm is not authenticated; run `mixer auth login glm`"))?;
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        openai_client::chat_completion(self.id(), &url, &api_key, AuthScheme::Bearer, timeout, req)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::common::oauth_refresh::AuthenticationError;
    use crate::providers::common::openai_client::test_support::{sample_request, start_mock_chat};
    use futures::StreamExt;
    use serde_json::json;
    use tempfile::TempDir;

    fn store_in(tmp: &TempDir) -> CredentialStore {
        CredentialStore::with_dir_for_tests(tmp.path().to_path_buf())
    }

    fn settings_with_base(base_url: &str) -> ProviderSettings {
        ProviderSettings {
            base_url: Some(base_url.to_string()),
            ..ProviderSettings::default_enabled()
        }
    }

    fn sse(payloads: &[&str]) -> String {
        payloads.iter().map(|p| format!("data: {p}\n\n")).collect()
    }

    #[tokio::test]
    async fn chat_completion_proxies_through_base_url_with_bearer_auth() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("glm", &json!({ "api_key": "sk-glm" })).unwrap();

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"glm-4.6","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let stream = GlmProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("no stream errors");
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-glm"),
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("glm", &json!({ "api_key": "wrong" })).unwrap();
        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"unauthorized"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = GlmProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be AuthenticationError");
        assert!(auth.message.contains("mixer auth login glm"));
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let result = GlmProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }
}
