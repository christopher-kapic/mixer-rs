//! Minimax subscription provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `https://api.minimax.chat/v1` (override via
//! `providers.minimax.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`.
//!
//! Login stores the pasted API key at `~/.config/mixer/credentials/minimax.json`
//! with permissions 0600. At dispatch time the key resolves env-var-first via
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
use crate::usage::UsageSnapshot;

const DEFAULT_BASE_URL: &str = "https://api.minimax.chat/v1";
const CHAT_PATH: &str = "/chat/completions";

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
            ModelInfo::new("MiniMax-M2", "MiniMax M2", false),
            ModelInfo::new("MiniMax-M2-vl", "MiniMax M2 (vision)", true),
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
        eprintln!("Generate a Minimax API key at https://platform.minimaxi.com/");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    /// Minimax exposes no public usage/quota endpoint — subscription
    /// consumption is only visible inside the platform dashboard.
    async fn usage(
        &self,
        _store: &CredentialStore,
        _settings: &ProviderSettings,
    ) -> Result<Option<UsageSnapshot>> {
        Ok(None)
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let api_key = store.load_api_key(self.id(), settings).ok_or_else(|| {
            anyhow!("minimax is not authenticated; run `mixer auth login minimax`")
        })?;
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
        store
            .save("minimax", &json!({ "api_key": "sk-minimax-test" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"resp-1","object":"chat.completion.chunk","created":1,"model":"MiniMax-M2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        let stream = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("no stream errors");
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-minimax-test")
        );
        let body = mock.captured.body();
        assert!(
            body.contains("\"stream\":true"),
            "stream flag should be forced: {body}"
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("minimax", &json!({ "api_key": "wrong" }))
            .unwrap();

        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"invalid api key"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be an AuthenticationError");
        assert!(auth.message.contains("mixer auth login minimax"));
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        // Neither env nor file has a key.
        let settings = ProviderSettings::default_enabled();
        let result = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }

    #[tokio::test]
    async fn usage_returns_none() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let snap = MinimaxProvider.usage(&store, &settings).await.unwrap();
        assert!(
            snap.is_none(),
            "minimax has no public usage endpoint — must return Ok(None)"
        );
    }

    #[tokio::test]
    async fn chat_completion_prefers_env_var_over_stored_file() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("minimax", &json!({ "api_key": "sk-from-file" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"resp-1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let var = "MIXER_TEST_MINIMAX_KEY";
        // SAFETY: tests share no global state with this unique var name.
        unsafe { std::env::set_var(var, "sk-from-env") };
        let settings = ProviderSettings {
            base_url: Some(format!("http://{}", mock.addr)),
            api_key_env: Some(var.to_string()),
            ..ProviderSettings::default_enabled()
        };
        let stream = MinimaxProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let _: Vec<_> = stream.collect().await;
        unsafe { std::env::remove_var(var) };

        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-from-env"),
            "env var should shadow the stored file key",
        );
    }
}
