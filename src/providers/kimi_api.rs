//! Moonshot pay-per-token API provider ("kimi-api").
//!
//! Wire protocol: OpenAI-compatible Chat Completions.
//! Base URL: `https://api.moonshot.ai/v1` (override via
//! `providers.kimi-api.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`.
//!
//! This is the non-subscription path — users generate a key at
//! `platform.moonshot.ai` and pay per token. Compared to `kimi-code`, there
//! is no OAuth flow, no subscription quota, and no usage introspection
//! endpoint, so `usage()` always returns `None`.
//!
//! Kept as a sibling of `kimi-code` (rather than folded into it) because the
//! `AuthKind` is fundamentally different: one value per provider, used by
//! `mixer auth status` and the CLI dispatch. Users can enable either or
//! both; `kimi-api` is off by default so the subscription path is the
//! advertised one.

use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::api_key_login::prompt_and_store_api_key;
use crate::providers::common::models_list::fetch_openai_models;
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{
    AuthKind, ChatStream, ModelInfo, Provider, ReasoningFormat, RemoteModelEntry,
};

const DEFAULT_BASE_URL: &str = "https://api.moonshot.ai/v1";
const CHAT_PATH: &str = "/chat/completions";
const MODELS_PATH: &str = "/models";

pub struct KimiApiProvider;

#[async_trait]
impl Provider for KimiApiProvider {
    fn id(&self) -> &'static str {
        "kimi-api"
    }

    fn display_name(&self) -> &'static str {
        "Kimi (Moonshot API key)"
    }

    fn models(&self) -> Vec<ModelInfo> {
        // Sourced from the models.dev central catalogue (which opencode also
        // builds against). Moonshot's pay-per-token endpoint exposes the
        // specific K2.x checkpoints; the `kimi-code` subscription gateway is
        // a sibling provider that only addresses `kimi-for-coding`.
        vec![
            ModelInfo::new(
                "kimi-k2-0905-preview",
                "Kimi K2 (0905 preview)",
                false,
                256_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "kimi-k2-thinking-turbo",
                "Kimi K2 Thinking (turbo)",
                false,
                256_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("kimi-k2-thinking", "Kimi K2 Thinking", false, 256_000)
                .with_reasoning(ReasoningFormat::Structured),
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
        eprintln!("Generate a Moonshot API key at https://platform.moonshot.ai/");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> {
        let api_key = store.load_api_key(self.id(), settings).ok_or_else(|| {
            anyhow!("kimi-api is not authenticated; run `mixer auth login kimi-api`")
        })?;
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{CHAT_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        openai_client::chat_completion(
            self.id(),
            &url,
            &api_key,
            AuthScheme::Bearer,
            timeout,
            None,
            req,
        )
        .await
    }

    async fn list_remote_models(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
    ) -> Result<Option<Vec<RemoteModelEntry>>> {
        let api_key = store.load_api_key(self.id(), settings).ok_or_else(|| {
            anyhow!("kimi-api is not authenticated; run `mixer auth login kimi-api`")
        })?;
        let base = settings.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL);
        let url = format!("{}{MODELS_PATH}", base.trim_end_matches('/'));
        let timeout = settings.request_timeout_secs.map(Duration::from_secs);
        let entries = fetch_openai_models(
            self.id(),
            &url,
            &api_key,
            AuthScheme::Bearer,
            timeout,
            None,
        )
        .await?;
        Ok(Some(entries))
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
            .save("kimi-api", &json!({ "api_key": "sk-moonshot-test" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));

        let stream = KimiApiProvider
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
            Some("Bearer sk-moonshot-test")
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("kimi-api", &json!({ "api_key": "wrong" }))
            .unwrap();
        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"unauthorized"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = KimiApiProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be AuthenticationError");
        assert!(auth.message.contains("mixer auth login kimi-api"));
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let result = KimiApiProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }

    #[tokio::test]
    async fn chat_completion_prefers_env_var_over_stored_file() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("kimi-api", &json!({ "api_key": "sk-from-file" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"r","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let var = "MIXER_TEST_KIMI_API_KEY";
        unsafe { std::env::set_var(var, "sk-from-env") };
        let settings = ProviderSettings {
            base_url: Some(format!("http://{}", mock.addr)),
            api_key_env: Some(var.to_string()),
            ..ProviderSettings::default_enabled()
        };
        let stream = KimiApiProvider
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
