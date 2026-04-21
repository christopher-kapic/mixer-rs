//! opencode.ai Zen subscription provider.
//!
//! Wire protocol: OpenAI-compatible Chat Completions (the Zen gateway also
//! exposes Anthropic and Responses-API routes at sibling paths; we use
//! `/chat/completions`).
//! Base URL: `https://opencode.ai/zen/v1` (override via
//! `providers.opencode.base_url` in config).
//! Auth: `Authorization: Bearer <api_key>`. (Earlier releases of this provider
//! sent `x-api-key`; the Zen gateway still accepts `x-api-key` on the
//! `/models` endpoint but rejects it on `/chat/completions` with
//! `"Missing API key."` — verified live on 2026-04-21.)
//!
//! Login stores the pasted API key at
//! `~/.config/mixer/credentials/opencode.json` (0600). At dispatch time the
//! key resolves env-var-first via [`CredentialStore::load_api_key`], so an
//! environment-provided key wins over anything on disk (see plan.md §3.6.2).

use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::config::ProviderSettings;
use crate::credentials::CredentialStore;
use crate::openai::ChatRequest;
use crate::providers::common::api_key_login::prompt_and_store_api_key;
use crate::providers::common::openai_client::{self, AuthScheme};
use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider, ReasoningFormat};
use crate::usage::UsageSnapshot;

const DEFAULT_BASE_URL: &str = "https://opencode.ai/zen/v1";
const CHAT_PATH: &str = "/chat/completions";

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
        // Sourced from https://opencode.ai/docs/zen — opencode Zen publishes
        // model IDs as flat strings (no `<vendor>/` prefix). Older versions
        // of this catalogue used `anthropic/<id>` / `openai/<id>`; those no
        // longer match what the Zen gateway accepts.
        //
        // Qwen entries intentionally lead: no other provider mixer ships
        // with fronts a Qwen model, so opencode is the only route for users
        // who want Qwen in their pool. `qwen3.6-plus` is the current newest.
        //
        // Only non-deprecated entries are listed (as of 2026-04-21); Qwen3
        // Coder 480B, GLM 4.7/4.6 via Zen, Gemini 3 Pro, Kimi K2/K2 Thinking
        // via Zen, and MiniMax M2.1 via Zen are all past their deprecation
        // date per the Zen docs.
        // Reasoning dialect assignments follow the upstream convention per
        // model family: Qwen emits `<think>…</think>` inline in content, so
        // the normalizer strips the tags client-side. Every other Zen-fronted
        // model exposes reasoning via the structured `reasoning_content`
        // field once Zen translates the upstream's native format.
        vec![
            ModelInfo::new("qwen3.6-plus", "Qwen3.6 Plus (via opencode)", true, 128_000)
                .with_reasoning(ReasoningFormat::InlineThinkTags),
            ModelInfo::new("qwen3.5-plus", "Qwen3.5 Plus (via opencode)", true, 128_000)
                .with_reasoning(ReasoningFormat::InlineThinkTags),
            ModelInfo::new(
                "claude-opus-4-7",
                "Claude Opus 4.7 (via opencode)",
                true,
                1_000_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "claude-sonnet-4-6",
                "Claude Sonnet 4.6 (via opencode)",
                true,
                1_000_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "claude-haiku-4-5",
                "Claude Haiku 4.5 (via opencode)",
                true,
                1_000_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("gpt-5.4", "GPT-5.4 (via opencode)", true, 400_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("gpt-5.4-mini", "GPT-5.4 Mini (via opencode)", true, 400_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "gpt-5.3-codex",
                "GPT-5.3 Codex (via opencode)",
                true,
                400_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("gpt-5.2", "GPT-5.2 (via opencode)", true, 400_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "gemini-3.1-pro",
                "Gemini 3.1 Pro (via opencode)",
                true,
                1_000_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "gemini-3-flash",
                "Gemini 3 Flash (via opencode)",
                true,
                1_000_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new(
                "minimax-m2.5",
                "MiniMax M2.5 (via opencode)",
                false,
                192_000,
            )
            .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("glm-5.1", "GLM 5.1 (via opencode)", false, 200_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("glm-5", "GLM 5 (via opencode)", false, 200_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("kimi-k2.6", "Kimi K2.6 (via opencode)", false, 256_000)
                .with_reasoning(ReasoningFormat::Structured),
            ModelInfo::new("kimi-k2.5", "Kimi K2.5 (via opencode)", false, 256_000)
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
        eprintln!("Sign in at https://opencode.ai/auth to generate a Zen API key");
        prompt_and_store_api_key(store, self.id(), self.display_name())
    }

    /// The Zen gateway tracks usage server-side but offers no client-facing
    /// quota endpoint (verified against opencode's own source — usage is only
    /// surfaced via per-request response metadata).
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
            anyhow!("opencode is not authenticated; run `mixer auth login opencode`")
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
    async fn chat_completion_uses_bearer_auth() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("opencode", &json!({ "api_key": "sk-zen" }))
            .unwrap();

        let body = sse(&[
            r#"{"id":"r","object":"chat.completion.chunk","created":1,"model":"claude-sonnet-4-6","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = start_mock_chat(200, "text/event-stream", body).await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let stream = OpencodeProvider
            .chat_completion(&store, &settings, sample_request())
            .await
            .unwrap();
        let _: Vec<_> = stream.collect().await;

        assert_eq!(
            mock.captured.header("authorization").as_deref(),
            Some("Bearer sk-zen"),
        );
        assert!(
            mock.captured.header("x-api-key").is_none(),
            "opencode must not send x-api-key (rejected by the Zen /chat/completions endpoint)",
        );
    }

    #[tokio::test]
    async fn chat_completion_401_surfaces_as_authentication_error() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store
            .save("opencode", &json!({ "api_key": "wrong" }))
            .unwrap();
        let mock = start_mock_chat(
            401,
            "application/json",
            r#"{"error":"bad key"}"#.to_string(),
        )
        .await;
        let settings = settings_with_base(&format!("http://{}", mock.addr));
        let result = OpencodeProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("should be AuthenticationError");
        assert!(auth.message.contains("mixer auth login opencode"));
    }

    #[tokio::test]
    async fn usage_returns_none() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let snap = OpencodeProvider.usage(&store, &settings).await.unwrap();
        assert!(
            snap.is_none(),
            "opencode Zen has no client-facing usage endpoint — must return Ok(None)"
        );
    }

    #[tokio::test]
    async fn chat_completion_errors_when_api_key_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let settings = ProviderSettings::default_enabled();
        let result = OpencodeProvider
            .chat_completion(&store, &settings, sample_request())
            .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("missing credentials should error"),
        };
        assert!(err.to_string().contains("not authenticated"));
    }
}
