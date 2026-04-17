//! OpenAI-compatible wire types for the `/v1/chat/completions` endpoint.
//!
//! mixer deliberately only deserializes the fields it needs for routing;
//! everything else passes through via `extra` so that providers can forward
//! fields they understand without mixer knowing about them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequest {
    /// The client-facing model name — this is a *mixer model*, not a provider
    /// model. The router rewrites this field to the backend's provider model
    /// before dispatch.
    pub model: String,

    pub messages: Vec<ChatMessage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    /// Catch-all for fields mixer doesn't parse (e.g. `response_format`,
    /// `seed`, `logit_bias`). Forwarded verbatim to the provider.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI allows message content to be either a plain string or an array of
/// content parts (text + image_url for multimodal input).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    /// Anything else (e.g. `input_audio`). Treated as non-image for routing.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Returns true if any message in the request carries an image part.
pub fn request_has_images(req: &ChatRequest) -> bool {
    req.messages.iter().any(|m| match &m.content {
        MessageContent::Text(_) => false,
        MessageContent::Parts(parts) => parts
            .iter()
            .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
    })
}

/// Minimal OpenAI-style chat completion response. Providers are free to
/// return a richer body by shoving extra fields into `extra`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponse {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,

    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// OpenAI-compatible `/v1/models` response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelListEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelListEntry {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_has_no_images() {
        let req: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        assert!(!request_has_images(&req));
    }

    #[test]
    fn image_part_is_detected() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":"describe"},
                        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
                    ]
                }]
            }"#,
        )
        .unwrap();
        assert!(request_has_images(&req));
    }

    #[test]
    fn parts_without_image_are_not_detected() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{
                    "role":"user",
                    "content":[{"type":"text","text":"hi"}]
                }]
            }"#,
        )
        .unwrap();
        assert!(!request_has_images(&req));
    }
}
