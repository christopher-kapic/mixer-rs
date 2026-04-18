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

/// OpenAI-compatible streaming chunk (`chat.completion.chunk`). Providers
/// unconditionally yield these; the server accumulates them for non-streaming
/// clients via [`ChatResponse::from_chunks`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,

    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    #[serde(default)]
    pub delta: ChatDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,

    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatResponse {
    /// Aggregate a sequence of streaming chunks into a single non-streaming
    /// [`ChatResponse`]. Concatenates per-choice text deltas, keeps the last
    /// non-null `finish_reason`, and sums numeric usage fields when present.
    pub fn from_chunks(chunks: Vec<ChatCompletionChunk>) -> Self {
        let mut id = String::new();
        let mut created = 0_i64;
        let mut model = String::new();
        let mut extra: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut usage: Option<Value> = None;

        // choices are assembled keyed by `index`; we preserve first-seen order.
        let mut order: Vec<u32> = Vec::new();
        let mut roles: std::collections::HashMap<u32, Option<String>> =
            std::collections::HashMap::new();
        let mut contents: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        let mut finishes: std::collections::HashMap<u32, Option<String>> =
            std::collections::HashMap::new();
        let mut tool_calls: std::collections::HashMap<u32, Option<Value>> =
            std::collections::HashMap::new();

        for chunk in chunks {
            if id.is_empty() {
                id = chunk.id;
            }
            if created == 0 {
                created = chunk.created;
            }
            if model.is_empty() {
                model = chunk.model;
            }
            for (k, v) in chunk.extra {
                extra.entry(k).or_insert(v);
            }
            if let Some(u) = chunk.usage {
                usage = Some(merge_usage(usage.take(), u));
            }

            for choice in chunk.choices {
                if !order.contains(&choice.index) {
                    order.push(choice.index);
                }
                if let Some(role) = choice.delta.role {
                    roles.entry(choice.index).or_insert(Some(role));
                }
                if let Some(text) = choice.delta.content {
                    contents.entry(choice.index).or_default().push_str(&text);
                }
                if let Some(tc) = choice.delta.tool_calls {
                    tool_calls.insert(choice.index, Some(tc));
                }
                if choice.finish_reason.is_some() {
                    finishes.insert(choice.index, choice.finish_reason);
                }
            }
        }

        let choices: Vec<ChatChoice> = order
            .into_iter()
            .map(|idx| ChatChoice {
                index: idx,
                message: ChatMessage {
                    role: roles
                        .remove(&idx)
                        .flatten()
                        .unwrap_or_else(|| "assistant".to_string()),
                    content: MessageContent::Text(contents.remove(&idx).unwrap_or_default()),
                    name: None,
                    tool_calls: tool_calls.remove(&idx).flatten(),
                    tool_call_id: None,
                },
                finish_reason: finishes.remove(&idx).flatten(),
            })
            .collect();

        ChatResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices,
            usage,
            extra,
        }
    }
}

/// Sum numeric fields of two `usage` objects; non-numeric fields fall back to
/// the value from `incoming`. Used by [`ChatResponse::from_chunks`].
fn merge_usage(existing: Option<Value>, incoming: Value) -> Value {
    let Some(existing) = existing else {
        return incoming;
    };
    let (Value::Object(mut a), Value::Object(b)) = (existing, incoming) else {
        // Couldn't merge structurally — keep the newer view.
        return Value::Null;
    };
    for (k, v) in b {
        match (a.get(&k), &v) {
            (Some(Value::Number(an)), Value::Number(bn)) => {
                let sum = an.as_f64().unwrap_or(0.0) + bn.as_f64().unwrap_or(0.0);
                if let Some(n) = serde_json::Number::from_f64(sum) {
                    a.insert(k, Value::Number(n));
                } else {
                    a.insert(k, v);
                }
            }
            _ => {
                a.insert(k, v);
            }
        }
    }
    Value::Object(a)
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

    fn chunk(content: &str, finish: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "id-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 42,
            model: "m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                    tool_calls: None,
                    extra: Default::default(),
                },
                finish_reason: finish.map(|s| s.to_string()),
            }],
            usage: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn from_chunks_concatenates_content() {
        let first = ChatCompletionChunk {
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                    extra: Default::default(),
                },
                finish_reason: None,
            }],
            ..chunk("", None)
        };
        let chunks = vec![first, chunk("Hello, ", None), chunk("world!", Some("stop"))];
        let resp = ChatResponse::from_chunks(chunks);

        assert_eq!(resp.id, "id-1");
        assert_eq!(resp.model, "m");
        assert_eq!(resp.created, 42);
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        let choice = &resp.choices[0];
        assert_eq!(choice.message.role, "assistant");
        match &choice.message.content {
            MessageContent::Text(s) => assert_eq!(s, "Hello, world!"),
            _ => panic!("expected text content"),
        }
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn from_chunks_sums_usage() {
        let mut a = chunk("a", None);
        a.usage = Some(serde_json::json!({"prompt_tokens": 5, "completion_tokens": 1}));
        let mut b = chunk("b", Some("stop"));
        b.usage = Some(serde_json::json!({"prompt_tokens": 0, "completion_tokens": 2}));
        let resp = ChatResponse::from_chunks(vec![a, b]);
        let usage = resp.usage.expect("usage should be present");
        assert_eq!(usage["prompt_tokens"].as_f64(), Some(5.0));
        assert_eq!(usage["completion_tokens"].as_f64(), Some(3.0));
    }

    #[test]
    fn from_chunks_keeps_last_finish_reason() {
        let chunks = vec![
            chunk("a", Some("length")),
            chunk("b", None),
            chunk("c", Some("stop")),
        ];
        let resp = ChatResponse::from_chunks(chunks);
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }
}
