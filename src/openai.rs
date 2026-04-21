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

    /// OpenAI-Chat-Completions renamed `max_tokens` to `max_completion_tokens`;
    /// modern SDKs send the latter. Mixer accepts both, prefers
    /// `max_completion_tokens` when present (see
    /// [`ChatRequest::resolved_max_tokens`]), and forwards whichever field(s)
    /// the client sent so the upstream sees the exact shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

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
    /// Null/absent for assistant messages that carry only `tool_calls` — the
    /// OpenAI spec allows this shape and agent clients routinely send it.
    /// Always serialised (as `null` when absent) so upstream providers that
    /// key off the field's presence see what the client actually sent.
    #[serde(default)]
    pub content: Option<MessageContent>,

    /// Canonical chain-of-thought channel, populated by the provider or the
    /// reasoning-normalization pipeline. Established by DeepSeek, adopted by
    /// Kimi/GLM/MiniMax; mixer normalizes every other upstream dialect into
    /// this field before emitting to the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

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

/// A single content part inside a multimodal message.
///
/// Known shapes (`text`, `image_url`) deserialize into typed variants so
/// routing and the Responses API translator can reason about them. Any other
/// shape — `input_audio`, future provider-specific part types, malformed
/// objects — is preserved verbatim as a raw JSON value so it round-trips to
/// the upstream provider exactly as the client sent it.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ContentPart {
    Typed(TypedContentPart),
    Unknown(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ChatRequest {
    /// Resolve the effective output-token budget, preferring the newer
    /// `max_completion_tokens` field when both are present. Used by the
    /// context-window filter and the Responses API translator so a modern
    /// client that only sends `max_completion_tokens` is honored correctly.
    pub fn resolved_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

/// Returns true if any message in the request carries an image part.
pub fn request_has_images(req: &ChatRequest) -> bool {
    req.messages.iter().any(|m| match &m.content {
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .any(|p| matches!(p, ContentPart::Typed(TypedContentPart::ImageUrl { .. }))),
        _ => false,
    })
}

/// Rough token estimate for routing decisions. Uses the standard ~4-chars/token
/// heuristic across serialized text content, tool definitions, and tool_choice.
/// This is deliberately an overestimate in most cases — routing errs toward
/// protecting the request from a too-small context window.
pub fn estimate_input_tokens(req: &ChatRequest) -> u32 {
    let mut chars: usize = 0;
    for m in &req.messages {
        chars = chars.saturating_add(m.role.len()).saturating_add(4);
        match &m.content {
            None => {}
            Some(MessageContent::Text(s)) => chars = chars.saturating_add(s.len()),
            Some(MessageContent::Parts(parts)) => {
                for part in parts {
                    match part {
                        ContentPart::Typed(TypedContentPart::Text { text }) => {
                            chars = chars.saturating_add(text.len())
                        }
                        ContentPart::Typed(TypedContentPart::ImageUrl { .. }) => {
                            chars = chars.saturating_add(85 * 4)
                        }
                        ContentPart::Unknown(v) => {
                            if let Ok(s) = serde_json::to_string(v) {
                                chars = chars.saturating_add(s.len());
                            }
                        }
                    }
                }
            }
        }
        if let Some(name) = &m.name {
            chars = chars.saturating_add(name.len());
        }
        if let Some(tc) = &m.tool_calls
            && let Ok(s) = serde_json::to_string(tc)
        {
            chars = chars.saturating_add(s.len());
        }
        if let Some(id) = &m.tool_call_id {
            chars = chars.saturating_add(id.len());
        }
    }
    if let Some(tools) = &req.tools
        && let Ok(s) = serde_json::to_string(tools)
    {
        chars = chars.saturating_add(s.len());
    }
    if let Some(tc) = &req.tool_choice
        && let Ok(s) = serde_json::to_string(tc)
    {
        chars = chars.saturating_add(s.len());
    }
    let tokens = chars / 4;
    u32::try_from(tokens).unwrap_or(u32::MAX)
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
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
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
    /// Canonical chain-of-thought channel for streaming deltas. Providers
    /// that natively use DeepSeek's `reasoning_content` surface it here via
    /// serde; providers that emit other dialects (Qwen inline `<think>` tags,
    /// OpenAI Responses-API reasoning-summary events) are rewritten into this
    /// field by [`crate::reasoning`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,

    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatResponse {
    /// Aggregate a sequence of streaming chunks into a single non-streaming
    /// [`ChatResponse`]. Concatenates per-choice text deltas and tool-call
    /// argument fragments (keyed by each tool call's `index`), keeps the last
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
        let mut reasonings: std::collections::HashMap<u32, String> =
            std::collections::HashMap::new();
        let mut finishes: std::collections::HashMap<u32, Option<String>> =
            std::collections::HashMap::new();
        let mut tool_calls: std::collections::HashMap<u32, Vec<Value>> =
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
                if let Some(text) = choice.delta.reasoning_content {
                    reasonings.entry(choice.index).or_default().push_str(&text);
                }
                if let Some(tc) = choice.delta.tool_calls {
                    let entry = tool_calls.entry(choice.index).or_default();
                    merge_tool_call_deltas(entry, tc);
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
                    content: Some(MessageContent::Text(
                        contents.remove(&idx).unwrap_or_default(),
                    )),
                    reasoning_content: reasonings.remove(&idx).filter(|s| !s.is_empty()),
                    name: None,
                    tool_calls: tool_calls
                        .remove(&idx)
                        .filter(|v| !v.is_empty())
                        .map(Value::Array),
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
/// the value from `incoming`. If either side isn't an object, prefer the
/// `incoming` view — it's the newer one and the one the upstream just sent.
/// Used by [`ChatResponse::from_chunks`].
fn merge_usage(existing: Option<Value>, incoming: Value) -> Value {
    let Some(existing) = existing else {
        return incoming;
    };
    let Value::Object(mut a) = existing else {
        return incoming;
    };
    let Value::Object(b) = incoming else {
        return Value::Object(a);
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

/// Merge an incoming delta's `tool_calls` array into the accumulated per-choice
/// list. OpenAI streams each tool call in multiple fragments keyed by `index`:
/// the first fragment typically carries `id`, `type`, and `function.name`;
/// subsequent fragments extend `function.arguments` character by character.
/// Without this merge, a non-streaming client would only see the final
/// fragment and receive a truncated arguments JSON.
///
/// Uses an `index → position` map so the merge stays O(n) in the total number
/// of fragments instead of O(n²) — matters for agent loops that dispatch many
/// parallel tool calls in a single response.
fn merge_tool_call_deltas(accumulated: &mut Vec<Value>, incoming: Value) {
    let Value::Array(incoming_arr) = incoming else {
        return;
    };

    fn read_index(v: &Value) -> Option<u64> {
        v.as_object().and_then(|obj| obj.get("index"))?.as_u64()
    }

    let mut index_to_pos: std::collections::HashMap<u64, usize> = accumulated
        .iter()
        .enumerate()
        .filter_map(|(pos, v)| read_index(v).map(|idx| (idx, pos)))
        .collect();

    for item in incoming_arr {
        match read_index(&item).and_then(|idx| index_to_pos.get(&idx).copied()) {
            Some(pos) => merge_tool_call_into(&mut accumulated[pos], item),
            None => {
                if let Some(idx) = read_index(&item) {
                    index_to_pos.insert(idx, accumulated.len());
                }
                accumulated.push(item);
            }
        }
    }
}

fn merge_tool_call_into(existing: &mut Value, incoming: Value) {
    let Value::Object(existing_obj) = existing else {
        *existing = incoming;
        return;
    };
    let Value::Object(incoming_obj) = incoming else {
        return;
    };
    for (k, v) in incoming_obj {
        if k == "function" {
            merge_function_delta(existing_obj, v);
        } else {
            existing_obj.entry(k).or_insert(v);
        }
    }
}

fn merge_function_delta(container: &mut serde_json::Map<String, Value>, incoming: Value) {
    let Value::Object(incoming_fn) = incoming else {
        return;
    };
    let existing_fn = match container.get_mut("function") {
        Some(Value::Object(m)) => m,
        _ => {
            container.insert("function".to_string(), Value::Object(incoming_fn));
            return;
        }
    };
    for (fk, fv) in incoming_fn {
        if fk == "arguments" {
            match (existing_fn.get_mut("arguments"), fv) {
                (Some(Value::String(existing_s)), Value::String(incoming_s)) => {
                    existing_s.push_str(&incoming_s);
                }
                (_, fv) => {
                    existing_fn.insert(fk, fv);
                }
            }
        } else {
            existing_fn.entry(fk).or_insert(fv);
        }
    }
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
    fn resolved_max_tokens_prefers_max_completion_tokens() {
        let only_max_tokens: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":512}"#,
        )
        .unwrap();
        assert_eq!(only_max_tokens.resolved_max_tokens(), Some(512));

        let only_new: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":1024}"#,
        )
        .unwrap();
        assert_eq!(only_new.resolved_max_tokens(), Some(1024));

        // Both present — prefer the newer field.
        let both: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":512,"max_completion_tokens":1024}"#,
        )
        .unwrap();
        assert_eq!(both.resolved_max_tokens(), Some(1024));

        let neither: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        assert_eq!(neither.resolved_max_tokens(), None);
    }

    #[test]
    fn max_completion_tokens_round_trips_without_landing_in_extra() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":256}"#,
        )
        .unwrap();
        assert_eq!(req.max_completion_tokens, Some(256));
        // The pass-through catch-all must not also capture the field we now
        // parse explicitly — otherwise we'd double-serialise it when
        // forwarding to an upstream.
        assert!(!req.extra.contains_key("max_completion_tokens"));
        let back = serde_json::to_value(&req).unwrap();
        assert_eq!(back["max_completion_tokens"], 256);
    }

    #[test]
    fn estimate_input_tokens_empty_request_is_small() {
        let req: ChatRequest = serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
        assert!(
            estimate_input_tokens(&req) < 4,
            "empty request should estimate near zero",
        );
    }

    #[test]
    fn estimate_input_tokens_scales_with_text_length() {
        let short: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        let long_text = "a".repeat(4000);
        let long_json =
            format!(r#"{{"model":"m","messages":[{{"role":"user","content":"{long_text}"}}]}}"#);
        let long: ChatRequest = serde_json::from_str(&long_json).unwrap();

        let short_est = estimate_input_tokens(&short);
        let long_est = estimate_input_tokens(&long);
        assert!(
            long_est > short_est + 900,
            "4000 extra chars should add ~1000 tokens (chars/4): short={short_est}, long={long_est}",
        );
    }

    #[test]
    fn estimate_input_tokens_counts_images() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":""},
                        {"type":"image_url","image_url":{"url":"x"}}
                    ]
                }]
            }"#,
        )
        .unwrap();
        let baseline: ChatRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":""}
                    ]
                }]
            }"#,
        )
        .unwrap();
        let delta = estimate_input_tokens(&req) - estimate_input_tokens(&baseline);
        assert!(
            delta >= 85,
            "image part should contribute at least 85 tokens, got {delta}",
        );
    }

    #[test]
    fn estimate_input_tokens_accounts_for_tools() {
        let without: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        let with_tools: ChatRequest = serde_json::from_str(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[
                    {"type":"function","function":{"name":"get_weather","description":"Fetch current weather for a city by name, returning a structured JSON body","parameters":{"type":"object"}}}
                ]
            }"#,
        )
        .unwrap();
        assert!(
            estimate_input_tokens(&with_tools) > estimate_input_tokens(&without) + 10,
            "tools payload should contribute to the estimate",
        );
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
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content.to_string())
                    },
                    ..Default::default()
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
                    ..Default::default()
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
            Some(MessageContent::Text(s)) => assert_eq!(s, "Hello, world!"),
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

    fn tool_call_chunk(tool_calls: Value, finish: Option<&str>) -> ChatCompletionChunk {
        let mut c = chunk("", finish);
        c.choices[0].delta.tool_calls = Some(tool_calls);
        c
    }

    #[test]
    fn from_chunks_aggregates_tool_call_argument_fragments() {
        let first = tool_call_chunk(
            serde_json::json!([{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\""}
            }]),
            None,
        );
        let second = tool_call_chunk(
            serde_json::json!([{
                "index": 0,
                "function": {"arguments": "city\":"}
            }]),
            None,
        );
        let third = tool_call_chunk(
            serde_json::json!([{
                "index": 0,
                "function": {"arguments": "\"NYC\"}"}
            }]),
            Some("tool_calls"),
        );

        let resp = ChatResponse::from_chunks(vec![first, second, third]);
        let tc = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls should be set");
        let arr = tc.as_array().expect("tool_calls should be an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "call_abc");
        assert_eq!(arr[0]["type"], "function");
        assert_eq!(arr[0]["function"]["name"], "get_weather");
        assert_eq!(arr[0]["function"]["arguments"], "{\"city\":\"NYC\"}");
    }

    #[test]
    fn from_chunks_keeps_parallel_tool_calls_separate() {
        let first = tool_call_chunk(
            serde_json::json!([
                {"index": 0, "id": "call_a", "type": "function", "function": {"name": "f_a", "arguments": "{\"x\":"}},
                {"index": 1, "id": "call_b", "type": "function", "function": {"name": "f_b", "arguments": "{\"y\":"}}
            ]),
            None,
        );
        let second = tool_call_chunk(
            serde_json::json!([
                {"index": 0, "function": {"arguments": "1}"}},
                {"index": 1, "function": {"arguments": "2}"}}
            ]),
            Some("tool_calls"),
        );

        let resp = ChatResponse::from_chunks(vec![first, second]);
        let arr = resp.choices[0]
            .message
            .tool_calls
            .as_ref()
            .and_then(|v| v.as_array())
            .expect("tool_calls array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["id"], "call_a");
        assert_eq!(arr[0]["function"]["arguments"], "{\"x\":1}");
        assert_eq!(arr[1]["id"], "call_b");
        assert_eq!(arr[1]["function"]["arguments"], "{\"y\":2}");
    }

    #[test]
    fn merge_usage_prefers_incoming_when_existing_is_not_object() {
        let existing = Some(Value::Null);
        let incoming = serde_json::json!({"prompt_tokens": 5});
        let merged = merge_usage(existing, incoming);
        assert_eq!(merged["prompt_tokens"].as_f64(), Some(5.0));
    }

    #[test]
    fn merge_usage_preserves_existing_when_incoming_is_not_object() {
        let existing = Some(serde_json::json!({"prompt_tokens": 5}));
        let incoming = Value::Null;
        let merged = merge_usage(existing, incoming);
        assert_eq!(merged["prompt_tokens"].as_f64(), Some(5.0));
    }

    #[test]
    fn assistant_tool_call_message_with_null_content_deserializes() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                        ]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
                ]
            }"#,
        )
        .expect("tool-call transcript with null content must parse");
        assert_eq!(req.messages.len(), 3);
        assert!(req.messages[1].content.is_none());
        assert!(req.messages[1].tool_calls.is_some());
        // Round-trips — serialisation emits `content: null` so upstream
        // providers see the exact shape the client sent.
        let back = serde_json::to_value(&req).unwrap();
        assert!(back["messages"][1]["content"].is_null());
    }

    #[test]
    fn assistant_message_with_missing_content_field_deserializes() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                    }
                ]
            }"#,
        )
        .expect("assistant message with omitted content must parse");
        assert!(req.messages[0].content.is_none());
    }

    #[test]
    fn unknown_content_part_round_trips_verbatim() {
        let raw = r#"{
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
                ]
            }]
        }"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        let parts = match &req.messages[0].content {
            Some(MessageContent::Parts(p)) => p,
            _ => panic!("expected parts"),
        };
        assert_eq!(parts.len(), 2);
        match &parts[1] {
            ContentPart::Unknown(v) => {
                assert_eq!(v["type"], "input_audio");
                assert_eq!(v["input_audio"]["data"], "AAAA");
                assert_eq!(v["input_audio"]["format"], "wav");
            }
            _ => panic!("expected unknown variant for input_audio"),
        }
        // Serialisation preserves the original shape instead of collapsing it.
        let back = serde_json::to_value(&req).unwrap();
        let out_parts = back["messages"][0]["content"].as_array().unwrap();
        assert_eq!(out_parts[1]["type"], "input_audio");
        assert_eq!(out_parts[1]["input_audio"]["data"], "AAAA");
        assert_eq!(out_parts[1]["input_audio"]["format"], "wav");
    }
}
