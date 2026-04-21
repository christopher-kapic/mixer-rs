//! Bidirectional translator between OpenAI **Chat Completions** (mixer's
//! internal wire shape) and the OpenAI **Responses API** (the upstream
//! currently used by `codex`, and the direction the industry seems to be
//! moving).
//!
//! Two pure entry points:
//!
//! - [`chat_request_to_responses_body`] — translate an inbound
//!   [`ChatRequest`] into a Responses API request body (`Value`).
//! - [`responses_sse_to_chat_chunks`] — translate a stream of Responses API
//!   SSE events ([`eventsource_stream::Event`]) into a stream of
//!   [`ChatCompletionChunk`]s.
//!
//! Both are deliberately free of codex-specific values (endpoint URLs, auth
//! headers, `chatgpt-account-id`) — those live in `providers/codex.rs`. This
//! keeps the module reusable for any future "OpenAI Responses API"-shaped
//! upstream.

// First consumer (the real codex provider body) lands in a later step of the
// plan. Until then only the unit tests exercise this module.
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use async_stream::try_stream;
use eventsource_stream::Event;
use futures::{Stream, StreamExt};
use serde_json::{Map, Value, json};

use crate::openai::{
    ChatCompletionChunk, ChatDelta, ChatMessage, ChatRequest, ChunkChoice, ContentPart, ImageUrl,
    MessageContent, TypedContentPart,
};

// ── Request: Chat Completions → Responses API ─────────────────────────────

/// Translate a Chat Completions request body into a Responses API request
/// body.
///
/// `model` is the provider-native model id the caller will dispatch against —
/// supplied separately because the Responses API uses `model` at the top
/// level (unlike Chat Completions where it lives on `req.model`; the router
/// has already rewritten that field by the time a provider is called, but
/// accepting an explicit argument keeps this helper self-contained and
/// testable).
///
/// The returned body always has `stream: true` and `store: false`:
///
/// - **`stream`**: the Responses API has no non-streaming variant against the
///   ChatGPT backend — even non-streaming Chat Completions clients must be
///   served by consuming the stream and accumulating. Forcing `true` here
///   guarantees the caller gets SSE.
/// - **`store`**: mixer is stateless and does not persist server-side threads.
pub fn chat_request_to_responses_body(req: &ChatRequest, model: &str) -> Value {
    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();

    for message in &req.messages {
        match message.role.as_str() {
            "system" | "developer" => {
                if let Some(text) = message_text_only(&message.content)
                    && !text.is_empty()
                {
                    instructions.push(text);
                }
            }
            "user" => input.push(message_to_input(&message.content, "user")),
            "assistant" => append_assistant(&mut input, message),
            "tool" => input.push(tool_message_to_input(message)),
            // Unknown role — forward as a user message so translation never
            // silently drops content.
            _ => input.push(message_to_input(&message.content, "user")),
        }
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("instructions".into(), json!(instructions.join("\n\n")));
    body.insert("input".into(), Value::Array(input));
    body.insert("stream".into(), json!(true));
    body.insert("store".into(), json!(false));

    if let Some(tools) = req.tools.as_ref() {
        body.insert("tools".into(), translate_tools(tools));
    }
    if let Some(tool_choice) = req.tool_choice.as_ref() {
        body.insert("tool_choice".into(), translate_tool_choice(tool_choice));
    }
    if let Some(max_tokens) = req.resolved_max_tokens() {
        body.insert("max_output_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = req.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = req.top_p {
        body.insert("top_p".into(), json!(top_p));
    }

    // Pass-through fields the Responses API recognizes verbatim.
    for key in [
        "parallel_tool_calls",
        "prompt_cache_key",
        "reasoning",
        "service_tier",
        "client_metadata",
    ] {
        if let Some(v) = req.extra.get(key) {
            body.insert(key.into(), v.clone());
        }
    }

    // `reasoning_effort` (Chat Completions shorthand) folds into
    // `reasoning.effort` when the caller didn't provide a full `reasoning`
    // object already.
    if !body.contains_key("reasoning")
        && let Some(effort) = req.extra.get("reasoning_effort").and_then(Value::as_str)
    {
        body.insert("reasoning".into(), json!({ "effort": effort }));
    }

    Value::Object(body)
}

fn message_text_only(content: &Option<MessageContent>) -> Option<String> {
    match content.as_ref()? {
        MessageContent::Text(s) => Some(s.clone()),
        MessageContent::Parts(parts) => {
            let joined: String = parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Typed(TypedContentPart::Text { text }) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            (!joined.is_empty()).then_some(joined)
        }
    }
}

fn message_to_input(content: &Option<MessageContent>, role: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": content_to_input_parts(content),
    })
}

fn content_to_input_parts(content: &Option<MessageContent>) -> Vec<Value> {
    match content {
        None => Vec::new(),
        Some(MessageContent::Text(s)) => vec![json!({"type": "input_text", "text": s})],
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Typed(TypedContentPart::Text { text }) => {
                    Some(json!({"type": "input_text", "text": text}))
                }
                ContentPart::Typed(TypedContentPart::ImageUrl {
                    image_url: ImageUrl { url, .. },
                }) => Some(json!({"type": "input_image", "image_url": url})),
                ContentPart::Unknown(_) => None,
            })
            .collect(),
    }
}

fn append_assistant(input: &mut Vec<Value>, message: &ChatMessage) {
    // Assistant messages can carry plain text, tool_calls, or both. Emit an
    // `output_text` message item for any text, and one `function_call` item
    // per tool call — matching how the Responses API itself represents
    // assistant output in a transcript.
    let text = match &message.content {
        None => None,
        Some(MessageContent::Text(s)) => (!s.is_empty()).then(|| s.clone()),
        Some(MessageContent::Parts(parts)) => {
            let joined: String = parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Typed(TypedContentPart::Text { text }) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            (!joined.is_empty()).then_some(joined)
        }
    };
    if let Some(text) = text {
        input.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }));
    }
    if let Some(tool_calls) = message.tool_calls.as_ref().and_then(Value::as_array) {
        for call in tool_calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let function = call.get("function").cloned().unwrap_or(Value::Null);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            input.push(json!({
                "type": "function_call",
                "name": name,
                "arguments": arguments,
                "call_id": id,
            }));
        }
    }
}

fn tool_message_to_input(message: &ChatMessage) -> Value {
    let output = match &message.content {
        None => String::new(),
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Typed(TypedContentPart::Text { text }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    };
    json!({
        "type": "function_call_output",
        "call_id": message.tool_call_id.clone().unwrap_or_default(),
        "output": output,
    })
}

fn translate_tools(tools: &Value) -> Value {
    let Some(arr) = tools.as_array() else {
        return tools.clone();
    };
    Value::Array(arr.iter().map(translate_single_tool).collect())
}

fn translate_single_tool(tool: &Value) -> Value {
    let Value::Object(obj) = tool else {
        return tool.clone();
    };
    if obj.get("type").and_then(Value::as_str) != Some("function") {
        return tool.clone();
    }
    // Already flat (`{"type":"function","name":"…", …}`) — pass through.
    let Some(function) = obj.get("function").and_then(Value::as_object) else {
        return tool.clone();
    };
    let mut out = Map::new();
    out.insert("type".into(), json!("function"));
    for key in ["name", "description", "parameters", "strict"] {
        if let Some(v) = function.get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    Value::Object(out)
}

fn translate_tool_choice(tc: &Value) -> Value {
    // "auto" / "required" / "none" pass through as-is.
    if tc.is_string() {
        return tc.clone();
    }
    let Some(obj) = tc.as_object() else {
        return tc.clone();
    };
    if obj.get("type").and_then(Value::as_str) != Some("function") {
        return tc.clone();
    }
    if let Some(fun) = obj.get("function").and_then(Value::as_object)
        && let Some(name) = fun.get("name")
    {
        return json!({"type": "function", "name": name});
    }
    tc.clone()
}

// ── Response: Responses API SSE → Chat Completions chunks ─────────────────

/// Translate a stream of Responses API SSE events into Chat Completions
/// [`ChatCompletionChunk`]s.
///
/// Event mapping:
///
/// | Responses API event                | Emitted chunk                           |
/// |------------------------------------|-----------------------------------------|
/// | `response.created`                 | opener with `delta.role = "assistant"`  |
/// | `response.output_text.delta`       | chunk with text `delta.content`         |
/// | `response.output_item.done` (function_call) | chunk with a single `tool_calls` delta (`id ← call_id`, `function.name`, `function.arguments`) |
/// | `response.completed`               | final chunk with `finish_reason` + usage|
/// | `response.failed` / `response.incomplete` | stream ends with `Err`           |
///
/// Tool-call IDs are preserved: Responses API `function_call.call_id` becomes
/// the Chat Completions `tool_calls[*].id`, so downstream tool-result round
/// trips through the translator correctly (plan.md §5.2.2).
pub fn responses_sse_to_chat_chunks<S>(events: S) -> impl Stream<Item = Result<ChatCompletionChunk>>
where
    S: Stream<Item = Result<Event>>,
{
    try_stream! {
        futures::pin_mut!(events);

        let mut response_id = String::new();
        let mut model_name = String::new();
        let mut created: i64 = 0;
        let mut role_emitted = false;
        let mut tool_call_index: u32 = 0;
        let mut saw_tool_call = false;

        while let Some(event) = events.next().await {
            let event = event?;
            if event.data.is_empty() {
                continue;
            }
            let payload: Value = serde_json::from_str(&event.data)
                .map_err(|e| anyhow!("failed to parse Responses API SSE data `{}`: {e}", event.data))?;
            let kind = payload.get("type").and_then(Value::as_str).unwrap_or("");

            match kind {
                "response.created" => {
                    if let Some(resp) = payload.get("response") {
                        if response_id.is_empty()
                            && let Some(id) = resp.get("id").and_then(Value::as_str) {
                            response_id = id.to_string();
                        }
                        if model_name.is_empty()
                            && let Some(model) = resp.get("model").and_then(Value::as_str) {
                            model_name = model.to_string();
                        }
                        if created == 0
                            && let Some(t) = resp.get("created_at").and_then(Value::as_i64) {
                            created = t;
                        }
                    }
                    if !role_emitted {
                        role_emitted = true;
                        yield opener_chunk(&response_id, &model_name, created);
                    }
                }
                "response.output_text.delta" => {
                    let delta = payload.get("delta").and_then(Value::as_str).unwrap_or("");
                    if delta.is_empty() {
                        continue;
                    }
                    if !role_emitted {
                        role_emitted = true;
                        yield opener_chunk(&response_id, &model_name, created);
                    }
                    yield text_chunk(&response_id, &model_name, created, delta);
                }
                "response.output_item.done" => {
                    let Some(item) = payload.get("item") else { continue };
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = item.get("arguments").and_then(Value::as_str).unwrap_or("");
                        let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                        if !role_emitted {
                            role_emitted = true;
                            yield opener_chunk(&response_id, &model_name, created);
                        }
                        let idx = tool_call_index;
                        tool_call_index += 1;
                        saw_tool_call = true;
                        yield tool_call_chunk(
                            &response_id, &model_name, created, idx, call_id, name, args,
                        );
                    }
                }
                "response.failed" | "response.incomplete" => {
                    let message = payload
                        .get("response")
                        .and_then(|r| r.get("error"))
                        .and_then(|e| e.get("message"))
                        .and_then(Value::as_str)
                        .unwrap_or(if kind == "response.failed" {
                            "response.failed"
                        } else {
                            "response.incomplete"
                        });
                    Err::<(), _>(anyhow!("{message}"))?;
                }
                "response.completed" => {
                    let usage = payload
                        .get("response")
                        .and_then(|r| r.get("usage"))
                        .cloned();
                    yield final_chunk(
                        &response_id, &model_name, created, saw_tool_call, usage,
                    );
                }
                _ => {
                    // Unhandled events (reasoning summaries, rate-limit
                    // metadata, etc.) are ignored — Chat Completions has no
                    // 1:1 surface for them and dropping them is safer than
                    // guessing.
                }
            }
        }
    }
}

fn opener_chunk(id: &str, model: &str, created: i64) -> ChatCompletionChunk {
    base_chunk(
        id,
        model,
        created,
        ChatDelta {
            role: Some("assistant".to_string()),
            ..Default::default()
        },
        None,
        None,
    )
}

fn text_chunk(id: &str, model: &str, created: i64, delta: &str) -> ChatCompletionChunk {
    base_chunk(
        id,
        model,
        created,
        ChatDelta {
            content: Some(delta.to_string()),
            ..Default::default()
        },
        None,
        None,
    )
}

fn tool_call_chunk(
    id: &str,
    model: &str,
    created: i64,
    index: u32,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ChatCompletionChunk {
    let tool_calls = json!([{
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }]);
    base_chunk(
        id,
        model,
        created,
        ChatDelta {
            tool_calls: Some(tool_calls),
            ..Default::default()
        },
        None,
        None,
    )
}

fn final_chunk(
    id: &str,
    model: &str,
    created: i64,
    tool_call: bool,
    usage: Option<Value>,
) -> ChatCompletionChunk {
    let reason = if tool_call { "tool_calls" } else { "stop" };
    base_chunk(
        id,
        model,
        created,
        ChatDelta::default(),
        Some(reason.to_string()),
        usage,
    )
}

fn base_chunk(
    id: &str,
    model: &str,
    created: i64,
    delta: ChatDelta,
    finish_reason: Option<String>,
    usage: Option<Value>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: chunk_id(id),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
        extra: Default::default(),
    }
}

/// Synthesize a stable id when the upstream didn't provide one (rare, but
/// guards against chunks with empty ids surfacing to the client).
fn chunk_id(upstream_id: &str) -> String {
    if !upstream_id.is_empty() {
        return upstream_id.to_string();
    }
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("chatcmpl-mixer-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatRequest;
    use futures::stream;

    fn parse_req(json_str: &str) -> ChatRequest {
        serde_json::from_str(json_str).expect("valid ChatRequest fixture")
    }

    // ── Request translation tests ───────────────────────────────────────

    #[test]
    fn request_text_only_message() {
        let req = parse_req(
            r#"{
                "model": "mixer",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"}
                ]
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["model"], "gpt-5.2");
        assert_eq!(body["instructions"], "be terse");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hi");
    }

    #[test]
    fn request_joins_system_and_developer_instructions() {
        let req = parse_req(
            r#"{
                "model": "mixer",
                "messages": [
                    {"role": "system", "content": "rule 1"},
                    {"role": "developer", "content": "rule 2"},
                    {"role": "user", "content": "go"}
                ]
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["instructions"], "rule 1\n\nrule 2");
    }

    #[test]
    fn request_translates_image_parts() {
        let req = parse_req(
            r#"{
                "model": "mixer",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                }]
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        let parts = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["image_url"], "data:image/png;base64,abc");
    }

    #[test]
    fn request_hoists_tool_function_fields() {
        let req = parse_req(
            r#"{
                "model": "mixer",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "look it up",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }]
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let t = &tools[0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["name"], "get_weather");
        assert_eq!(t["description"], "look it up");
        assert_eq!(t["parameters"]["type"], "object");
        assert!(t.get("function").is_none(), "wrapper should be gone");
    }

    #[test]
    fn request_translates_tool_choice_variants() {
        for (input, expected) in [
            (r#"{"tool_choice": "auto"}"#, json!("auto")),
            (r#"{"tool_choice": "required"}"#, json!("required")),
            (r#"{"tool_choice": "none"}"#, json!("none")),
        ] {
            let wrapped = format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"x"}}],{}}}"#,
                input.trim_start_matches('{').trim_end_matches('}')
            );
            let req = parse_req(&wrapped);
            let body = chat_request_to_responses_body(&req, "gpt-5.2");
            assert_eq!(body["tool_choice"], expected);
        }

        let req = parse_req(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"x"}],
                "tool_choice": {"type":"function","function":{"name":"foo"}}
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["name"], "foo");
        assert!(body["tool_choice"].get("function").is_none());
    }

    #[test]
    fn request_maps_max_tokens_to_max_output_tokens() {
        let req = parse_req(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"x"}],
                "max_tokens":512
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["max_output_tokens"], 512);
    }

    #[test]
    fn request_maps_max_completion_tokens_to_max_output_tokens() {
        let req = parse_req(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"x"}],
                "max_completion_tokens":1024
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(
            body["max_output_tokens"], 1024,
            "Responses API must honor the modern max_completion_tokens alias",
        );
    }

    #[test]
    fn request_prefers_max_completion_tokens_when_both_are_set() {
        let req = parse_req(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"x"}],
                "max_tokens":512,
                "max_completion_tokens":1024
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["max_output_tokens"], 1024);
    }

    #[test]
    fn request_surfaces_reasoning_effort_shorthand() {
        let req = parse_req(
            r#"{
                "model":"m",
                "messages":[{"role":"user","content":"x"}],
                "reasoning_effort":"high"
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn request_round_trips_assistant_tool_calls_and_outputs() {
        let req = parse_req(
            r#"{
                "model": "mixer",
                "messages": [
                    {"role":"user","content":"what's the weather?"},
                    {
                        "role":"assistant",
                        "content":"",
                        "tool_calls":[{
                            "id":"call_42",
                            "type":"function",
                            "function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}
                        }]
                    },
                    {"role":"tool","tool_call_id":"call_42","content":"62F sunny"}
                ]
            }"#,
        );
        let body = chat_request_to_responses_body(&req, "gpt-5.2");
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        // Assistant message with empty content emits no text message item —
        // just the function_call.
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_42");
        assert_eq!(input[1]["name"], "get_weather");
        assert_eq!(input[1]["arguments"], "{\"city\":\"SF\"}");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_42");
        assert_eq!(input[2]["output"], "62F sunny");
    }

    // ── SSE translation tests ───────────────────────────────────────────

    fn sse_event(data: impl Into<String>) -> Event {
        Event {
            event: "message".to_string(),
            data: data.into(),
            id: String::new(),
            retry: None,
        }
    }

    fn sse_events(payloads: &[&str]) -> Vec<Result<Event>> {
        payloads.iter().map(|p| Ok(sse_event(*p))).collect()
    }

    async fn collect_chunks(payloads: &[&str]) -> Result<Vec<ChatCompletionChunk>> {
        let events = stream::iter(sse_events(payloads));
        let chunks = responses_sse_to_chat_chunks(events);
        futures::pin_mut!(chunks);
        let mut out = Vec::new();
        while let Some(c) = chunks.next().await {
            out.push(c?);
        }
        Ok(out)
    }

    const PLAIN_TEXT_SSE: &[&str] = &[
        r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-5.2","created_at":42}}"#,
        r#"{"type":"response.output_text.delta","delta":"Hello, "}"#,
        r#"{"type":"response.output_text.delta","delta":"world!"}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":5,"output_tokens":3}}}"#,
    ];

    const TOOL_CALL_ONLY_SSE: &[&str] = &[
        r#"{"type":"response.created","response":{"id":"resp_2","model":"gpt-5.2","created_at":7}}"#,
        r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_a","name":"get_weather","arguments":"{\"city\":\"SF\"}"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_2"}}"#,
    ];

    const MIXED_SSE: &[&str] = &[
        r#"{"type":"response.created","response":{"id":"resp_3","model":"gpt-5.2","created_at":9}}"#,
        r#"{"type":"response.output_text.delta","delta":"sure, "}"#,
        r#"{"type":"response.output_text.delta","delta":"let me check."}"#,
        r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_b","name":"lookup","arguments":"{}"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_3"}}"#,
    ];

    const FAILED_SSE: &[&str] = &[
        r#"{"type":"response.created","response":{"id":"resp_4","model":"gpt-5.2"}}"#,
        r#"{"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded","message":"slow down"}}}"#,
    ];

    #[tokio::test]
    async fn sse_plain_text_is_translated() {
        let chunks = collect_chunks(PLAIN_TEXT_SSE).await.expect("ok");
        // opener + 2 deltas + final = 4
        assert_eq!(chunks.len(), 4);
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        assert_eq!(chunks[0].id, "resp_1");
        assert_eq!(chunks[0].model, "gpt-5.2");
        assert_eq!(chunks[0].created, 42);

        assert_eq!(
            chunks[1].choices[0].delta.content.as_deref(),
            Some("Hello, ")
        );
        assert_eq!(
            chunks[2].choices[0].delta.content.as_deref(),
            Some("world!")
        );

        assert_eq!(chunks[3].choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = chunks[3].usage.as_ref().expect("final chunk carries usage");
        assert_eq!(usage["input_tokens"], 5);
        assert_eq!(usage["output_tokens"], 3);
    }

    #[tokio::test]
    async fn sse_tool_call_only_preserves_ids_and_finish_reason() {
        let chunks = collect_chunks(TOOL_CALL_ONLY_SSE).await.expect("ok");
        // opener + tool_call + final = 3
        assert_eq!(chunks.len(), 3);
        let tc = chunks[1].choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls on the tool-call chunk");
        let call = &tc[0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_a");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"SF\"}");

        assert_eq!(
            chunks[2].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn sse_mixed_emits_text_then_tool_call_then_tool_calls_finish() {
        let chunks = collect_chunks(MIXED_SSE).await.expect("ok");
        // opener + 2 text + 1 tool_call + final = 5
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        assert_eq!(
            chunks[1].choices[0].delta.content.as_deref(),
            Some("sure, ")
        );
        assert_eq!(
            chunks[2].choices[0].delta.content.as_deref(),
            Some("let me check.")
        );
        assert!(chunks[3].choices[0].delta.tool_calls.is_some());
        assert_eq!(
            chunks[4].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn sse_failed_event_surfaces_as_stream_error() {
        let events = stream::iter(sse_events(FAILED_SSE));
        let chunks = responses_sse_to_chat_chunks(events);
        futures::pin_mut!(chunks);

        // First the opener should come through.
        let first = chunks.next().await.expect("opener").expect("ok");
        assert_eq!(first.choices[0].delta.role.as_deref(), Some("assistant"));

        // Then the failed event terminates with an Err.
        let err = chunks
            .next()
            .await
            .expect("failure frame")
            .expect_err("should surface as error");
        assert!(err.to_string().contains("slow down"), "got: {err:#}");

        // Stream ends after the error.
        assert!(chunks.next().await.is_none());
    }

    #[tokio::test]
    async fn sse_tool_call_ids_increment_across_multiple_calls() {
        let payloads = [
            r#"{"type":"response.created","response":{"id":"resp_x","model":"gpt-5.2"}}"#,
            r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_a","name":"f","arguments":"{}"}}"#,
            r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_b","name":"g","arguments":"{}"}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_x"}}"#,
        ];
        let chunks = collect_chunks(&payloads).await.expect("ok");
        let first_call = chunks[1].choices[0].delta.tool_calls.as_ref().unwrap();
        let second_call = chunks[2].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(first_call[0]["index"], 0);
        assert_eq!(first_call[0]["id"], "call_a");
        assert_eq!(second_call[0]["index"], 1);
        assert_eq!(second_call[0]["id"], "call_b");
    }
}
