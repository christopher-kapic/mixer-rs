//! OpenAI-compatible Chat Completions client shared by API-key providers
//! (minimax, glm, opencode).
//!
//! Entry point: [`chat_completion`]. The caller supplies the endpoint URL, the
//! API key, and an [`AuthScheme`] (each provider uses a slightly different
//! authentication header convention — `Bearer` for minimax and glm, a custom
//! `x-api-key` header for opencode Zen). The returned [`ChatStream`] yields
//! [`ChatCompletionChunk`]s regardless of whether the upstream responded with
//! SSE or a plain JSON body, so callers do not need to branch on content type.
//!
//! Stream handling:
//!
//! - `stream: true` is set unconditionally on the upstream request, so the
//!   returned stream is "real" for well-behaved upstreams.
//! - If the upstream honors SSE (`Content-Type: text/event-stream`), each
//!   `data: {...}` line is deserialized as a [`ChatCompletionChunk`]. The
//!   sentinel `data: [DONE]` is swallowed.
//! - If the upstream ignores `stream: true` and returns `application/json`
//!   (some self-hosted OpenAI-compatible endpoints do this), the body is
//!   parsed as a [`ChatResponse`] and wrapped in a single-chunk stream.
//!
//! Errors are surfaced with the HTTP status and a truncated body snippet so
//! clients see actionable messages rather than a generic "request failed".

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use reqwest::{Client, RequestBuilder, StatusCode, header::CONTENT_TYPE};

use crate::openai::{
    ChatCompletionChunk, ChatDelta, ChatRequest, ChatResponse, ChunkChoice, ContentPart,
    MessageContent,
};
use crate::providers::ChatStream;
use crate::providers::common::oauth_refresh::AuthenticationError;

/// How the API key is presented to the upstream.
#[derive(Debug, Clone, Copy)]
pub enum AuthScheme {
    /// `Authorization: Bearer <api_key>`. Used by minimax and glm.
    Bearer,
    /// `<header_name>: <api_key>`. Used by opencode (`x-api-key`).
    ApiKeyHeader(&'static str),
}

/// POST an OpenAI-compatible Chat Completions request and return a stream of
/// [`ChatCompletionChunk`]s. See the module docs for content-type fallback
/// behavior.
///
/// `provider_id` and `provider_display` are used solely to build an
/// actionable [`AuthenticationError`] when the upstream returns 401; they are
/// not appended to the URL or sent over the wire.
pub async fn chat_completion(
    provider_id: &str,
    url: &str,
    api_key: &str,
    auth_scheme: AuthScheme,
    timeout: Option<Duration>,
    mut req: ChatRequest,
) -> Result<ChatStream> {
    // Always ask the upstream to stream so the returned stream is real; we
    // still cope with non-SSE JSON responses below.
    req.stream = Some(true);

    let client = build_http_client(timeout)?;

    let request = apply_auth(
        client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&req),
        api_key,
        auth_scheme,
    );

    let resp = request
        .send()
        .await
        .with_context(|| format!("posting to {url}"))?;

    let status = resp.status();
    if status == StatusCode::UNAUTHORIZED {
        // An API-key upstream rejecting the key is an auth problem the user
        // can fix — surface it as AuthenticationError so the server layer
        // renders it as a real 401 with an actionable message.
        return Err(anyhow::Error::new(AuthenticationError {
            message: format!(
                "{provider_id} api key rejected — run `mixer auth login {provider_id}` \
                 or update the stored key"
            ),
        }));
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(1024).collect();
        bail!("upstream returned {status}: {snippet}");
    }

    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.contains("text/event-stream") {
        let events = resp
            .bytes_stream()
            .eventsource()
            .map(|r| r.map_err(|e| anyhow!("SSE stream error: {e}")));
        Ok(Box::pin(sse_events_to_chunks(events)))
    } else {
        let body = resp
            .text()
            .await
            .context("reading upstream response body")?;
        let chunk = json_response_to_chunk(&body)?;
        Ok(Box::pin(futures::stream::once(async move { Ok(chunk) })))
    }
}

fn build_http_client(timeout: Option<Duration>) -> Result<Client> {
    let mut builder = Client::builder();
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().context("building reqwest client")
}

fn apply_auth(builder: RequestBuilder, api_key: &str, auth_scheme: AuthScheme) -> RequestBuilder {
    match auth_scheme {
        AuthScheme::Bearer => builder.bearer_auth(api_key),
        AuthScheme::ApiKeyHeader(name) => builder.header(name, api_key),
    }
}

/// Translate an OpenAI-compatible SSE event stream into
/// [`ChatCompletionChunk`]s. Each well-formed `data: {...}` event
/// deserializes directly; the sentinel `data: [DONE]` is filtered out.
fn sse_events_to_chunks<S>(events: S) -> impl Stream<Item = Result<ChatCompletionChunk>>
where
    S: Stream<Item = Result<eventsource_stream::Event>>,
{
    try_stream! {
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            let event = event?;
            let data = event.data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let chunk: ChatCompletionChunk = serde_json::from_str(data)
                .map_err(|e| anyhow!("failed to parse SSE chunk `{data}`: {e}"))?;
            yield chunk;
        }
    }
}

/// Fold a non-SSE JSON [`ChatResponse`] into a single [`ChatCompletionChunk`]
/// so the rest of the pipeline stays stream-shaped.
fn json_response_to_chunk(body: &str) -> Result<ChatCompletionChunk> {
    let resp: ChatResponse = serde_json::from_str(body)
        .with_context(|| format!("parsing non-SSE JSON response `{body}`"))?;

    let choices = resp
        .choices
        .into_iter()
        .map(|c| ChunkChoice {
            index: c.index,
            delta: ChatDelta {
                role: Some(c.message.role),
                content: message_content_to_text(c.message.content),
                tool_calls: c.message.tool_calls,
                extra: Default::default(),
            },
            finish_reason: c.finish_reason,
        })
        .collect();

    Ok(ChatCompletionChunk {
        id: resp.id,
        object: "chat.completion.chunk".to_string(),
        created: resp.created,
        model: resp.model,
        choices,
        usage: resp.usage,
        extra: resp.extra,
    })
}

fn message_content_to_text(content: MessageContent) -> Option<String> {
    let text = match content {
        MessageContent::Text(s) => s,
        MessageContent::Parts(parts) => parts
            .into_iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    };
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventsource_stream::Event;
    use futures::stream;

    fn sse_event(data: &str) -> Event {
        Event {
            event: "message".to_string(),
            data: data.to_string(),
            id: String::new(),
            retry: None,
        }
    }

    #[tokio::test]
    async fn sse_events_decode_into_chunks() {
        let chunk_json = |content: &str, finish: Option<&str>| {
            let finish = finish
                .map(|f| format!("\"{f}\""))
                .unwrap_or_else(|| "null".to_string());
            format!(
                r#"{{"id":"id-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{{"index":0,"delta":{{"content":"{content}"}},"finish_reason":{finish}}}]}}"#
            )
        };
        let payloads = [
            chunk_json("Hello ", None),
            chunk_json("world", Some("stop")),
        ];
        let events = stream::iter(
            payloads
                .iter()
                .map(|p| Ok::<_, anyhow::Error>(sse_event(p)))
                .collect::<Vec<_>>(),
        );
        let chunks = sse_events_to_chunks(events);
        futures::pin_mut!(chunks);
        let mut out = Vec::new();
        while let Some(c) = chunks.next().await {
            out.push(c.unwrap());
        }
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("Hello "));
        assert_eq!(out[1].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn sse_done_sentinel_is_filtered_out() {
        let payloads = [
            r#"{"id":"id-1","model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}"#
                .to_string(),
            "[DONE]".to_string(),
        ];
        let events = stream::iter(
            payloads
                .iter()
                .map(|p| Ok::<_, anyhow::Error>(sse_event(p)))
                .collect::<Vec<_>>(),
        );
        let chunks = sse_events_to_chunks(events);
        futures::pin_mut!(chunks);
        let mut out = Vec::new();
        while let Some(c) = chunks.next().await {
            out.push(c.unwrap());
        }
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn json_response_folds_to_single_chunk() {
        let body = r#"{
            "id": "abc",
            "object": "chat.completion",
            "created": 42,
            "model": "m",
            "choices": [
                {"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        }"#;
        let chunk = json_response_to_chunk(body).unwrap();
        assert_eq!(chunk.id, "abc");
        assert_eq!(chunk.model, "m");
        assert_eq!(chunk.created, 42);
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.role.as_deref(),
            Some("assistant"),
            "fallback should carry role through",
        );
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi there"),);
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn json_response_falls_back_to_empty_content_when_message_empty() {
        let body = r#"{
            "id": "abc",
            "model": "m",
            "choices": [
                {"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}
            ]
        }"#;
        let chunk = json_response_to_chunk(body).unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[tokio::test]
    async fn chat_completion_parses_sse_response() {
        let sse = sse_body(&[
            r#"{"id":"id-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"id-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hi!"},"finish_reason":null}]}"#,
            r#"{"id":"id-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"[DONE]"#,
        ]);
        let mock = test_support::start_mock_chat(200, "text/event-stream", sse).await;
        let url = format!("http://{}/chat/completions", mock.addr);
        let stream = chat_completion(
            "test",
            &url,
            "sk-test",
            AuthScheme::Bearer,
            None,
            test_support::sample_request(),
        )
        .await
        .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(chunks.len(), 3, "DONE should be filtered out");
        let text: String = chunks
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        assert_eq!(text, "Hi!");

        let body = mock.captured.body();
        assert!(
            body.contains("\"stream\":true"),
            "stream=true should be forced: {body}"
        );
        let auth = mock.captured.header("authorization").unwrap_or_default();
        assert_eq!(auth, "Bearer sk-test");
    }

    #[tokio::test]
    async fn chat_completion_falls_back_to_json_body() {
        let json = r#"{
            "id": "abc",
            "object": "chat.completion",
            "created": 42,
            "model": "m",
            "choices": [
                {"index":0,"message":{"role":"assistant","content":"single shot"},"finish_reason":"stop"}
            ]
        }"#;
        let mock = test_support::start_mock_chat(200, "application/json", json.to_string()).await;
        let url = format!("http://{}/chat/completions", mock.addr);
        let stream = chat_completion(
            "test",
            &url,
            "sk-test",
            AuthScheme::Bearer,
            None,
            test_support::sample_request(),
        )
        .await
        .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let chunks: Vec<_> = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(chunks.len(), 1, "non-SSE JSON should yield one chunk");
        assert_eq!(
            chunks[0].choices[0].delta.content.as_deref(),
            Some("single shot"),
        );
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn chat_completion_maps_401_to_authentication_error() {
        let mock = test_support::start_mock_chat(
            401,
            "application/json",
            r#"{"error":{"message":"bad key"}}"#.to_string(),
        )
        .await;
        let url = format!("http://{}/chat/completions", mock.addr);
        let result = chat_completion(
            "minimax",
            &url,
            "wrong-key",
            AuthScheme::Bearer,
            None,
            test_support::sample_request(),
        )
        .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("401 should surface as an error"),
        };
        let auth = err
            .downcast_ref::<AuthenticationError>()
            .expect("401 should downcast to AuthenticationError");
        assert!(
            auth.message.contains("mixer auth login minimax"),
            "message should include the login command: {}",
            auth.message
        );
    }

    #[tokio::test]
    async fn chat_completion_surfaces_5xx_with_status_and_body() {
        let mock = test_support::start_mock_chat(
            500,
            "application/json",
            r#"{"error":{"message":"upstream boom"}}"#.to_string(),
        )
        .await;
        let url = format!("http://{}/chat/completions", mock.addr);
        let result = chat_completion(
            "test",
            &url,
            "sk-test",
            AuthScheme::Bearer,
            None,
            test_support::sample_request(),
        )
        .await;
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("5xx should error"),
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("500"), "error should include status: {msg}");
        assert!(
            msg.contains("upstream boom"),
            "error should include body: {msg}"
        );
    }

    #[tokio::test]
    async fn chat_completion_uses_custom_api_key_header() {
        let sse = sse_body(&[
            r#"{"id":"id-1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        ]);
        let mock = test_support::start_mock_chat(200, "text/event-stream", sse).await;
        let url = format!("http://{}/chat/completions", mock.addr);
        let stream = chat_completion(
            "opencode",
            &url,
            "sk-zen",
            AuthScheme::ApiKeyHeader("x-api-key"),
            None,
            test_support::sample_request(),
        )
        .await
        .unwrap();
        let _: Vec<_> = stream.collect().await;
        assert_eq!(mock.captured.header("x-api-key").as_deref(), Some("sk-zen"));
        assert!(mock.captured.header("authorization").is_none());
    }

    fn sse_body(payloads: &[&str]) -> String {
        payloads.iter().map(|p| format!("data: {p}\n\n")).collect()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Tiny axum-backed mock used by this module's tests and by each
    //! API-key provider's tests. One route: `POST /chat/completions`. Each
    //! call to [`start_mock_chat`] returns a listening socket bound to a
    //! random port plus a handle for asserting on the captured request
    //! headers and body.

    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    use axum::{
        Router,
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::Response,
        routing::post,
    };
    use tokio::net::TcpListener;

    use crate::openai::ChatRequest;

    pub struct MockChat {
        pub addr: SocketAddr,
        pub captured: Captured,
    }

    #[derive(Default, Clone)]
    pub struct Captured {
        headers: Arc<Mutex<Vec<(String, String)>>>,
        body: Arc<Mutex<String>>,
    }

    impl Captured {
        pub fn header(&self, name: &str) -> Option<String> {
            let name = name.to_ascii_lowercase();
            self.headers
                .lock()
                .unwrap()
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.clone())
        }

        pub fn body(&self) -> String {
            self.body.lock().unwrap().clone()
        }
    }

    struct MockState {
        captured: Captured,
        status: u16,
        content_type: &'static str,
        body: String,
    }

    async fn handler(
        State(state): State<Arc<MockState>>,
        headers: HeaderMap,
        body: String,
    ) -> Response {
        let snapshot: Vec<(String, String)> = headers
            .iter()
            .map(|(n, v)| {
                (
                    n.as_str().to_ascii_lowercase(),
                    v.to_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        *state.captured.headers.lock().unwrap() = snapshot;
        *state.captured.body.lock().unwrap() = body;
        Response::builder()
            .status(StatusCode::from_u16(state.status).unwrap())
            .header("Content-Type", state.content_type)
            .body(Body::from(state.body.clone()))
            .unwrap()
    }

    pub async fn start_mock_chat(
        status: u16,
        content_type: &'static str,
        body: String,
    ) -> MockChat {
        let captured = Captured::default();
        let state = MockState {
            captured: captured.clone(),
            status,
            content_type,
            body,
        };
        let app = Router::new()
            .route("/chat/completions", post(handler))
            .with_state(Arc::new(state));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        MockChat { addr, captured }
    }

    pub fn sample_request() -> ChatRequest {
        serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap()
    }
}
