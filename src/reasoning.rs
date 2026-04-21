//! Reasoning / chain-of-thought normalization pipeline.
//!
//! Every upstream model has its own convention for surfacing chain-of-thought
//! alongside the primary output. Without normalization, clients see a different
//! rendering per model — literal `<think>…</think>` tags for Qwen, structured
//! `reasoning_content` deltas for DeepSeek/Kimi/GLM/MiniMax, unhandled Responses
//! API reasoning events for GPT-5. Consumers then have to special-case each.
//!
//! The pipeline here collapses all of those dialects onto a single canonical
//! shape — `delta.reasoning_content` for streaming, `message.reasoning_content`
//! for non-streaming — and then optionally renders the canonical shape back to
//! an alternate wire format for the client.
//!
//! Two stages, composed with [`StreamExt`]:
//!
//! - [`normalize`] takes the provider's raw chunk stream and a declared
//!   [`ReasoningFormat`] (from the picked [`ModelInfo`]), and produces a stream
//!   whose reasoning always lives on the typed field. For [`InlineThinkTags`],
//!   this runs the [`ThinkTagStripper`] state machine — a tag-aware splitter
//!   that tolerates partial matches at chunk boundaries.
//! - [`render`] takes the canonical stream and a [`ReasoningOutput`] preference
//!   and reshapes it for the wire (`structured` = passthrough, `inline_tags` =
//!   wrap reasoning back in `<think>…</think>` inside `content`, `omit` = drop
//!   the reasoning entirely).
//!
//! Both stages preserve chunk cadence: one input chunk in, one output chunk
//! out (the stripper also emits at most one deferred-bytes tail on EOS).
//!
//! [`InlineThinkTags`]: ReasoningFormat::InlineThinkTags

use async_stream::try_stream;
use futures::{Stream, StreamExt};

use crate::config::ReasoningOutput;
use crate::openai::{ChatCompletionChunk, ChatDelta, ChunkChoice};
use crate::providers::{ChatStream, ReasoningFormat};

/// Stage 1 (normalize): move the upstream's chain-of-thought onto the typed
/// `delta.reasoning_content` field regardless of how it arrived. Structured
/// and Responses-API formats already satisfy that invariant at the source, so
/// they pass through unchanged; [`ReasoningFormat::InlineThinkTags`] runs the
/// tag-stripper.
pub fn normalize(stream: ChatStream, format: ReasoningFormat) -> ChatStream {
    match format {
        ReasoningFormat::InlineThinkTags => Box::pin(strip_think_tags(stream)),
        ReasoningFormat::None
        | ReasoningFormat::Structured
        | ReasoningFormat::ResponsesApiSummary => stream,
    }
}

/// Stage 2 (render): reshape the canonical stream for the client wire. Runs
/// *after* [`normalize`], so every reasoning variant lands here as
/// `delta.reasoning_content`.
pub fn render(stream: ChatStream, mode: ReasoningOutput) -> ChatStream {
    match mode {
        ReasoningOutput::Structured => stream,
        ReasoningOutput::InlineTags => Box::pin(render_inline_tags(stream)),
        ReasoningOutput::Omit => Box::pin(render_omit(stream)),
    }
}

// ── Stage 1: tag-stripping state machine ───────────────────────────────────

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// Streaming `<think>…</think>` extractor. Operates on byte-level UTF-8 —
/// `<`, `>`, `/`, and the ASCII tag letters are all single-byte, so splits
/// never land inside a multi-byte char.
///
/// The only subtlety is chunk boundaries: an upstream can deliver `"<thi"` in
/// one delta and `"nk>"` in the next, so a `<` near the tail of a chunk is
/// buffered as *deferred* bytes and re-processed when the next chunk arrives.
/// The deferred buffer is bounded by `max(OPEN_TAG.len(), CLOSE_TAG.len())`.
#[derive(Default, Debug)]
struct ThinkTagStripper {
    inside_think: bool,
    deferred: String,
}

impl ThinkTagStripper {
    fn new() -> Self {
        Self::default()
    }

    /// Split `input` into (content bytes, reasoning bytes) according to the
    /// current state + any buffered tail from the previous chunk. Either side
    /// may be empty.
    fn process(&mut self, input: &str) -> (String, String) {
        let mut text = std::mem::take(&mut self.deferred);
        text.push_str(input);

        let mut content = String::new();
        let mut reasoning = String::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let target = if self.inside_think {
                CLOSE_TAG
            } else {
                OPEN_TAG
            };
            let lt_rel = bytes[i..].iter().position(|&b| b == b'<');

            let Some(rel) = lt_rel else {
                // No more `<` in the remainder — flush straight to the active side.
                let tail = &text[i..];
                if self.inside_think {
                    reasoning.push_str(tail);
                } else {
                    content.push_str(tail);
                }
                break;
            };

            let lt_pos = i + rel;
            // Everything strictly before the `<` is unambiguous.
            let head = &text[i..lt_pos];
            if self.inside_think {
                reasoning.push_str(head);
            } else {
                content.push_str(head);
            }

            let remaining = bytes.len() - lt_pos;
            let compare_len = remaining.min(target.len());
            let prefix_matches = text[lt_pos..lt_pos + compare_len] == target[..compare_len];

            if !prefix_matches {
                // This `<` can't possibly start our target tag (e.g. `<=`, `< `,
                // `</s>` when we're expecting `<think>`). Emit it as content
                // and keep scanning from the next byte.
                if self.inside_think {
                    reasoning.push('<');
                } else {
                    content.push('<');
                }
                i = lt_pos + 1;
            } else if compare_len == target.len() {
                // Full match — flip state and skip past the tag.
                self.inside_think = !self.inside_think;
                i = lt_pos + target.len();
            } else {
                // Prefix matches but the chunk is too short to confirm. Defer
                // the tail so the next chunk can complete the decision.
                self.deferred = text[lt_pos..].to_string();
                return (content, reasoning);
            }
        }

        (content, reasoning)
    }

    /// Stream end: whatever is still deferred was a lone `<` that never
    /// resolved into a tag. Route it to the currently-active side so nothing
    /// gets silently dropped.
    fn flush(&mut self) -> (String, String) {
        let deferred = std::mem::take(&mut self.deferred);
        let flipped_state = self.inside_think;
        self.inside_think = false;
        if flipped_state {
            (String::new(), deferred)
        } else {
            (deferred, String::new())
        }
    }
}

fn strip_think_tags(
    stream: ChatStream,
) -> impl Stream<Item = anyhow::Result<ChatCompletionChunk>> + Send {
    try_stream! {
        let mut stream = stream;
        let mut stripper = ThinkTagStripper::new();
        let mut last_id = String::new();
        let mut last_model = String::new();
        let mut last_created: i64 = 0;

        while let Some(item) = stream.next().await {
            let mut chunk = item?;
            if !chunk.id.is_empty() { last_id = chunk.id.clone(); }
            if !chunk.model.is_empty() { last_model = chunk.model.clone(); }
            if chunk.created != 0 { last_created = chunk.created; }

            for choice in chunk.choices.iter_mut() {
                let Some(content) = choice.delta.content.take() else { continue };
                let (content_out, reasoning_out) = stripper.process(&content);
                choice.delta.content = if content_out.is_empty() {
                    None
                } else {
                    Some(content_out)
                };
                if !reasoning_out.is_empty() {
                    // Append so a pre-existing structured reasoning delta (rare
                    // but allowed) merges cleanly with the stripped bytes.
                    let merged = match choice.delta.reasoning_content.take() {
                        Some(mut existing) => {
                            existing.push_str(&reasoning_out);
                            existing
                        }
                        None => reasoning_out,
                    };
                    choice.delta.reasoning_content = Some(merged);
                }
            }
            yield chunk;
        }

        let (tail_content, tail_reasoning) = stripper.flush();
        if !tail_content.is_empty() || !tail_reasoning.is_empty() {
            yield ChatCompletionChunk {
                id: last_id,
                object: "chat.completion.chunk".to_string(),
                created: last_created,
                model: last_model,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        content: (!tail_content.is_empty()).then_some(tail_content),
                        reasoning_content: (!tail_reasoning.is_empty()).then_some(tail_reasoning),
                        ..Default::default()
                    },
                    finish_reason: None,
                }],
                usage: None,
                extra: Default::default(),
            };
        }
    }
}

// ── Stage 2: output rendering ──────────────────────────────────────────────

fn render_inline_tags(
    stream: ChatStream,
) -> impl Stream<Item = anyhow::Result<ChatCompletionChunk>> + Send {
    try_stream! {
        let mut stream = stream;
        // Track per-choice state so opening/closing tags are emitted exactly
        // once per reasoning span, regardless of how many deltas it takes to
        // arrive. Indexed by choice.index so parallel choices don't interfere.
        let mut inside: std::collections::HashMap<u32, bool> = std::collections::HashMap::new();

        while let Some(item) = stream.next().await {
            let mut chunk = item?;
            for choice in chunk.choices.iter_mut() {
                let was_inside = *inside.get(&choice.index).unwrap_or(&false);
                let Some(reasoning) = choice.delta.reasoning_content.take() else {
                    if was_inside {
                        // A chunk with no reasoning after we've been streaming
                        // reasoning means the thinking phase just ended. Close
                        // the tag, then preserve whatever else the chunk carries
                        // (text content, finish_reason, etc.).
                        inside.insert(choice.index, false);
                        let closed = format!(
                            "</think>{}",
                            choice.delta.content.take().unwrap_or_default(),
                        );
                        choice.delta.content = Some(closed);
                    }
                    continue;
                };
                // Reasoning present — emit an opening tag on the first chunk of
                // the span, append the bytes, leave the closing tag for later.
                let mut inline = String::new();
                if !was_inside {
                    inline.push_str("<think>");
                    inside.insert(choice.index, true);
                }
                inline.push_str(&reasoning);
                // If this delta also carries primary content, the thinking span
                // has ended mid-delta — close the tag before the content.
                if let Some(content) = choice.delta.content.take() {
                    inline.push_str("</think>");
                    inline.push_str(&content);
                    inside.insert(choice.index, false);
                }
                choice.delta.content = Some(inline);
            }
            yield chunk;
        }
    }
}

fn render_omit(
    stream: ChatStream,
) -> impl Stream<Item = anyhow::Result<ChatCompletionChunk>> + Send {
    try_stream! {
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            let mut chunk = item?;
            for choice in chunk.choices.iter_mut() {
                choice.delta.reasoning_content = None;
            }
            yield chunk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn chunk(content: Option<&str>, reasoning: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "id-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1,
            model: "m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChatDelta {
                    content: content.map(str::to_string),
                    reasoning_content: reasoning.map(str::to_string),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
            extra: Default::default(),
        }
    }

    fn into_stream(chunks: Vec<ChatCompletionChunk>) -> ChatStream {
        Box::pin(stream::iter(chunks.into_iter().map(Ok)))
    }

    async fn collect_into(stream: ChatStream) -> Vec<ChatCompletionChunk> {
        let mut stream = stream;
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.expect("no stream errors"));
        }
        out
    }

    // ── ThinkTagStripper unit tests ────────────────────────────────────────

    #[test]
    fn stripper_passes_through_when_no_tags() {
        let mut s = ThinkTagStripper::new();
        let (c, r) = s.process("hello world");
        assert_eq!(c, "hello world");
        assert_eq!(r, "");
    }

    #[test]
    fn stripper_extracts_fully_contained_think_block() {
        let mut s = ThinkTagStripper::new();
        let (c, r) = s.process("pre<think>reasoning bytes</think>post");
        assert_eq!(c, "prepost");
        assert_eq!(r, "reasoning bytes");
        assert!(!s.inside_think);
    }

    #[test]
    fn stripper_handles_open_tag_split_across_chunks() {
        let mut s = ThinkTagStripper::new();
        let (c1, r1) = s.process("before<thi");
        assert_eq!(c1, "before");
        assert_eq!(r1, "");

        let (c2, r2) = s.process("nk>mid</thi");
        assert_eq!(c2, "");
        assert_eq!(r2, "mid");

        let (c3, r3) = s.process("nk>after");
        assert_eq!(c3, "after");
        assert_eq!(r3, "");
    }

    #[test]
    fn stripper_handles_close_tag_split_at_byte_boundary() {
        let mut s = ThinkTagStripper::new();
        let (_, _) = s.process("<think>abc");
        let (c, r) = s.process("def</th");
        assert_eq!(c, "");
        assert_eq!(r, "def");
        let (c2, r2) = s.process("ink>tail");
        assert_eq!(c2, "tail");
        assert_eq!(r2, "");
    }

    #[test]
    fn stripper_passes_bare_lt_through() {
        // `<` not followed by `think>` or `/think>` is literal content, not a tag.
        let mut s = ThinkTagStripper::new();
        let (c, r) = s.process("x < y and a <= b");
        assert_eq!(c, "x < y and a <= b");
        assert_eq!(r, "");
    }

    #[test]
    fn stripper_handles_tag_like_literal_inside_reasoning() {
        let mut s = ThinkTagStripper::new();
        let (c, r) = s.process("<think>a<b and c</think>z");
        assert_eq!(c, "z");
        assert_eq!(r, "a<b and c");
    }

    #[test]
    fn stripper_flush_emits_unresolved_deferred_bytes() {
        // Stream ends right after `<` — the deferred `<` should surface as
        // literal content in the flush.
        let mut s = ThinkTagStripper::new();
        let (c, _) = s.process("hello<");
        assert_eq!(c, "hello");
        let (tc, tr) = s.flush();
        assert_eq!(tc, "<");
        assert_eq!(tr, "");
    }

    // ── Stage 1 (normalize) stream tests ───────────────────────────────────

    #[tokio::test]
    async fn normalize_passes_structured_through() {
        let stream = into_stream(vec![chunk(Some("hi"), Some("thought"))]);
        let out = collect_into(normalize(stream, ReasoningFormat::Structured)).await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("hi"));
        assert_eq!(
            out[0].choices[0].delta.reasoning_content.as_deref(),
            Some("thought"),
        );
    }

    #[tokio::test]
    async fn normalize_strips_inline_think_tags_in_single_chunk() {
        let stream = into_stream(vec![chunk(
            Some("pre<think>the reasoning</think>after"),
            None,
        )]);
        let out = collect_into(normalize(stream, ReasoningFormat::InlineThinkTags)).await;
        let merged_content: String = out
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        let merged_reasoning: String = out
            .iter()
            .flat_map(|c| {
                c.choices
                    .iter()
                    .filter_map(|ch| ch.delta.reasoning_content.clone())
            })
            .collect();
        assert_eq!(merged_content, "preafter");
        assert_eq!(merged_reasoning, "the reasoning");
    }

    #[tokio::test]
    async fn normalize_strips_inline_think_tags_across_chunk_boundaries() {
        // Realistic Qwen-style stream: opening tag splits across two chunks,
        // closing tag splits across two more.
        let chunks = vec![
            chunk(Some("hello <thi"), None),
            chunk(Some("nk>a"), None),
            chunk(Some("bc</thi"), None),
            chunk(Some("nk> world"), None),
        ];
        let out = collect_into(normalize(
            into_stream(chunks),
            ReasoningFormat::InlineThinkTags,
        ))
        .await;
        let merged_content: String = out
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        let merged_reasoning: String = out
            .iter()
            .flat_map(|c| {
                c.choices
                    .iter()
                    .filter_map(|ch| ch.delta.reasoning_content.clone())
            })
            .collect();
        assert_eq!(merged_content, "hello  world");
        assert_eq!(merged_reasoning, "abc");
    }

    // ── Stage 2 (render) stream tests ──────────────────────────────────────

    #[tokio::test]
    async fn render_structured_is_passthrough() {
        let stream = into_stream(vec![chunk(Some("hi"), Some("t"))]);
        let out = collect_into(render(stream, ReasoningOutput::Structured)).await;
        assert_eq!(
            out[0].choices[0].delta.reasoning_content.as_deref(),
            Some("t"),
        );
    }

    #[tokio::test]
    async fn render_inline_tags_wraps_reasoning_into_content() {
        let chunks = vec![
            chunk(None, Some("first ")),
            chunk(None, Some("second")),
            chunk(Some("visible"), None),
        ];
        let out = collect_into(render(into_stream(chunks), ReasoningOutput::InlineTags)).await;
        let merged: String = out
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        assert_eq!(merged, "<think>first second</think>visible");
        assert!(
            out.iter().all(|c| c
                .choices
                .iter()
                .all(|ch| ch.delta.reasoning_content.is_none())),
            "inline_tags must move reasoning out of the reasoning_content field",
        );
    }

    #[tokio::test]
    async fn render_inline_tags_closes_span_when_reasoning_and_content_share_chunk() {
        // Some upstreams emit the final reasoning byte and the first content
        // byte in the same chunk. The tag-closer must fire in that chunk.
        let chunks = vec![chunk(Some("answer"), Some("end of thought"))];
        let out = collect_into(render(into_stream(chunks), ReasoningOutput::InlineTags)).await;
        let merged: String = out
            .iter()
            .flat_map(|c| c.choices.iter().filter_map(|ch| ch.delta.content.clone()))
            .collect();
        assert_eq!(merged, "<think>end of thought</think>answer");
    }

    #[tokio::test]
    async fn render_omit_drops_reasoning_content_field() {
        let stream = into_stream(vec![chunk(Some("hi"), Some("drop me"))]);
        let out = collect_into(render(stream, ReasoningOutput::Omit)).await;
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("hi"));
        assert!(out[0].choices[0].delta.reasoning_content.is_none());
    }
}
