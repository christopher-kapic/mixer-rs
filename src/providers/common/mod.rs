//! Helpers shared between providers whose upstream wire protocols overlap.
//!
//! Ships the OpenAI Chat Completions ↔ OpenAI Responses API translator first —
//! reused by `codex` and any future provider that speaks the Responses API.

pub mod api_key_login;
pub mod oauth_refresh;
pub mod openai_client;
pub mod responses_api;
