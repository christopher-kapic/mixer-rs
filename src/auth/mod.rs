//! Shared authentication helpers usable by any provider.
//!
//! Currently ships the RFC 8628 device-authorization flow. Future additions
//! (OAuth 2.0 PKCE, etc.) land here alongside it so each provider just picks
//! the helper that matches its upstream.

pub mod device_flow;
