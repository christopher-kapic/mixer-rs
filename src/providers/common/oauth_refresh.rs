//! OAuth refresh primitives shared by device-flow providers (codex today,
//! GitHub Copilot later). Per plan.md §3.6.3 the strategy is:
//!
//!   1. Proactive expiry check at request time — refresh when `exp` is within
//!      [`EXPIRY_THRESHOLD_SECS`] of now.
//!   2. 401 safety net — one refresh-and-retry after a 401 from the upstream.
//!   3. Refresh-endpoint rejection surfaces as [`AuthenticationError`], which
//!      the server layer downcasts into an OpenAI-style 401 with an actionable
//!      "run `mixer auth login <provider>`" message.
//!
//! Concurrency is handled via a per-provider [`tokio::sync::Mutex`]: the first
//! task to enter refresh holds the lock while the rest wait, then re-read the
//! credentials on acquiring the lock and find the fresh token already written.
//! See [`provider_refresh_lock`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use serde_json::Value;
use tokio::sync::Mutex;

/// Proactive refresh threshold. If the access token expires within this many
/// seconds, refresh before dispatching the request.
pub const EXPIRY_THRESHOLD_SECS: i64 = 60;

/// Returned when the refresh endpoint itself rejects our refresh token
/// (revoked, password changed, session expired). The server layer downcasts
/// this into a 401 `authentication_error` body so clients see a real auth
/// error and users see an actionable fix.
#[derive(Debug, Clone)]
pub struct AuthenticationError {
    pub message: String,
}

impl std::fmt::Display for AuthenticationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AuthenticationError {}

/// Decode the `exp` claim from a JWT without verifying the signature. Returns
/// `None` if the token is not a well-formed JWT or the claim is absent.
pub fn decode_jwt_exp(token: &str) -> Option<i64> {
    use std::collections::HashSet;

    use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};

    let mut validation = Validation::new(Algorithm::HS256);
    validation.insecure_disable_signature_validation();
    validation.validate_exp = false;
    validation.validate_nbf = false;
    validation.validate_aud = false;
    validation.required_spec_claims = HashSet::new();

    let data = decode::<Value>(token, &DecodingKey::from_secret(b"ignored"), &validation).ok()?;
    data.claims.get("exp").and_then(Value::as_i64)
}

/// Per-provider mutex used to serialize token refreshes. Returns the same
/// `Arc<Mutex>` for the same `provider_id` across the process's lifetime so
/// concurrent in-flight requests for that provider coordinate a single refresh.
pub fn provider_refresh_lock(provider_id: &str) -> Arc<Mutex<()>> {
    static REGISTRY: OnceLock<std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = registry.lock().unwrap();
    guard
        .entry(provider_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Read the absolute `expires_at` (unix seconds) for a stored credential blob.
/// Prefers the explicit `expires_at` field written at login/refresh time, and
/// falls back to decoding the access token's `exp` claim.
pub fn current_expiry(blob: &Value) -> Option<i64> {
    if let Some(exp) = blob.get("expires_at").and_then(Value::as_i64) {
        return Some(exp);
    }
    let access_token = blob.get("access_token").and_then(Value::as_str)?;
    decode_jwt_exp(access_token)
}

/// Is the stored access token within `threshold_secs` of its expiry? Missing
/// expiry is treated as "not near" — we fall back to the 401 retry path
/// rather than proactively refreshing a token we can't reason about.
pub fn is_near_expiry(blob: &Value, now: i64, threshold_secs: i64) -> bool {
    match current_expiry(blob) {
        Some(exp) => exp - now <= threshold_secs,
        None => false,
    }
}

/// Three-way classification of a device-flow credential blob. Drives both
/// `Provider::is_authenticated` and the `(oauth, …)` suffix emitted by
/// `mixer auth status` (per plan.md §3.6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OauthFreshness {
    /// `expires_at` is in the future (or missing — treat as "still good" so
    /// older blobs written before `expires_at` tracking existed don't surface
    /// as spuriously expired; the real clock check happens at dispatch time).
    Valid,
    /// Access token is expired but a non-empty `refresh_token` is stored, so
    /// the next dispatch can transparently recover.
    ExpiredRefreshable,
    /// Access token is expired and there is no refresh token — only a full
    /// `mixer auth login` can recover.
    ExpiredDead,
}

/// Classify the OAuth freshness of a device-flow credential blob relative to
/// `now_secs`. Uses [`current_expiry`] so both the explicit `expires_at`
/// field and a JWT `exp` claim on the access token are honored.
pub fn oauth_freshness(blob: &Value, now_secs: i64) -> OauthFreshness {
    let expires_at = current_expiry(blob);
    let has_refresh = blob
        .get("refresh_token")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    match expires_at {
        Some(exp) if exp <= now_secs => {
            if has_refresh {
                OauthFreshness::ExpiredRefreshable
            } else {
                OauthFreshness::ExpiredDead
            }
        }
        _ => OauthFreshness::Valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use serde_json::json;

    fn jwt_with(claims: &Value) -> String {
        encode(
            &Header::default(),
            claims,
            &EncodingKey::from_secret(b"test"),
        )
        .unwrap()
    }

    #[test]
    fn decode_jwt_exp_returns_claim() {
        let token = jwt_with(&json!({ "exp": 1_700_000_000_i64 }));
        assert_eq!(decode_jwt_exp(&token), Some(1_700_000_000));
    }

    #[test]
    fn decode_jwt_exp_returns_none_for_malformed() {
        assert_eq!(decode_jwt_exp("not-a-jwt"), None);
        assert_eq!(decode_jwt_exp("aaa.bbb"), None);
    }

    #[test]
    fn decode_jwt_exp_returns_none_when_claim_absent() {
        let token = jwt_with(&json!({ "sub": "user" }));
        assert_eq!(decode_jwt_exp(&token), None);
    }

    #[test]
    fn current_expiry_prefers_expires_at_field() {
        let token = jwt_with(&json!({ "exp": 999_i64 }));
        let blob = json!({ "expires_at": 12345, "access_token": token });
        assert_eq!(current_expiry(&blob), Some(12345));
    }

    #[test]
    fn current_expiry_falls_back_to_jwt_exp() {
        let token = jwt_with(&json!({ "exp": 777_i64 }));
        let blob = json!({ "access_token": token });
        assert_eq!(current_expiry(&blob), Some(777));
    }

    #[test]
    fn current_expiry_none_when_both_missing() {
        let blob = json!({ "access_token": "opaque" });
        assert_eq!(current_expiry(&blob), None);
    }

    #[test]
    fn is_near_expiry_respects_threshold() {
        let blob = json!({ "expires_at": 1000_i64 });
        assert!(is_near_expiry(&blob, 950, 60));
        assert!(is_near_expiry(&blob, 999, 60));
        assert!(!is_near_expiry(&blob, 900, 60));
    }

    #[test]
    fn is_near_expiry_false_when_unknown() {
        let blob = json!({ "access_token": "opaque" });
        assert!(!is_near_expiry(&blob, 0, 60));
    }

    #[test]
    fn authentication_error_display_matches_message() {
        let e = AuthenticationError {
            message: "boom".to_string(),
        };
        assert_eq!(format!("{e}"), "boom");
    }

    #[test]
    fn oauth_freshness_valid_when_expiry_in_future() {
        let blob = json!({ "expires_at": 1_000_i64, "refresh_token": "rt" });
        assert_eq!(oauth_freshness(&blob, 500), OauthFreshness::Valid);
    }

    #[test]
    fn oauth_freshness_valid_when_expiry_unknown() {
        // No expires_at, no JWT access_token — treat as Valid to avoid
        // spuriously reporting old blobs as expired in `mixer auth status`.
        let blob = json!({ "access_token": "opaque" });
        assert_eq!(oauth_freshness(&blob, 99_999), OauthFreshness::Valid);
    }

    #[test]
    fn oauth_freshness_expired_refreshable_when_refresh_token_present() {
        let blob = json!({ "expires_at": 100_i64, "refresh_token": "rt" });
        assert_eq!(
            oauth_freshness(&blob, 500),
            OauthFreshness::ExpiredRefreshable,
        );
    }

    #[test]
    fn oauth_freshness_expired_dead_when_no_refresh_token() {
        let blob = json!({ "expires_at": 100_i64 });
        assert_eq!(oauth_freshness(&blob, 500), OauthFreshness::ExpiredDead);
    }

    #[test]
    fn oauth_freshness_uses_jwt_exp_fallback() {
        let token = jwt_with(&json!({ "exp": 100_i64 }));
        let blob = json!({ "access_token": token, "refresh_token": "rt" });
        assert_eq!(
            oauth_freshness(&blob, 500),
            OauthFreshness::ExpiredRefreshable,
        );
    }

    #[tokio::test]
    async fn provider_refresh_lock_returns_same_instance() {
        let a = provider_refresh_lock("oauth-refresh-test-same");
        let b = provider_refresh_lock("oauth-refresh-test-same");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn provider_refresh_lock_distinct_per_provider() {
        let a = provider_refresh_lock("oauth-refresh-test-X");
        let b = provider_refresh_lock("oauth-refresh-test-Y");
        assert!(!Arc::ptr_eq(&a, &b));
    }
}
