//! OAuth 2.0 Device Authorization Grant (RFC 8628) helper.
//!
//! Shared across providers whose upstream supports the device flow (codex,
//! and — later — GitHub Copilot). Callers supply a [`DeviceFlowConfig`] with
//! `client_id`, scopes, and the two endpoint URLs; [`run_device_flow`] does
//! the rest:
//!
//! 1. POST to `device_authorization_url` and parse the `user_code` +
//!    `verification_uri`.
//! 2. Print instructions to stderr and attempt to open the verification URL
//!    in the user's browser (best-effort).
//! 3. Poll `token_url` at the server-specified `interval`, respecting the
//!    RFC 8628 error codes:
//!
//!    - `authorization_pending` → keep polling
//!    - `slow_down` → add 5s to the interval and keep polling
//!    - `expired_token` / `access_denied` → return an error
//!
//! 4. Return on success or when `expires_in` elapses.

// The first consumer (codex login) lands in a later step of the plan; until
// then this helper is only exercised by its own tests.
#![allow(dead_code)]

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tokio::time::{Instant, sleep};

/// RFC 8628 §3.4 — grant_type used when exchanging the device code for
/// tokens.
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// RFC 8628 §3.5 — on a `slow_down` error the client MUST increase its
/// polling interval by 5 seconds.
const SLOW_DOWN_INCREMENT: Duration = Duration::from_secs(5);

/// All the per-provider bits [`run_device_flow`] needs. No codex-specific
/// defaults live here — everything is supplied by the caller so this helper
/// stays reusable.
#[derive(Debug, Clone)]
pub struct DeviceFlowConfig {
    pub client_id: String,
    pub scopes: Vec<String>,
    pub device_authorization_url: String,
    pub token_url: String,
    /// OAuth `audience` parameter when the provider requires one.
    pub audience: Option<String>,
    /// Extra form fields sent on *both* the device-authorization and token
    /// requests (e.g. `client_secret` for providers that still require one).
    pub extra_params: Vec<(String, String)>,
    /// Attempt to open the verification URL in the user's browser. Disabled
    /// in tests so `cargo test` does not launch real browser tabs.
    pub open_browser: bool,
}

/// Tokens returned by a successful device-authorization flow. The full JSON
/// body is kept on `raw` so providers with extra claims (`chatgpt-account-id`
/// in the ID token, etc.) don't need new fields on this struct.
#[derive(Debug, Clone)]
pub struct DeviceFlowTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub expires_in: Option<u64>,
    pub token_type: String,
    pub scope: Option<String>,
    pub raw: Value,
}

#[derive(Debug, Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    #[serde(default = "default_interval")]
    interval: u64,
}

fn default_interval() -> u64 {
    5
}

#[derive(Debug, Deserialize)]
struct TokenErrorResponse {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

/// Execute the full RFC 8628 device-authorization dance. Returns once the
/// user has authorized (success), denied, or the device code has expired.
pub async fn run_device_flow(cfg: &DeviceFlowConfig, client: &Client) -> Result<DeviceFlowTokens> {
    let auth = request_device_authorization(cfg, client).await?;

    print_user_instructions(&auth);

    if cfg.open_browser {
        let browser_url = auth
            .verification_uri_complete
            .clone()
            .unwrap_or_else(|| auth.verification_uri.clone());
        let _ = open::that_detached(browser_url);
    }

    let deadline = Instant::now() + Duration::from_secs(auth.expires_in);
    let mut interval = Duration::from_secs(auth.interval);

    loop {
        if Instant::now() >= deadline {
            bail!(
                "device authorization timed out after {}s — run login again",
                auth.expires_in
            );
        }

        match poll_token(cfg, client, &auth.device_code).await? {
            PollOutcome::Success(tokens) => return Ok(tokens),
            PollOutcome::Pending => {}
            PollOutcome::SlowDown => {
                interval += SLOW_DOWN_INCREMENT;
            }
            PollOutcome::Expired => {
                bail!("device code expired before the user authorized — run login again");
            }
            PollOutcome::Denied => {
                bail!("user denied the authorization request");
            }
            PollOutcome::Other { code, description } => {
                let desc = description.as_deref().unwrap_or("(no description)");
                bail!("device flow token endpoint returned error `{code}`: {desc}");
            }
        }

        sleep(interval).await;
    }
}

async fn request_device_authorization(
    cfg: &DeviceFlowConfig,
    client: &Client,
) -> Result<DeviceAuthorizationResponse> {
    let scope_str = cfg.scopes.join(" ");
    let mut form: Vec<(&str, &str)> = vec![("client_id", cfg.client_id.as_str())];
    if !scope_str.is_empty() {
        form.push(("scope", scope_str.as_str()));
    }
    if let Some(audience) = cfg.audience.as_deref() {
        form.push(("audience", audience));
    }
    for (k, v) in &cfg.extra_params {
        form.push((k.as_str(), v.as_str()));
    }

    let resp = client
        .post(&cfg.device_authorization_url)
        .form(&form)
        .send()
        .await
        .context("requesting device authorization")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("device authorization endpoint returned {status}: {body}");
    }

    resp.json::<DeviceAuthorizationResponse>()
        .await
        .context("parsing device authorization response")
}

fn print_user_instructions(auth: &DeviceAuthorizationResponse) {
    eprintln!();
    eprintln!("To complete sign-in, open the following URL and enter the code:");
    eprintln!("  URL:  {}", auth.verification_uri);
    eprintln!("  Code: {}", auth.user_code);
    if let Some(complete) = &auth.verification_uri_complete {
        eprintln!("  (or open directly: {complete})");
    }
    eprintln!();
}

enum PollOutcome {
    Success(DeviceFlowTokens),
    Pending,
    SlowDown,
    Expired,
    Denied,
    Other {
        code: String,
        description: Option<String>,
    },
}

async fn poll_token(
    cfg: &DeviceFlowConfig,
    client: &Client,
    device_code: &str,
) -> Result<PollOutcome> {
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", DEVICE_CODE_GRANT_TYPE),
        ("device_code", device_code),
        ("client_id", cfg.client_id.as_str()),
    ];
    for (k, v) in &cfg.extra_params {
        form.push((k.as_str(), v.as_str()));
    }

    let resp = client
        .post(&cfg.token_url)
        .form(&form)
        .send()
        .await
        .context("polling token endpoint")?;

    let status = resp.status();
    let bytes = resp
        .bytes()
        .await
        .context("reading token endpoint response body")?;

    if status.is_success() {
        let raw: Value = serde_json::from_slice(&bytes).context("parsing token response JSON")?;
        let access_token = raw
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("token response missing `access_token`"))?
            .to_string();
        let token_type = raw
            .get("token_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Bearer")
            .to_string();
        return Ok(PollOutcome::Success(DeviceFlowTokens {
            access_token,
            refresh_token: raw
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            id_token: raw
                .get("id_token")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            expires_in: raw.get("expires_in").and_then(|v| v.as_u64()),
            token_type,
            scope: raw
                .get("scope")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            raw,
        }));
    }

    let err: TokenErrorResponse = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parsing token endpoint error response ({status}): {}",
            String::from_utf8_lossy(&bytes)
        )
    })?;

    Ok(match err.error.as_str() {
        "authorization_pending" => PollOutcome::Pending,
        "slow_down" => PollOutcome::SlowDown,
        "expired_token" => PollOutcome::Expired,
        "access_denied" => PollOutcome::Denied,
        other => PollOutcome::Other {
            code: other.to_string(),
            description: err.error_description,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::State,
        http::StatusCode,
        response::{IntoResponse, Response},
        routing::post,
    };
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    /// What the mock `/token` endpoint should return for the next poll.
    #[derive(Clone, Copy)]
    enum TokenStep {
        Success,
        Pending,
        SlowDown,
        Expired,
        Denied,
    }

    struct MockState {
        steps: Mutex<Vec<TokenStep>>,
    }

    async fn device_auth_handler() -> Response {
        (
            StatusCode::OK,
            Json(json!({
                "device_code": "dc-test",
                "user_code": "USER-CODE",
                "verification_uri": "https://example.test/verify",
                "verification_uri_complete": "https://example.test/verify?code=USER-CODE",
                "expires_in": 60,
                "interval": 0,
            })),
        )
            .into_response()
    }

    async fn token_handler(State(state): State<Arc<MockState>>) -> Response {
        let step = {
            let mut q = state.steps.lock().unwrap();
            if q.is_empty() {
                TokenStep::Success
            } else {
                q.remove(0)
            }
        };

        match step {
            TokenStep::Success => (
                StatusCode::OK,
                Json(json!({
                    "access_token": "at-abc",
                    "refresh_token": "rt-abc",
                    "id_token": "id-abc",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "openid profile",
                })),
            )
                .into_response(),
            TokenStep::Pending => (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "authorization_pending" })),
            )
                .into_response(),
            TokenStep::SlowDown => (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "slow_down" })),
            )
                .into_response(),
            TokenStep::Expired => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "expired_token",
                    "error_description": "code expired"
                })),
            )
                .into_response(),
            TokenStep::Denied => (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "access_denied" })),
            )
                .into_response(),
        }
    }

    async fn start_mock_server(steps: Vec<TokenStep>) -> String {
        let state = Arc::new(MockState {
            steps: Mutex::new(steps),
        });
        let app = Router::new()
            .route("/device_authorization", post(device_auth_handler))
            .route("/token", post(token_handler))
            .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    fn test_config(base: &str) -> DeviceFlowConfig {
        DeviceFlowConfig {
            client_id: "test-client".to_string(),
            scopes: vec!["openid".to_string(), "profile".to_string()],
            device_authorization_url: format!("{base}/device_authorization"),
            token_url: format!("{base}/token"),
            audience: None,
            extra_params: vec![],
            open_browser: false,
        }
    }

    #[tokio::test]
    async fn happy_path_returns_tokens_on_first_poll() {
        let base = start_mock_server(vec![TokenStep::Success]).await;
        let cfg = test_config(&base);
        let tokens = run_device_flow(&cfg, &reqwest::Client::new())
            .await
            .expect("device flow should succeed");
        assert_eq!(tokens.access_token, "at-abc");
        assert_eq!(tokens.refresh_token.as_deref(), Some("rt-abc"));
        assert_eq!(tokens.id_token.as_deref(), Some("id-abc"));
        assert_eq!(tokens.token_type, "Bearer");
        assert_eq!(tokens.expires_in, Some(3600));
        assert_eq!(tokens.scope.as_deref(), Some("openid profile"));
        assert_eq!(tokens.raw.get("access_token").unwrap(), "at-abc");
    }

    #[tokio::test]
    async fn authorization_pending_then_success() {
        let base = start_mock_server(vec![
            TokenStep::Pending,
            TokenStep::Pending,
            TokenStep::Success,
        ])
        .await;
        let cfg = test_config(&base);
        let tokens = run_device_flow(&cfg, &reqwest::Client::new())
            .await
            .expect("device flow should succeed after pending polls");
        assert_eq!(tokens.access_token, "at-abc");
    }

    #[tokio::test]
    async fn slow_down_adds_five_seconds_and_continues() {
        let base = start_mock_server(vec![TokenStep::SlowDown, TokenStep::Success]).await;
        let cfg = test_config(&base);
        let start = std::time::Instant::now();
        let tokens = run_device_flow(&cfg, &reqwest::Client::new())
            .await
            .expect("device flow should succeed after slow_down bump");
        let elapsed = start.elapsed();
        assert_eq!(tokens.access_token, "at-abc");
        assert!(
            elapsed >= SLOW_DOWN_INCREMENT,
            "expected slow_down to add at least {:?} of wait, got {elapsed:?}",
            SLOW_DOWN_INCREMENT
        );
    }

    #[tokio::test]
    async fn expired_token_surfaces_as_error() {
        let base = start_mock_server(vec![TokenStep::Expired]).await;
        let cfg = test_config(&base);
        let err = run_device_flow(&cfg, &reqwest::Client::new())
            .await
            .expect_err("expired_token should error");
        let msg = format!("{err:#}");
        assert!(msg.contains("expired"), "got: {msg}");
    }

    #[tokio::test]
    async fn access_denied_surfaces_as_error() {
        let base = start_mock_server(vec![TokenStep::Denied]).await;
        let cfg = test_config(&base);
        let err = run_device_flow(&cfg, &reqwest::Client::new())
            .await
            .expect_err("access_denied should error");
        let msg = format!("{err:#}");
        assert!(msg.contains("denied"), "got: {msg}");
    }
}
