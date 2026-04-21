//! `mixer auth {login, logout, status}` implementations.
//!
//! `status` reports, per provider, which credential source resolved (an env
//! var, the stored file, or nothing) and — for device-flow providers — the
//! OAuth freshness state. Output shape follows plan.md §3.6.2.

use anyhow::Result;
use chrono::{TimeZone, Utc};
use serde_json::{Value, json};

use crate::config::{Config, ProviderSettings};
use crate::credentials::CredentialStore;
use crate::providers::common::oauth_refresh::{OauthFreshness, current_expiry, oauth_freshness};
use crate::providers::{AuthKind, Provider, builtin_registry};

pub async fn login(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;
    let provider = registry.get(provider_id)?;
    let settings = provider_settings(&config, provider_id);

    provider.login(&credentials).await?;

    // For API-key providers, a freshly stored key is shadowed by an env var
    // that's set at request time — warn so the user isn't surprised.
    if provider.auth_kind() == AuthKind::ApiKey
        && let Some(var) = settings.api_key_env.as_deref()
        && env_var_is_set(var)
    {
        eprintln!("warning: env var `{var}` is set; it will shadow the stored key at request time");
    }

    eprintln!("logged in to `{}`", provider.id());
    Ok(())
}

pub async fn logout(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;
    let provider = registry.get(provider_id)?;
    let settings = provider_settings(&config, provider_id);

    provider.logout(&credentials).await?;

    // Deleting the file doesn't revoke env-sourced credentials.
    if provider.auth_kind() == AuthKind::ApiKey
        && let Some(var) = settings.api_key_env.as_deref()
        && env_var_is_set(var)
    {
        eprintln!("warning: env var `{var}` is still set; provider remains authenticated via env");
    }

    eprintln!("logged out of `{}`", provider.id());
    Ok(())
}

pub async fn status(provider: Option<&str>, json: bool) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;
    let now = Utc::now().timestamp();

    let ids: Vec<String> = match provider {
        Some(id) => {
            // Validate ahead of time so an unknown provider errors instead of
            // silently reporting "missing".
            registry.get(id)?;
            vec![id.to_string()]
        }
        None => registry.ids().into_iter().map(|s| s.to_string()).collect(),
    };

    let entries: Vec<StatusEntry> = ids
        .iter()
        .map(|id| {
            let p = registry.get(id).expect("id comes from registry");
            let settings = provider_settings(&config, id);
            describe_status(p.as_ref(), &credentials, &settings, now)
        })
        .collect();

    if json {
        print_status_json(&entries);
    } else {
        print_status_text(&entries, now);
    }
    Ok(())
}

fn provider_settings(config: &Config, provider_id: &str) -> ProviderSettings {
    config
        .providers
        .get(provider_id)
        .cloned()
        .unwrap_or_default()
}

fn env_var_is_set(var: &str) -> bool {
    std::env::var(var).map(|v| !v.is_empty()).unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq)]
struct StatusEntry {
    provider_id: String,
    auth_kind: AuthKind,
    source: CredentialSource,
    authenticated: bool,
    /// Present only for device-flow providers with a usable stored blob.
    oauth: Option<OauthReport>,
}

#[derive(Debug, Clone, PartialEq)]
enum CredentialSource {
    Env {
        var: String,
    },
    File {
        path: String,
    },
    Missing,
    /// Provider doesn't authenticate — e.g. self-hosted ollama.
    NoAuth,
}

#[derive(Debug, Clone, PartialEq)]
struct OauthReport {
    expires_at: Option<i64>,
    state: OauthFreshness,
}

fn describe_status(
    provider: &dyn Provider,
    store: &CredentialStore,
    settings: &ProviderSettings,
    now: i64,
) -> StatusEntry {
    match provider.auth_kind() {
        AuthKind::ApiKey => describe_api_key(provider.id(), store, settings),
        AuthKind::DeviceFlow => describe_device_flow(provider.id(), store, now),
        AuthKind::None => StatusEntry {
            provider_id: provider.id().to_string(),
            auth_kind: AuthKind::None,
            source: CredentialSource::NoAuth,
            authenticated: true,
            oauth: None,
        },
    }
}

fn describe_api_key(
    provider_id: &str,
    store: &CredentialStore,
    settings: &ProviderSettings,
) -> StatusEntry {
    if let Some(var) = settings.api_key_env.as_deref()
        && env_var_is_set(var)
    {
        return StatusEntry {
            provider_id: provider_id.to_string(),
            auth_kind: AuthKind::ApiKey,
            source: CredentialSource::Env {
                var: var.to_string(),
            },
            authenticated: true,
            oauth: None,
        };
    }

    let has_file_key = store
        .load_blob(provider_id)
        .ok()
        .flatten()
        .and_then(|v| v.get("api_key").and_then(Value::as_str).map(str::to_string))
        .is_some_and(|s| !s.is_empty());

    if has_file_key {
        StatusEntry {
            provider_id: provider_id.to_string(),
            auth_kind: AuthKind::ApiKey,
            source: CredentialSource::File {
                path: file_display(provider_id),
            },
            authenticated: true,
            oauth: None,
        }
    } else {
        StatusEntry {
            provider_id: provider_id.to_string(),
            auth_kind: AuthKind::ApiKey,
            source: CredentialSource::Missing,
            authenticated: false,
            oauth: None,
        }
    }
}

fn describe_device_flow(provider_id: &str, store: &CredentialStore, now: i64) -> StatusEntry {
    let Ok(Some(blob)) = store.load_blob(provider_id) else {
        return StatusEntry {
            provider_id: provider_id.to_string(),
            auth_kind: AuthKind::DeviceFlow,
            source: CredentialSource::Missing,
            authenticated: false,
            oauth: None,
        };
    };
    let has_access = blob
        .get("access_token")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !has_access {
        // A file exists but has no usable access token — treat as missing.
        return StatusEntry {
            provider_id: provider_id.to_string(),
            auth_kind: AuthKind::DeviceFlow,
            source: CredentialSource::Missing,
            authenticated: false,
            oauth: None,
        };
    }

    let state = oauth_freshness(&blob, now);
    let authenticated = !matches!(state, OauthFreshness::ExpiredDead);
    let expires_at = current_expiry(&blob);

    StatusEntry {
        provider_id: provider_id.to_string(),
        auth_kind: AuthKind::DeviceFlow,
        source: CredentialSource::File {
            path: file_display(provider_id),
        },
        authenticated,
        oauth: Some(OauthReport { expires_at, state }),
    }
}

fn file_display(provider_id: &str) -> String {
    format!("credentials/{provider_id}.json")
}

fn source_label(source: &CredentialSource) -> String {
    match source {
        CredentialSource::Env { var } => format!("env:{var}"),
        CredentialSource::File { path } => format!("file:{path}"),
        CredentialSource::Missing => "missing".to_string(),
        CredentialSource::NoAuth => "none (no auth required)".to_string(),
    }
}

fn oauth_suffix(oauth: &OauthReport, now: i64) -> String {
    match oauth.state {
        OauthFreshness::ExpiredRefreshable => "(oauth, expired; refreshable)".to_string(),
        OauthFreshness::ExpiredDead => "(oauth, expired; re-login required)".to_string(),
        OauthFreshness::Valid => match oauth.expires_at {
            Some(exp) => {
                let remaining = exp.saturating_sub(now).max(0) as u64;
                format!("(oauth, expires in {})", format_duration(remaining))
            }
            None => "(oauth)".to_string(),
        },
    }
}

/// Render a positive seconds count as a coarse-grained duration: `12d`, `3h`,
/// `45m`, `30s`. Matches the `(oauth, expires in 12d)` style in plan.md §3.6.2.
fn format_duration(secs: u64) -> String {
    const MIN: u64 = 60;
    const HOUR: u64 = 60 * MIN;
    const DAY: u64 = 24 * HOUR;
    if secs >= DAY {
        format!("{}d", secs / DAY)
    } else if secs >= HOUR {
        format!("{}h", secs / HOUR)
    } else if secs >= MIN {
        format!("{}m", secs / MIN)
    } else {
        format!("{secs}s")
    }
}

fn render_status_text(entries: &[StatusEntry], now: i64) -> String {
    let mut out = String::new();
    for entry in entries {
        let mut line = format!("{:10} {}", entry.provider_id, source_label(&entry.source));
        if let Some(oauth) = &entry.oauth {
            line.push(' ');
            line.push_str(&oauth_suffix(oauth, now));
        } else if matches!(entry.source, CredentialSource::File { .. })
            && entry.auth_kind == AuthKind::ApiKey
        {
            line.push(' ');
            line.push_str("(api_key)");
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn build_status_json(entries: &[StatusEntry]) -> Value {
    let mut map = serde_json::Map::new();
    for entry in entries {
        let mut obj = serde_json::Map::new();
        obj.insert("authenticated".to_string(), json!(entry.authenticated));
        match &entry.source {
            CredentialSource::Env { var } => {
                obj.insert("source".to_string(), json!("env"));
                obj.insert("var".to_string(), json!(var));
            }
            CredentialSource::File { path } => {
                obj.insert("source".to_string(), json!("file"));
                obj.insert("path".to_string(), json!(path));
            }
            CredentialSource::Missing => {
                obj.insert("source".to_string(), json!("missing"));
            }
            CredentialSource::NoAuth => {
                obj.insert("source".to_string(), json!("none"));
            }
        }
        if let Some(oauth) = &entry.oauth {
            if let Some(exp) = oauth.expires_at
                && let Some(ts) = Utc.timestamp_opt(exp, 0).single()
            {
                obj.insert(
                    "oauth_expires_at".to_string(),
                    json!(ts.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
                );
            }
            obj.insert(
                "oauth_state".to_string(),
                json!(oauth_state_label(oauth.state)),
            );
        }
        map.insert(entry.provider_id.clone(), Value::Object(obj));
    }
    Value::Object(map)
}

fn print_status_text(entries: &[StatusEntry], now: i64) {
    print!("{}", render_status_text(entries, now));
}

fn print_status_json(entries: &[StatusEntry]) {
    println!(
        "{}",
        serde_json::to_string_pretty(&build_status_json(entries)).unwrap()
    );
}

fn oauth_state_label(state: OauthFreshness) -> &'static str {
    match state {
        OauthFreshness::Valid => "valid",
        OauthFreshness::ExpiredRefreshable => "expired-refreshable",
        OauthFreshness::ExpiredDead => "expired-dead",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn store_in(dir: &TempDir) -> CredentialStore {
        CredentialStore::with_dir_for_tests(dir.path().to_path_buf())
    }

    fn settings_with_env(var: Option<&str>) -> ProviderSettings {
        ProviderSettings {
            api_key_env: var.map(str::to_string),
            ..ProviderSettings::default_enabled()
        }
    }

    #[test]
    fn api_key_env_source_when_var_set() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let var = "MIXER_TEST_STATUS_ENV_API_KEY";
        unsafe { std::env::set_var(var, "k") };
        let entry = describe_api_key("minimax", &store, &settings_with_env(Some(var)));
        unsafe { std::env::remove_var(var) };

        assert_eq!(
            entry.source,
            CredentialSource::Env {
                var: var.to_string()
            }
        );
        assert!(entry.authenticated);
        assert_eq!(source_label(&entry.source), format!("env:{var}"));
    }

    #[test]
    fn api_key_file_source_when_env_unset_and_file_has_key() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("minimax", &json!({ "api_key": "k" })).unwrap();

        let entry = describe_api_key("minimax", &store, &settings_with_env(None));
        assert_eq!(
            entry.source,
            CredentialSource::File {
                path: "credentials/minimax.json".to_string()
            }
        );
        assert!(entry.authenticated);
        assert_eq!(source_label(&entry.source), "file:credentials/minimax.json");
    }

    #[test]
    fn api_key_missing_when_neither_source_has_key() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let entry = describe_api_key("minimax", &store, &settings_with_env(None));
        assert_eq!(entry.source, CredentialSource::Missing);
        assert!(!entry.authenticated);
    }

    #[test]
    fn api_key_empty_stored_key_counts_as_missing() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("minimax", &json!({ "api_key": "" })).unwrap();
        let entry = describe_api_key("minimax", &store, &settings_with_env(None));
        assert_eq!(entry.source, CredentialSource::Missing);
        assert!(!entry.authenticated);
    }

    #[test]
    fn api_key_empty_env_falls_back_to_file() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        store.save("minimax", &json!({ "api_key": "k" })).unwrap();

        let var = "MIXER_TEST_STATUS_EMPTY_ENV";
        unsafe { std::env::set_var(var, "") };
        let entry = describe_api_key("minimax", &store, &settings_with_env(Some(var)));
        unsafe { std::env::remove_var(var) };

        assert!(matches!(entry.source, CredentialSource::File { .. }));
        assert!(entry.authenticated);
    }

    #[test]
    fn device_flow_missing_when_file_absent() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let entry = describe_device_flow("codex", &store, 1_000);
        assert_eq!(entry.source, CredentialSource::Missing);
        assert!(!entry.authenticated);
        assert!(entry.oauth.is_none());
    }

    #[test]
    fn device_flow_valid_reports_expires_in() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let now = 1_000_000_i64;
        let exp = now + 12 * 24 * 60 * 60; // 12 days
        store
            .save(
                "codex",
                &json!({
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_at": exp,
                }),
            )
            .unwrap();

        let entry = describe_device_flow("codex", &store, now);
        assert!(entry.authenticated);
        let oauth = entry.oauth.as_ref().expect("oauth details");
        assert_eq!(oauth.state, OauthFreshness::Valid);
        assert_eq!(oauth.expires_at, Some(exp));
        assert_eq!(oauth_suffix(oauth, now), "(oauth, expires in 12d)");
    }

    #[test]
    fn device_flow_expired_refreshable_reports_refreshable() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let now = 1_000_000_i64;
        store
            .save(
                "codex",
                &json!({
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_at": now - 60,
                }),
            )
            .unwrap();

        let entry = describe_device_flow("codex", &store, now);
        assert!(entry.authenticated, "refreshable counts as authenticated");
        let oauth = entry.oauth.as_ref().expect("oauth details");
        assert_eq!(oauth.state, OauthFreshness::ExpiredRefreshable);
        assert_eq!(oauth_suffix(oauth, now), "(oauth, expired; refreshable)");
    }

    #[test]
    fn device_flow_expired_dead_reports_re_login() {
        let tmp = TempDir::new().unwrap();
        let store = store_in(&tmp);
        let now = 1_000_000_i64;
        store
            .save(
                "codex",
                &json!({
                    "access_token": "at",
                    "expires_at": now - 60,
                }),
            )
            .unwrap();

        let entry = describe_device_flow("codex", &store, now);
        assert!(
            !entry.authenticated,
            "expired-dead should not count as authenticated"
        );
        let oauth = entry.oauth.as_ref().expect("oauth details");
        assert_eq!(oauth.state, OauthFreshness::ExpiredDead);
        assert_eq!(
            oauth_suffix(oauth, now),
            "(oauth, expired; re-login required)",
        );
    }

    #[test]
    fn format_duration_uses_coarsest_unit() {
        assert_eq!(format_duration(0), "0s");
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(120), "2m");
        assert_eq!(format_duration(7_200), "2h");
        assert_eq!(format_duration(12 * 24 * 3_600), "12d");
    }

    #[test]
    fn json_output_includes_env_var_for_env_source() {
        let entry = StatusEntry {
            provider_id: "minimax".to_string(),
            auth_kind: AuthKind::ApiKey,
            source: CredentialSource::Env {
                var: "MINIMAX_API_KEY".to_string(),
            },
            authenticated: true,
            oauth: None,
        };
        let v = build_status_json(&[entry]);
        assert_eq!(v["minimax"]["source"], "env");
        assert_eq!(v["minimax"]["var"], "MINIMAX_API_KEY");
        assert_eq!(v["minimax"]["authenticated"], true);
        assert!(v["minimax"].get("path").is_none());
        assert!(v["minimax"].get("oauth_state").is_none());
    }

    #[test]
    fn json_output_for_missing_has_only_source_and_authenticated() {
        let entry = StatusEntry {
            provider_id: "opencode".to_string(),
            auth_kind: AuthKind::ApiKey,
            source: CredentialSource::Missing,
            authenticated: false,
            oauth: None,
        };
        let v = build_status_json(&[entry]);
        assert_eq!(v["opencode"]["source"], "missing");
        assert_eq!(v["opencode"]["authenticated"], false);
        assert!(v["opencode"].get("var").is_none());
        assert!(v["opencode"].get("path").is_none());
    }

    #[test]
    fn json_output_for_device_flow_includes_oauth_fields() {
        let entry = StatusEntry {
            provider_id: "codex".to_string(),
            auth_kind: AuthKind::DeviceFlow,
            source: CredentialSource::File {
                path: "credentials/codex.json".to_string(),
            },
            authenticated: true,
            oauth: Some(OauthReport {
                expires_at: Some(1_700_000_000),
                state: OauthFreshness::Valid,
            }),
        };
        let v = build_status_json(&[entry]);
        assert_eq!(v["codex"]["source"], "file");
        assert_eq!(v["codex"]["path"], "credentials/codex.json");
        assert_eq!(v["codex"]["authenticated"], true);
        assert_eq!(v["codex"]["oauth_state"], "valid");
        // RFC3339 shape with trailing Z.
        let exp = v["codex"]["oauth_expires_at"].as_str().unwrap();
        assert!(exp.ends_with('Z'), "expected RFC3339 UTC, got {exp}");
    }

    #[test]
    fn text_output_matches_plan_examples() {
        let now = 1_000_000_i64;
        let entries = vec![
            StatusEntry {
                provider_id: "codex".to_string(),
                auth_kind: AuthKind::DeviceFlow,
                source: CredentialSource::File {
                    path: "credentials/codex.json".to_string(),
                },
                authenticated: true,
                oauth: Some(OauthReport {
                    expires_at: Some(now + 12 * 24 * 60 * 60),
                    state: OauthFreshness::Valid,
                }),
            },
            StatusEntry {
                provider_id: "minimax".to_string(),
                auth_kind: AuthKind::ApiKey,
                source: CredentialSource::Env {
                    var: "MINIMAX_API_KEY".to_string(),
                },
                authenticated: true,
                oauth: None,
            },
            StatusEntry {
                provider_id: "glm".to_string(),
                auth_kind: AuthKind::ApiKey,
                source: CredentialSource::File {
                    path: "credentials/glm.json".to_string(),
                },
                authenticated: true,
                oauth: None,
            },
            StatusEntry {
                provider_id: "opencode".to_string(),
                auth_kind: AuthKind::ApiKey,
                source: CredentialSource::Missing,
                authenticated: false,
                oauth: None,
            },
        ];
        let rendered = render_status_text(&entries, now);
        assert!(
            rendered.contains("file:credentials/codex.json (oauth, expires in 12d)"),
            "codex line missing: {rendered}"
        );
        assert!(
            rendered.contains("env:MINIMAX_API_KEY"),
            "minimax line missing: {rendered}"
        );
        assert!(
            rendered.contains("file:credentials/glm.json (api_key)"),
            "glm line missing: {rendered}"
        );
        assert!(
            rendered
                .lines()
                .any(|l| l.contains("opencode") && l.contains("missing")),
            "opencode missing line absent: {rendered}"
        );
    }

    #[test]
    fn no_auth_provider_reports_authenticated_with_none_source() {
        // Ollama-shaped: AuthKind::None should report authenticated=true with
        // `source: none` in JSON, matching `providers list --json` which sees
        // the provider as always authenticated.
        let entry = StatusEntry {
            provider_id: "ollama".to_string(),
            auth_kind: AuthKind::None,
            source: CredentialSource::NoAuth,
            authenticated: true,
            oauth: None,
        };
        let v = build_status_json(std::slice::from_ref(&entry));
        assert_eq!(v["ollama"]["authenticated"], true);
        assert_eq!(v["ollama"]["source"], "none");
        assert!(v["ollama"].get("path").is_none());
        assert!(v["ollama"].get("var").is_none());

        let rendered = render_status_text(std::slice::from_ref(&entry), 0);
        assert!(
            rendered.contains("ollama") && rendered.contains("none"),
            "ollama line should state no-auth: {rendered}",
        );
    }
}
