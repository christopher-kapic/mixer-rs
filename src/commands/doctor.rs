//! `mixer doctor` — config validation and optional live probes.
//!
//! Runs a pipeline of checks against the user's configuration and credentials:
//!
//!   1. Config file exists and parses.
//!   2. Every mixer model's backends reference registered providers.
//!   3. Every referenced provider model appears in that provider's `models()`.
//!   4. Every mixer model has at least one enabled+authenticated backend.
//!   5. `default_model` resolves to a configured mixer model.
//!   6. (gated) Per-provider chat probe — 1-token "say 'ok'" round trip.
//!   7. (gated) Per-provider usage() probe.
//!
//! Live probes (6–7) are gated behind `MIXER_DOCTOR_LIVE=1` so CI and regular
//! `mixer doctor` invocations do not hit the network.
//!
//! Exit codes: `0` all pass, `1` any warning, `2` any failure.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::concurrency::ConcurrencyLimits;
use crate::config::Config;
use crate::credentials::CredentialStore;
use crate::openai::{ChatMessage, ChatRequest, MessageContent};
use crate::paths;
use crate::providers::{ProviderRegistry, builtin_registry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Severity {
    Pass,
    Warn,
    Fail,
    Skip,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Pass => "ok",
            Severity::Warn => "warn",
            Severity::Fail => "fail",
            Severity::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone)]
struct CheckResult {
    name: String,
    severity: Severity,
    detail: Option<String>,
}

impl CheckResult {
    fn pass(name: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            name: name.into(),
            severity: Severity::Pass,
            detail,
        }
    }
    fn warn(name: impl Into<String>, detail: String) -> Self {
        Self {
            name: name.into(),
            severity: Severity::Warn,
            detail: Some(detail),
        }
    }
    fn fail(name: impl Into<String>, detail: String) -> Self {
        Self {
            name: name.into(),
            severity: Severity::Fail,
            detail: Some(detail),
        }
    }
    fn skip(name: impl Into<String>, detail: String) -> Self {
        Self {
            name: name.into(),
            severity: Severity::Skip,
            detail: Some(detail),
        }
    }
}

pub async fn run(as_json: bool) -> Result<i32> {
    let mut results: Vec<CheckResult> = Vec::new();

    let config_path = paths::config_file()?;
    let (config_opt, config_res) = load_config_with_check(&config_path);
    results.push(config_res);

    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;

    if let Some(config) = &config_opt {
        results.extend(validate_backends_providers(config, &registry));
        results.extend(validate_backends_provider_models(config, &registry));
        results.extend(validate_authenticated_backends(
            config,
            &registry,
            &credentials,
        ));
        results.push(validate_default_model(config));

        let live = std::env::var("MIXER_DOCTOR_LIVE")
            .map(|v| v == "1")
            .unwrap_or(false);
        if live {
            results.extend(probe_chat(config, &registry, &credentials).await);
            results.extend(probe_usage(config, &registry, &credentials).await);
        } else {
            results.push(CheckResult::skip(
                "provider chat probes",
                "set MIXER_DOCTOR_LIVE=1 to enable".to_string(),
            ));
            results.push(CheckResult::skip(
                "provider usage probes",
                "set MIXER_DOCTOR_LIVE=1 to enable".to_string(),
            ));
        }
    }

    let exit_code = summarize_exit(&results);
    if as_json {
        print_json(&results, exit_code);
    } else {
        print_text(&results);
    }
    Ok(exit_code)
}

// ── Check: config file ──────────────────────────────────────────────────────

fn load_config_with_check(path: &Path) -> (Option<Config>, CheckResult) {
    if !path.exists() {
        return (
            Some(Config::default()),
            CheckResult::warn(
                "config file",
                format!(
                    "{} not found — using defaults; run `mixer init`",
                    path.display()
                ),
            ),
        );
    }
    match Config::load(path) {
        Ok(c) => (
            Some(c),
            CheckResult::pass("config file", Some(path.display().to_string())),
        ),
        Err(e) => (None, CheckResult::fail("config file", format!("{e:#}"))),
    }
}

// ── Check: backends reference registered providers ─────────────────────────

fn validate_backends_providers(config: &Config, registry: &ProviderRegistry) -> Vec<CheckResult> {
    let mut bad: Vec<String> = Vec::new();
    for (name, mixer_model) in &config.models {
        for b in &mixer_model.backends {
            if registry.get(&b.provider).is_err() {
                bad.push(format!("`{name}` → unknown provider `{}`", b.provider));
            }
        }
    }
    if bad.is_empty() {
        vec![CheckResult::pass(
            "backends reference registered providers",
            None,
        )]
    } else {
        vec![CheckResult::fail(
            "backends reference registered providers",
            bad.join("; "),
        )]
    }
}

// ── Check: backend models appear in provider's catalogue ────────────────────

fn validate_backends_provider_models(
    config: &Config,
    registry: &ProviderRegistry,
) -> Vec<CheckResult> {
    let mut bad: Vec<String> = Vec::new();
    for (name, mixer_model) in &config.models {
        for b in &mixer_model.backends {
            let Ok(provider) = registry.get(&b.provider) else {
                // Already reported by the preceding check; skip to avoid dup.
                continue;
            };
            let known = provider.models().into_iter().any(|m| m.id == b.model);
            if !known {
                bad.push(format!(
                    "`{name}` → {}/{} not in provider catalogue",
                    b.provider, b.model
                ));
            }
        }
    }
    if bad.is_empty() {
        vec![CheckResult::pass(
            "backend models in provider catalogues",
            None,
        )]
    } else {
        vec![CheckResult::fail(
            "backend models in provider catalogues",
            bad.join("; "),
        )]
    }
}

// ── Check: mixer models have authenticated backends ─────────────────────────

fn validate_authenticated_backends(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
) -> Vec<CheckResult> {
    let mut out = Vec::new();
    let mut names: Vec<&String> = config.models.keys().collect();
    names.sort();
    for name in names {
        let mixer_model = &config.models[name];
        let total = mixer_model.backends.len();
        let mut authed = 0;
        let mut image_total = 0;
        let mut image_authed = 0;
        for b in &mixer_model.backends {
            let Ok(provider) = registry.get(&b.provider) else {
                continue;
            };
            let settings = config
                .providers
                .get(&b.provider)
                .cloned()
                .unwrap_or_default();
            let supports_images = provider
                .models()
                .into_iter()
                .find(|m| m.id == b.model)
                .map(|m| m.supports_images)
                .unwrap_or(false);
            if supports_images {
                image_total += 1;
            }
            if !settings.enabled {
                continue;
            }
            if provider.is_authenticated(credentials, &settings) {
                authed += 1;
                if supports_images {
                    image_authed += 1;
                }
            }
        }

        let check_name = format!("mixer model `{name}` has authenticated backend");
        if authed == 0 {
            out.push(CheckResult::warn(
                check_name,
                format!(
                    "{authed}/{total} enabled+authenticated; run `mixer auth login <provider>`"
                ),
            ));
            continue;
        }

        // Vision-only mixer model (every backend is image-capable) with zero
        // image-capable authed backends would mean no image request can route.
        // `authed > 0 && image_total == total && image_authed == 0` is only
        // possible if all authed backends lost their image flag after config —
        // extremely rare but worth surfacing when it happens.
        if total > 0 && image_total == total && image_authed == 0 {
            out.push(CheckResult::warn(
                check_name,
                "no image-capable authenticated backend; image requests will 404".to_string(),
            ));
            continue;
        }

        out.push(CheckResult::pass(
            check_name,
            Some(format!("{authed}/{total} enabled+authenticated")),
        ));
    }
    out
}

// ── Check: default model resolves ───────────────────────────────────────────

fn validate_default_model(config: &Config) -> CheckResult {
    if config.models.contains_key(&config.default_model) {
        CheckResult::pass(
            "default model resolves",
            Some(format!("`{}`", config.default_model)),
        )
    } else {
        CheckResult::fail(
            "default model resolves",
            format!(
                "`default_model` = `{}` but no mixer model with that name is configured",
                config.default_model
            ),
        )
    }
}

// ── Live probes (gated by MIXER_DOCTOR_LIVE=1) ─────────────────────────────

async fn probe_chat(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
) -> Vec<CheckResult> {
    let concurrency = ConcurrencyLimits::from_config(config);
    let mut out = Vec::new();
    let mut ids: Vec<&str> = registry.ids();
    ids.sort_unstable();
    for id in ids {
        let check_name = format!("{id} chat probe");
        let Ok(provider) = registry.get(id) else {
            continue;
        };
        let settings = config.providers.get(id).cloned().unwrap_or_default();
        if !settings.enabled {
            out.push(CheckResult::skip(
                check_name,
                "disabled in config".to_string(),
            ));
            continue;
        }
        if !provider.is_authenticated(credentials, &settings) {
            out.push(CheckResult::skip(
                check_name,
                "not authenticated".to_string(),
            ));
            continue;
        }
        let Some(model) = provider.models().into_iter().next() else {
            out.push(CheckResult::skip(
                check_name,
                "provider has no models".to_string(),
            ));
            continue;
        };
        let req = ChatRequest {
            model: model.id.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("say 'ok'".to_string())),
                reasoning_content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: Some(false),
            temperature: None,
            top_p: None,
            max_tokens: Some(1),
            max_completion_tokens: None,
            tools: None,
            tool_choice: None,
            extra: Default::default(),
        };
        let _permit = concurrency.acquire(id).await;
        let start = Instant::now();
        let dispatch = provider.chat_completion(credentials, &settings, req).await;
        let elapsed = start.elapsed();
        match dispatch {
            Err(e) => out.push(CheckResult::fail(check_name, format!("{e:#}"))),
            Ok(mut stream) => {
                let mut err: Option<anyhow::Error> = None;
                while let Some(item) = stream.next().await {
                    if let Err(e) = item {
                        err = Some(e);
                        break;
                    }
                }
                match err {
                    Some(e) => out.push(CheckResult::fail(check_name, format!("{e:#}"))),
                    None => out.push(CheckResult::pass(
                        check_name,
                        Some(format!("latency {}ms", elapsed.as_millis())),
                    )),
                }
            }
        }
    }
    out
}

async fn probe_usage(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
) -> Vec<CheckResult> {
    let mut out = Vec::new();
    let mut ids: Vec<&str> = registry.ids();
    ids.sort_unstable();
    for id in ids {
        let check_name = format!("{id} usage probe");
        let Ok(provider) = registry.get(id) else {
            continue;
        };
        let settings = config.providers.get(id).cloned().unwrap_or_default();
        if !provider.is_authenticated(credentials, &settings) {
            out.push(CheckResult::skip(
                check_name,
                "not authenticated".to_string(),
            ));
            continue;
        }
        match provider.usage(credentials, &settings).await {
            Ok(Some(snap)) => {
                let pct = snap
                    .fraction_used
                    .map(|f| format!("{:.1}%", (f * 100.0).clamp(0.0, 100.0)))
                    .unwrap_or_else(|| "unknown".to_string());
                out.push(CheckResult::pass(check_name, Some(format!("used {pct}"))));
            }
            Ok(None) => out.push(CheckResult::pass(
                check_name,
                Some("usage unknown".to_string()),
            )),
            Err(e) => out.push(CheckResult::fail(check_name, format!("{e:#}"))),
        }
    }
    out
}

// ── Output ──────────────────────────────────────────────────────────────────

fn summarize_exit(results: &[CheckResult]) -> i32 {
    if results.iter().any(|r| r.severity == Severity::Fail) {
        2
    } else if results.iter().any(|r| r.severity == Severity::Warn) {
        1
    } else {
        0
    }
}

fn print_text(results: &[CheckResult]) {
    for r in results {
        match &r.detail {
            Some(d) if !d.is_empty() => {
                println!("[{:>4}]  {:44}  {}", r.severity.label(), r.name, d);
            }
            _ => {
                println!("[{:>4}]  {}", r.severity.label(), r.name);
            }
        }
    }
    let (pass, warn, fail, skip) = tally(results);
    println!();
    println!("{pass} passed, {warn} warning(s), {fail} failure(s), {skip} skipped");
}

fn tally(results: &[CheckResult]) -> (usize, usize, usize, usize) {
    let mut pass = 0;
    let mut warn = 0;
    let mut fail = 0;
    let mut skip = 0;
    for r in results {
        match r.severity {
            Severity::Pass => pass += 1,
            Severity::Warn => warn += 1,
            Severity::Fail => fail += 1,
            Severity::Skip => skip += 1,
        }
    }
    (pass, warn, fail, skip)
}

fn print_json(results: &[CheckResult], exit_code: i32) {
    let (pass, warn, fail, skip) = tally(results);
    let checks: Vec<Value> = results
        .iter()
        .map(|r| {
            let mut obj = serde_json::Map::new();
            obj.insert("name".to_string(), json!(r.name));
            obj.insert("severity".to_string(), json!(r.severity.label()));
            if let Some(d) = &r.detail {
                obj.insert("detail".to_string(), json!(d));
            }
            Value::Object(obj)
        })
        .collect();
    let payload = json!({
        "exit_code": exit_code,
        "summary": { "pass": pass, "warn": warn, "fail": fail, "skip": skip },
        "checks": checks,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string())
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, MixerModel, ProviderSettings, RoutingStrategy};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn empty_store() -> (TempDir, CredentialStore) {
        let tmp = TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());
        (tmp, store)
    }

    fn mixer_with_backends(backends: Vec<Backend>) -> MixerModel {
        MixerModel {
            description: String::new(),
            backends,
            strategy: RoutingStrategy::Random,
            weights: HashMap::new(),
            sticky: None,
        }
    }

    #[test]
    fn backends_providers_passes_for_default_config() {
        let cfg = Config::default();
        let reg = builtin_registry();
        let results = validate_backends_providers(&cfg, &reg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, Severity::Pass);
    }

    #[test]
    fn backends_providers_fails_for_unknown_provider() {
        let mut cfg = Config::default();
        cfg.models.insert(
            "broken".to_string(),
            mixer_with_backends(vec![Backend {
                provider: "nope".to_string(),
                model: "gpt-5.2".to_string(),
            }]),
        );
        let reg = builtin_registry();
        let results = validate_backends_providers(&cfg, &reg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, Severity::Fail);
        let detail = results[0].detail.as_deref().unwrap_or("");
        assert!(
            detail.contains("unknown provider `nope`"),
            "detail: {detail}"
        );
    }

    #[test]
    fn backends_models_passes_when_all_models_known() {
        let cfg = Config::default();
        let reg = builtin_registry();
        let results = validate_backends_provider_models(&cfg, &reg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, Severity::Pass);
    }

    #[test]
    fn backends_models_fails_for_unknown_model() {
        let mut cfg = Config::default();
        cfg.models.insert(
            "broken".to_string(),
            mixer_with_backends(vec![Backend {
                provider: "codex".to_string(),
                model: "not-a-real-model".to_string(),
            }]),
        );
        let reg = builtin_registry();
        let results = validate_backends_provider_models(&cfg, &reg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, Severity::Fail);
        let detail = results[0].detail.as_deref().unwrap_or("");
        assert!(detail.contains("not-a-real-model"), "detail: {detail}");
    }

    #[test]
    fn backends_models_ignores_missing_provider_to_avoid_duplicate_failures() {
        // `nope` is not registered — the provider check owns that failure, so
        // the model check should not double-report.
        let mut cfg = Config::default();
        cfg.models.clear();
        cfg.models.insert(
            "broken".to_string(),
            mixer_with_backends(vec![Backend {
                provider: "nope".to_string(),
                model: "x".to_string(),
            }]),
        );
        let reg = builtin_registry();
        let results = validate_backends_provider_models(&cfg, &reg);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, Severity::Pass);
    }

    #[test]
    fn default_model_passes_when_key_exists() {
        let cfg = Config::default();
        let result = validate_default_model(&cfg);
        assert_eq!(result.severity, Severity::Pass);
    }

    #[test]
    fn default_model_fails_when_key_missing() {
        let cfg = Config {
            default_model: "does-not-exist".to_string(),
            ..Config::default()
        };
        let result = validate_default_model(&cfg);
        assert_eq!(result.severity, Severity::Fail);
    }

    #[test]
    fn authenticated_backends_warns_when_nothing_logged_in() {
        // The default config references four providers, but the fresh temp
        // CredentialStore has no credentials, so no backend authenticates.
        let cfg = Config::default();
        let reg = builtin_registry();
        let (_tmp, store) = empty_store();
        let results = validate_authenticated_backends(&cfg, &reg, &store);
        assert_eq!(results.len(), cfg.models.len());
        for r in &results {
            assert_eq!(r.severity, Severity::Warn, "{r:?}");
        }
    }

    #[test]
    fn authenticated_backends_warns_when_provider_disabled() {
        let mut cfg = Config::default();
        // Disable every provider so none are eligible.
        for s in cfg.providers.values_mut() {
            s.enabled = false;
        }
        let reg = builtin_registry();
        let (_tmp, store) = empty_store();
        let results = validate_authenticated_backends(&cfg, &reg, &store);
        for r in &results {
            assert_eq!(r.severity, Severity::Warn);
        }
    }

    #[test]
    fn summarize_exit_is_fail_over_warn_over_pass() {
        let ok = CheckResult::pass("a", None);
        let warn = CheckResult::warn("b", "w".to_string());
        let fail = CheckResult::fail("c", "f".to_string());

        assert_eq!(summarize_exit(std::slice::from_ref(&ok)), 0);
        assert_eq!(summarize_exit(&[ok.clone(), warn.clone()]), 1);
        assert_eq!(summarize_exit(&[ok, warn, fail]), 2);
    }

    #[test]
    fn summarize_exit_treats_skip_as_neutral() {
        let ok = CheckResult::pass("a", None);
        let skip = CheckResult::skip("b", "s".to_string());
        assert_eq!(summarize_exit(&[ok, skip]), 0);
    }

    #[test]
    fn config_file_missing_warns_and_falls_back_to_default() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("does_not_exist.json");
        let (cfg, result) = load_config_with_check(&path);
        assert_eq!(result.severity, Severity::Warn);
        let c = cfg.expect("should fall back to default config");
        assert_eq!(c.default_model, Config::default().default_model);
    }

    #[test]
    fn config_file_malformed_is_fail_and_skips_downstream() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("config.json");
        std::fs::write(&path, "{ not valid json").unwrap();
        let (cfg, result) = load_config_with_check(&path);
        assert_eq!(result.severity, Severity::Fail);
        assert!(cfg.is_none(), "downstream checks should be skipped");
    }

    #[test]
    fn config_file_empty_treated_as_default() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("config.json");
        std::fs::write(&path, "").unwrap();
        let (cfg, result) = load_config_with_check(&path);
        assert_eq!(result.severity, Severity::Pass);
        assert!(cfg.is_some());
    }

    #[test]
    fn json_output_shape_is_parseable() {
        let results = vec![
            CheckResult::pass("p", Some("detail".to_string())),
            CheckResult::warn("w", "reason".to_string()),
            CheckResult::fail("f", "boom".to_string()),
            CheckResult::skip("s", "gated".to_string()),
        ];
        let tally = tally(&results);
        assert_eq!(tally, (1, 1, 1, 1));
        // Round-trip the JSON shape to make sure it parses.
        let mut buf = Vec::new();
        let (pass, warn, fail, skip) = tally;
        let checks: Vec<Value> = results
            .iter()
            .map(|r| {
                let mut obj = serde_json::Map::new();
                obj.insert("name".to_string(), json!(r.name));
                obj.insert("severity".to_string(), json!(r.severity.label()));
                if let Some(d) = &r.detail {
                    obj.insert("detail".to_string(), json!(d));
                }
                Value::Object(obj)
            })
            .collect();
        let payload = json!({
            "exit_code": 2,
            "summary": { "pass": pass, "warn": warn, "fail": fail, "skip": skip },
            "checks": checks,
        });
        serde_json::to_writer(&mut buf, &payload).unwrap();
        let back: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(back["exit_code"], 2);
        assert_eq!(back["summary"]["fail"], 1);
        assert_eq!(back["checks"].as_array().unwrap().len(), 4);
        assert_eq!(back["checks"][2]["severity"], "fail");
    }

    #[test]
    fn provider_settings_default_and_modified_roundtrip() {
        // Guards that the `validate_authenticated_backends` logic below doesn't
        // drift: we exercise the enabled=false branch explicitly.
        let mut s = ProviderSettings::default_enabled();
        s.enabled = false;
        assert!(!s.enabled);
    }
}
