use anyhow::Result;
use serde_json::json;

use crate::cli::ProvidersCommand;
use crate::config::Config;
use crate::credentials::CredentialStore;
use crate::providers::builtin_registry;

pub async fn run(cmd: &ProvidersCommand) -> Result<()> {
    match cmd {
        ProvidersCommand::List { verbose, json } => list(*verbose, *json).await,
        ProvidersCommand::Show { name, json } => show(name, *json).await,
        ProvidersCommand::Models { name, json } => models(name, *json).await,
    }
}

async fn list(verbose: bool, as_json: bool) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;

    let mut rows = Vec::new();
    for id in registry.ids() {
        let p = registry.get(id)?;
        let settings = config.providers.get(id).cloned().unwrap_or_default();
        let authed = p.is_authenticated(&credentials, &settings);
        let enabled = settings.enabled;
        let max_conc = settings.max_concurrent_requests;
        rows.push(json!({
            "id": p.id(),
            "display_name": p.display_name(),
            "authenticated": authed,
            "enabled": enabled,
            "max_concurrent_requests": max_conc,
            "models": p.models().into_iter().map(|m| json!({
                "id": m.id,
                "display_name": m.display_name,
                "supports_images": m.supports_images,
            })).collect::<Vec<_>>(),
        }));
    }

    if as_json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    for row in rows {
        let id = row["id"].as_str().unwrap_or("");
        let name = row["display_name"].as_str().unwrap_or("");
        let authed = row["authenticated"].as_bool().unwrap_or(false);
        let enabled = row["enabled"].as_bool().unwrap_or(true);
        let status = match (enabled, authed) {
            (false, _) => "disabled",
            (true, false) => "not logged in",
            (true, true) => "ready",
        };
        println!("{id:12} {status:14} {name}");
        if verbose {
            if let Some(n) = row["max_concurrent_requests"].as_u64() {
                println!("  max_concurrent_requests: {n}");
            }
            if let Some(models) = row["models"].as_array() {
                for m in models {
                    let mid = m["id"].as_str().unwrap_or("");
                    let img = m["supports_images"].as_bool().unwrap_or(false);
                    println!("  - {mid}{}", if img { " (vision)" } else { "" });
                }
            }
        }
    }

    Ok(())
}

async fn show(name: &str, as_json: bool) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;
    let p = registry.get(name)?;

    let settings = config.providers.get(name).cloned().unwrap_or_default();
    let usage = p.usage(&credentials, &settings).await.unwrap_or(None);

    let payload = json!({
        "id": p.id(),
        "display_name": p.display_name(),
        "authenticated": p.is_authenticated(&credentials, &settings),
        "settings": settings,
        "usage": usage,
        "models": p.models().into_iter().map(|m| json!({
            "id": m.id,
            "display_name": m.display_name,
            "supports_images": m.supports_images,
        })).collect::<Vec<_>>(),
    });

    // Pretty JSON is the canonical human-readable format for `show`.
    let _ = as_json;
    println!("{}", serde_json::to_string_pretty(&payload)?);
    if let Some(snap) = &usage {
        println!();
        println!("{}", format_usage_line(snap));
    }
    Ok(())
}

/// Lists the models a provider currently serves by calling its live
/// `/v1/models` endpoint. Falls back to the hardcoded catalogue — with a
/// visible "not supported" note — for providers whose upstream has no
/// equivalent listing (codex's Responses API, opencode Zen, z.ai).
async fn models(name: &str, as_json: bool) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let config = Config::load_or_default()?;
    let p = registry.get(name)?;
    let settings = config.providers.get(name).cloned().unwrap_or_default();

    let hardcoded: Vec<_> = p
        .models()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "display_name": m.display_name,
                "supports_images": m.supports_images,
                "context_window": m.context_window,
            })
        })
        .collect();

    // `list_remote_models` returns `Ok(None)` for providers without a live
    // endpoint, `Err(...)` for providers that support it but failed this
    // call (network, 4xx/5xx, parse error). Both surface distinctly: None →
    // "unsupported" note, Err → printed error before we bail.
    let remote_result = p.list_remote_models(&credentials, &settings).await;

    if as_json {
        let payload = match &remote_result {
            Ok(Some(entries)) => json!({
                "provider": p.id(),
                "remote_supported": true,
                "remote": entries.iter().map(|e| &e.raw).collect::<Vec<_>>(),
                "hardcoded": hardcoded,
            }),
            Ok(None) => json!({
                "provider": p.id(),
                "remote_supported": false,
                "hardcoded": hardcoded,
            }),
            Err(e) => json!({
                "provider": p.id(),
                "remote_supported": true,
                "remote_error": format!("{e:#}"),
                "hardcoded": hardcoded,
            }),
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("Hardcoded catalogue (used by the router):");
    if hardcoded.is_empty() {
        println!("  (none)");
    } else {
        for m in &hardcoded {
            let id = m["id"].as_str().unwrap_or("");
            let display = m["display_name"].as_str().unwrap_or("");
            let ctx = m["context_window"].as_u64().unwrap_or(0);
            let vision = m["supports_images"].as_bool().unwrap_or(false);
            let vision_tag = if vision { ", vision" } else { "" };
            println!(
                "  {id:40} {display} ({} ctx{vision_tag})",
                humanize_ctx(ctx)
            );
        }
    }

    println!();
    match &remote_result {
        Ok(Some(entries)) => {
            println!("Live from upstream /v1/models:");
            if entries.is_empty() {
                println!("  (none — provider returned an empty list)");
            } else {
                for e in entries {
                    println!("  {}", e.id);
                }
                println!();
                println!(
                    "Note: live entries are advertised by the provider — not all may be \
                    available on your plan, and capability metadata (vision, context window) \
                    is not returned by /v1/models."
                );
            }
        }
        Ok(None) => {
            println!(
                "Live /v1/models not supported by this provider. Add new models by editing \
                 their ModelInfo list in `src/providers/{}.rs`.",
                p.id().replace('-', "_")
            );
        }
        Err(e) => {
            println!("Live /v1/models call failed: {e:#}");
        }
    }

    Ok(())
}

/// Render a raw token count as a compact width label (e.g. `200_000 → 200K`).
/// `0` is treated as "unknown" because the router uses `context_window: 0`
/// only as a sentinel for "not advertised".
fn humanize_ctx(n: u64) -> String {
    if n == 0 {
        return "?".to_string();
    }
    if n >= 1_000_000 {
        let m = n as f64 / 1_000_000.0;
        if (m.round() - m).abs() < f64::EPSILON {
            format!("{:.0}M", m)
        } else {
            format!("{:.1}M", m)
        }
    } else if n >= 1_000 {
        let k = n as f64 / 1_000.0;
        if (k.round() - k).abs() < f64::EPSILON {
            format!("{:.0}K", k)
        } else {
            format!("{:.1}K", k)
        }
    } else {
        n.to_string()
    }
}

/// Human-friendly one-liner for a [`UsageSnapshot`], shown alongside the JSON
/// payload. Intentionally concise — the JSON already carries the full shape.
fn format_usage_line(snap: &crate::usage::UsageSnapshot) -> String {
    let pct = snap
        .fraction_used
        .map(|f| format!("{:.1}%", (f * 100.0).clamp(0.0, 100.0)))
        .unwrap_or_else(|| "unknown".to_string());
    match &snap.label {
        Some(label) => format!("usage: {pct} ({} — {})", snap.window, label),
        None => format!("usage: {pct} ({})", snap.window),
    }
}
