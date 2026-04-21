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
