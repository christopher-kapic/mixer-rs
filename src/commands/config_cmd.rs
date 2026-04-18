use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::cli::ConfigCommand;
use crate::config::Config;
use crate::paths;

pub fn run(cmd: &ConfigCommand) -> Result<()> {
    match cmd {
        ConfigCommand::Show { json } => show(*json),
        ConfigCommand::Edit => edit(),
        ConfigCommand::Set { key, value } => set(key, value),
        ConfigCommand::Path => {
            println!("{}", paths::config_file()?.display());
            Ok(())
        }
    }
}

fn show(as_json: bool) -> Result<()> {
    // Pretty JSON is already the canonical human format for mixer's config.
    let _ = as_json;
    let config = Config::load_or_default()?;
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}

fn edit() -> Result<()> {
    let path = paths::config_file()?;
    if !path.exists() {
        Config::default().save(&path)?;
    }
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let status = Command::new(&editor)
        .arg(&path)
        .status()
        .with_context(|| format!("spawning editor `{editor}`"))?;
    if !status.success() {
        bail!("editor `{editor}` exited with status {status}");
    }
    // Validate on the way out so typos surface before the next `mixer serve`.
    Config::load(&path).context("re-loading config after edit")?;
    Ok(())
}

/// Set a dotted-path config value. Supported keys (for now):
///   - `listen_addr` (string)
///   - `default_model` (string)
///   - `providers.<id>.enabled` (bool)
///   - `providers.<id>.max_concurrent_requests` (integer)
///   - `providers.<id>.base_url` (string)
///   - `providers.<id>.request_timeout_secs` (integer)
///   - `providers.<id>.api_key_env` (string — name of the env var, not the key)
fn set(key: &str, value: &str) -> Result<()> {
    let path = paths::config_file()?;
    let mut config = if path.exists() {
        Config::load(&path)?
    } else {
        Config::default()
    };

    match key {
        "listen_addr" => config.listen_addr = value.to_string(),
        "default_model" => config.default_model = value.to_string(),
        k if k.starts_with("providers.") => {
            let rest = &k["providers.".len()..];
            let (provider_id, field) = rest
                .split_once('.')
                .with_context(|| format!("unsupported config key `{key}`"))?;
            let entry = config.providers.entry(provider_id.to_string()).or_default();
            match field {
                "enabled" => {
                    entry.enabled = parse_bool(value)?;
                }
                "max_concurrent_requests" => {
                    entry.max_concurrent_requests = parse_opt_u32(value)?;
                }
                "base_url" => {
                    entry.base_url = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "request_timeout_secs" => {
                    entry.request_timeout_secs = parse_opt_u64(value)?;
                }
                "api_key_env" => {
                    entry.api_key_env = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "api_key" => bail!(
                    "refusing to store a literal API key in config; use `api_key_env` to name an environment variable, or `mixer auth login {provider_id}` to store it in credentials"
                ),
                other => bail!("unsupported provider field `{other}`"),
            }
        }
        _ => bail!("unsupported config key `{key}`"),
    }

    config.save(&path)?;
    Ok(())
}

fn parse_bool(s: &str) -> Result<bool> {
    match s {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => bail!("expected `true` or `false`, got `{other}`"),
    }
}

fn parse_opt_u32(s: &str) -> Result<Option<u32>> {
    if s.is_empty() || s == "null" || s == "none" {
        return Ok(None);
    }
    let n: u32 = s
        .parse()
        .with_context(|| format!("expected a non-negative integer, got `{s}`"))?;
    Ok(Some(n))
}

fn parse_opt_u64(s: &str) -> Result<Option<u64>> {
    if s.is_empty() || s == "null" || s == "none" {
        return Ok(None);
    }
    let n: u64 = s
        .parse()
        .with_context(|| format!("expected a non-negative integer, got `{s}`"))?;
    Ok(Some(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bool_matrix() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("1").unwrap());
        assert!(parse_bool("on").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("off").unwrap());
        assert!(parse_bool("maybe").is_err());
    }

    #[test]
    fn parse_opt_u32_handles_clearing_sentinels() {
        assert_eq!(parse_opt_u32("").unwrap(), None);
        assert_eq!(parse_opt_u32("null").unwrap(), None);
        assert_eq!(parse_opt_u32("none").unwrap(), None);
        assert_eq!(parse_opt_u32("4").unwrap(), Some(4));
        assert!(parse_opt_u32("-1").is_err());
    }
}
