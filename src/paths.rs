use std::path::PathBuf;

use anyhow::{Context, Result};

const APP_NAME: &str = "mixer";

/// `$XDG_CONFIG_HOME/mixer` or `~/.config/mixer`.
pub fn config_dir() -> Result<PathBuf> {
    let base = xdg_dir_or_home_relative("XDG_CONFIG_HOME", &[".config"])?;
    Ok(base.join(APP_NAME))
}

/// `$XDG_DATA_HOME/mixer` or `~/.local/share/mixer`.
#[allow(dead_code)]
pub fn data_dir() -> Result<PathBuf> {
    let base = xdg_dir_or_home_relative("XDG_DATA_HOME", &[".local", "share"])?;
    Ok(base.join(APP_NAME))
}

/// `$XDG_STATE_HOME/mixer` or `~/.local/state/mixer`.
#[allow(dead_code)]
pub fn state_dir() -> Result<PathBuf> {
    let base = xdg_dir_or_home_relative("XDG_STATE_HOME", &[".local", "state"])?;
    Ok(base.join(APP_NAME))
}

fn xdg_dir_or_home_relative(var: &str, fallback: &[&str]) -> Result<PathBuf> {
    if let Some(val) = std::env::var_os(var) {
        let path = PathBuf::from(val);
        if path.is_absolute() {
            return Ok(path);
        }
    }
    let mut home = dirs::home_dir().context("could not determine home directory")?;
    for component in fallback {
        home.push(component);
    }
    Ok(home)
}

pub fn config_file() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.json"))
}

pub fn credentials_dir() -> Result<PathBuf> {
    Ok(config_dir()?.join("credentials"))
}

#[allow(dead_code)]
pub fn credentials_file(provider_id: &str) -> Result<PathBuf> {
    Ok(credentials_dir()?.join(format!("{provider_id}.json")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_dir_ends_with_mixer() {
        let dir = config_dir().unwrap();
        assert_eq!(dir.file_name().unwrap(), "mixer");
    }

    #[test]
    fn config_file_is_json() {
        let path = config_file().unwrap();
        assert_eq!(path.file_name().unwrap(), "config.json");
    }

    #[test]
    fn credentials_file_named_per_provider() {
        let path = credentials_file("codex").unwrap();
        assert_eq!(path.file_name().unwrap(), "codex.json");
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "credentials");
    }
}
