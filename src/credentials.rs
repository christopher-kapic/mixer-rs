//! Per-provider credential storage.
//!
//! Each provider owns an opaque JSON blob written to
//! `~/.config/mixer/credentials/<provider_id>.json`. The shape is entirely up
//! to the provider — mixer stores and retrieves it as `serde_json::Value`.
//!
//! On Unix, files are chmod'd to `0600` so only the owning user can read
//! them.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::paths;

pub struct CredentialStore {
    dir: PathBuf,
}

impl CredentialStore {
    pub fn new() -> Result<Self> {
        let dir = paths::credentials_dir()?;
        Ok(Self { dir })
    }

    /// Full path to the credentials file for a given provider id.
    pub fn file(&self, provider_id: &str) -> PathBuf {
        self.dir.join(format!("{provider_id}.json"))
    }

    pub fn exists(&self, provider_id: &str) -> bool {
        self.file(provider_id).exists()
    }

    /// Load a provider's credentials, returning `None` if the file is absent.
    #[allow(dead_code)]
    pub fn load(&self, provider_id: &str) -> Result<Option<Value>> {
        let path = self.file(provider_id);
        if !path.exists() {
            return Ok(None);
        }
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        if contents.trim().is_empty() {
            return Ok(None);
        }
        let v: Value = serde_json::from_str(&contents)
            .with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(v))
    }

    /// Atomically save a provider's credentials.
    #[allow(dead_code)]
    pub fn save(&self, provider_id: &str, value: &Value) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;

        let path = self.file(provider_id);
        let json = serde_json::to_string_pretty(value).context("serializing credentials")?;

        let mut tmp = tempfile::NamedTempFile::new_in(&self.dir)
            .with_context(|| format!("creating temp file in {}", self.dir.display()))?;
        tmp.write_all(json.as_bytes())
            .context("writing credentials")?;
        tmp.flush().context("flushing credentials temp file")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            tmp.as_file()
                .set_permissions(perms)
                .context("restricting credentials file permissions")?;
        }

        tmp.persist(&path)
            .with_context(|| format!("persisting credentials to {}", path.display()))?;
        Ok(())
    }

    /// Remove a provider's stored credentials. No-op if the file is absent.
    pub fn remove(&self, provider_id: &str) -> Result<()> {
        let path = self.file(provider_id);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }
}
