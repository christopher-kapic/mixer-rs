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

use crate::config::ProviderSettings;
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

    /// Load a provider's stored credential blob from disk, ignoring any
    /// environment-variable override. Returns `None` when the file is absent
    /// or empty. This is the file-only counterpart to `load_api_key`, which
    /// layers env precedence on top.
    #[allow(dead_code)]
    pub fn load_blob(&self, provider_id: &str) -> Result<Option<Value>> {
        self.load(provider_id)
    }

    /// Resolve the API key for a provider, preferring the environment variable
    /// named in `settings.api_key_env` when that variable is set and non-empty.
    /// Otherwise falls back to the `api_key` string field of the stored
    /// credential blob. Returns `None` when neither source yields a value.
    #[allow(dead_code)]
    pub fn load_api_key(&self, provider_id: &str, settings: &ProviderSettings) -> Option<String> {
        if let Some(var) = settings.api_key_env.as_deref() {
            if let Ok(val) = std::env::var(var) {
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }
        let blob = self.load_blob(provider_id).ok().flatten()?;
        blob.get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Remove a provider's stored credentials. No-op if the file is absent.
    pub fn remove(&self, provider_id: &str) -> Result<()> {
        let path = self.file(provider_id);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn with_dir(dir: PathBuf) -> Self {
        Self { dir }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn store() -> (TempDir, CredentialStore) {
        let tmp = TempDir::new().unwrap();
        let store = CredentialStore::with_dir(tmp.path().to_path_buf());
        (tmp, store)
    }

    fn settings_with_env(var: Option<&str>) -> ProviderSettings {
        ProviderSettings {
            api_key_env: var.map(|s| s.to_string()),
            ..ProviderSettings::default_enabled()
        }
    }

    #[test]
    fn load_api_key_prefers_env_over_file() {
        let (_tmp, store) = store();
        store
            .save("minimax", &json!({ "api_key": "from-file" }))
            .unwrap();

        let var = "MIXER_TEST_ENV_FIRST_KEY";
        // SAFETY: single-threaded test, scoped to a unique var name.
        unsafe { std::env::set_var(var, "from-env") };
        let key = store.load_api_key("minimax", &settings_with_env(Some(var)));
        unsafe { std::env::remove_var(var) };

        assert_eq!(key.as_deref(), Some("from-env"));
    }

    #[test]
    fn load_api_key_falls_back_to_file_when_env_unset() {
        let (_tmp, store) = store();
        store
            .save("minimax", &json!({ "api_key": "from-file" }))
            .unwrap();

        let var = "MIXER_TEST_ENV_UNSET_KEY";
        unsafe { std::env::remove_var(var) };
        let key = store.load_api_key("minimax", &settings_with_env(Some(var)));

        assert_eq!(key.as_deref(), Some("from-file"));
    }

    #[test]
    fn load_api_key_treats_empty_env_as_unset() {
        let (_tmp, store) = store();
        store
            .save("minimax", &json!({ "api_key": "from-file" }))
            .unwrap();

        let var = "MIXER_TEST_EMPTY_ENV_KEY";
        unsafe { std::env::set_var(var, "") };
        let key = store.load_api_key("minimax", &settings_with_env(Some(var)));
        unsafe { std::env::remove_var(var) };

        assert_eq!(key.as_deref(), Some("from-file"));
    }

    #[test]
    fn load_api_key_returns_none_when_neither_source_set() {
        let (_tmp, store) = store();

        let var = "MIXER_TEST_NEITHER_KEY";
        unsafe { std::env::remove_var(var) };
        let key = store.load_api_key("minimax", &settings_with_env(Some(var)));

        assert!(key.is_none());
    }

    #[test]
    fn load_api_key_falls_back_to_file_when_no_env_configured() {
        let (_tmp, store) = store();
        store
            .save("minimax", &json!({ "api_key": "from-file" }))
            .unwrap();

        let key = store.load_api_key("minimax", &settings_with_env(None));

        assert_eq!(key.as_deref(), Some("from-file"));
    }

    #[test]
    fn load_blob_returns_raw_file_ignoring_env() {
        let (_tmp, store) = store();
        store
            .save("minimax", &json!({ "api_key": "from-file" }))
            .unwrap();

        let var = "MIXER_TEST_BLOB_IGNORE_ENV";
        unsafe { std::env::set_var(var, "from-env") };
        let blob = store.load_blob("minimax").unwrap().unwrap();
        unsafe { std::env::remove_var(var) };

        assert_eq!(blob["api_key"], "from-file");
    }

    #[test]
    fn load_blob_returns_none_when_missing() {
        let (_tmp, store) = store();
        assert!(store.load_blob("nope").unwrap().is_none());
    }
}
