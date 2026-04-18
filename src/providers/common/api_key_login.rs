//! Interactive "paste an API key" login flow shared by API-key providers.
//!
//! Writes the resulting blob as `{ "api_key": "..." }` via
//! [`CredentialStore::save`], preserving the 0600 permissions that the store
//! sets on Unix. Each API-key provider's `login` delegates here so they all
//! share the same prompt ergonomics.
//!
//! The prompt reads via `rpassword`, so the key is not echoed back to the
//! terminal. Env-var shadowing warnings are handled by
//! `commands::auth_cmd::login` (which runs after `Provider::login`), not here.

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::credentials::CredentialStore;

/// Prompt for an API key on the terminal (not echoed), validate that it is
/// non-empty, and persist the `{ "api_key": … }` blob under `provider_id`.
///
/// `display_name` is shown in the prompt so the user knows which provider
/// they're pasting a key for.
pub fn prompt_and_store_api_key(
    store: &CredentialStore,
    provider_id: &str,
    display_name: &str,
) -> Result<()> {
    let key = rpassword::prompt_password(format!("Enter {display_name} API key: "))
        .context("reading api key from stdin")?;
    let key = key.trim().to_string();
    if key.is_empty() {
        bail!("no api key entered");
    }
    store.save(provider_id, &json!({ "api_key": key }))?;
    Ok(())
}
