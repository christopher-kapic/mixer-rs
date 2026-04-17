use anyhow::Result;

use crate::credentials::CredentialStore;
use crate::providers::builtin_registry;

pub async fn run(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let provider = registry.get(provider_id)?;
    provider.login(&credentials).await?;
    eprintln!("logged in to `{}`", provider.id());
    Ok(())
}
