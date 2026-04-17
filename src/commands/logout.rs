use anyhow::Result;

use crate::credentials::CredentialStore;
use crate::providers::builtin_registry;

pub async fn run(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let provider = registry.get(provider_id)?;
    provider.logout(&credentials).await?;
    eprintln!("logged out of `{}`", provider.id());
    Ok(())
}
