use anyhow::Result;

use crate::credentials::CredentialStore;
use crate::providers::builtin_registry;

pub async fn login(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let provider = registry.get(provider_id)?;
    provider.login(&credentials).await?;
    eprintln!("logged in to `{}`", provider.id());
    Ok(())
}

pub async fn logout(provider_id: &str) -> Result<()> {
    let registry = builtin_registry();
    let credentials = CredentialStore::new()?;
    let provider = registry.get(provider_id)?;
    provider.logout(&credentials).await?;
    eprintln!("logged out of `{}`", provider.id());
    Ok(())
}

pub async fn status(provider: Option<&str>, json: bool) -> Result<()> {
    let _ = (provider, json);
    eprintln!("mixer auth status: not implemented yet");
    Ok(())
}
