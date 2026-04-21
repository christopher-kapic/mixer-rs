use std::sync::Arc;

use anyhow::{Result, bail};

use crate::concurrency::ConcurrencyLimits;
use crate::config::Config;
use crate::credentials::CredentialStore;
use crate::providers::builtin_registry;
use crate::server::{AppState, serve};
use crate::usage::UsageCache;

pub async fn run(addr: Option<String>, port: Option<u16>, model: Option<String>) -> Result<()> {
    let config = Config::load_or_default()?;
    let credentials = CredentialStore::new()?;
    let registry = builtin_registry();

    if let Some(name) = &model
        && !config.models.contains_key(name)
    {
        bail!(
            "no mixer model named `{}` in config (available: {})",
            name,
            config.models.keys().cloned().collect::<Vec<_>>().join(", ")
        );
    }

    let listen_addr = resolve_listen_addr(&config.listen_addr, addr.as_deref(), port);
    let concurrency = ConcurrencyLimits::from_config(&config);

    let state = AppState {
        config: Arc::new(config),
        registry: Arc::new(registry),
        credentials: Arc::new(credentials),
        concurrency,
        usage_cache: UsageCache::default(),
        pinned_model: model,
    };

    serve(state, &listen_addr).await
}

fn resolve_listen_addr(
    config_addr: &str,
    addr_override: Option<&str>,
    port_override: Option<u16>,
) -> String {
    if let Some(a) = addr_override {
        return a.to_string();
    }
    let Some(p) = port_override else {
        return config_addr.to_string();
    };
    // Replace the port component of `config_addr` with `p`.
    let host = config_addr
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or("127.0.0.1");
    format!("{host}:{p}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_listen_addr_uses_config_by_default() {
        assert_eq!(
            resolve_listen_addr("127.0.0.1:4141", None, None),
            "127.0.0.1:4141"
        );
    }

    #[test]
    fn resolve_listen_addr_respects_addr_override() {
        assert_eq!(
            resolve_listen_addr("127.0.0.1:4141", Some("0.0.0.0:8080"), None),
            "0.0.0.0:8080"
        );
    }

    #[test]
    fn resolve_listen_addr_replaces_port() {
        assert_eq!(
            resolve_listen_addr("127.0.0.1:4141", None, Some(9000)),
            "127.0.0.1:9000"
        );
    }

    #[test]
    fn resolve_listen_addr_prefers_addr_over_port() {
        assert_eq!(
            resolve_listen_addr("127.0.0.1:4141", Some("0.0.0.0:8080"), Some(9000)),
            "0.0.0.0:8080"
        );
    }
}
