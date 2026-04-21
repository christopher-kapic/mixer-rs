//! Per-provider concurrency caps.
//!
//! Each provider can have a `max_concurrent_requests` setting. At server
//! startup we build a map from provider id to an optional [`Semaphore`]:
//! providers without a cap get no semaphore (unlimited), providers with a
//! cap get a semaphore sized to the cap. Requests acquire a permit before
//! dispatching and release it when the provider call returns; callers
//! beyond the cap queue inside `acquire`, they are not rejected.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::Config;

#[derive(Clone, Default)]
pub struct ConcurrencyLimits {
    caps: HashMap<String, Arc<Semaphore>>,
}

impl ConcurrencyLimits {
    pub fn from_config(config: &Config) -> Self {
        let mut caps = HashMap::new();
        for (id, settings) in &config.providers {
            if let Some(n) = settings.max_concurrent_requests
                && n > 0
            {
                caps.insert(id.clone(), Arc::new(Semaphore::new(n as usize)));
            }
        }
        Self { caps }
    }

    /// Acquire a permit for the given provider. Returns `None` when the
    /// provider is uncapped — the caller is free to proceed immediately.
    pub async fn acquire(&self, provider_id: &str) -> Option<OwnedSemaphorePermit> {
        let sem = self.caps.get(provider_id)?.clone();
        // A Semaphore we control is never closed, so acquire cannot error.
        Some(sem.acquire_owned().await.expect("mixer semaphore closed"))
    }

    #[cfg(test)]
    pub fn capped_providers(&self) -> Vec<&String> {
        self.caps.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderSettings;

    #[tokio::test]
    async fn uncapped_provider_returns_none() {
        let mut c = Config::default();
        c.providers.insert(
            "free".to_string(),
            ProviderSettings {
                enabled: true,
                ..ProviderSettings::default_enabled()
            },
        );
        let limits = ConcurrencyLimits::from_config(&c);
        assert!(limits.acquire("free").await.is_none());
    }

    #[tokio::test]
    async fn capped_provider_serialises_requests() {
        let mut c = Config::default();
        // Clear defaults so the assertion on `capped_providers()` only sees
        // the selfhost entry we're exercising — `Config::default` ships with
        // its own capped ollama entry that is irrelevant to this test.
        c.providers.clear();
        c.providers.insert(
            "selfhost".to_string(),
            ProviderSettings {
                enabled: true,
                max_concurrent_requests: Some(1),
                ..ProviderSettings::default_enabled()
            },
        );
        let limits = ConcurrencyLimits::from_config(&c);
        assert_eq!(limits.capped_providers(), vec![&"selfhost".to_string()]);

        let first = limits.acquire("selfhost").await.unwrap();
        // Second acquire must wait — time-box to prove it's blocked.
        let second = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            limits.acquire("selfhost"),
        )
        .await;
        assert!(
            second.is_err(),
            "second permit should be pending while first is held"
        );
        drop(first);
        let _second = limits.acquire("selfhost").await.unwrap();
    }
}
