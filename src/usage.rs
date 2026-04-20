//! Shared usage-snapshot type reported by providers that can introspect their
//! current subscription consumption. Consumed by the usage-aware router.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A provider's view of its current subscription usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageSnapshot {
    /// Fraction of the current billing period consumed, in `[0.0, 1.0]`.
    /// `None` means the provider cannot determine this (e.g. no usage
    /// endpoint, or usage is denominated in an incomparable unit).
    pub fraction_used: Option<f64>,

    /// Short human-readable window label (e.g. `"monthly"`, `"daily"`).
    pub window: String,

    /// Free-form label shown to the user (e.g. `"1.2M / 5M tokens"`).
    pub label: Option<String>,
}

impl UsageSnapshot {
    #[allow(dead_code)]
    pub fn unknown(window: impl Into<String>) -> Self {
        Self {
            fraction_used: None,
            window: window.into(),
            label: None,
        }
    }
}

/// TTL cache for `UsageSnapshot` lookups, keyed by provider id.
///
/// The usage-aware router consults a provider's [`crate::providers::Provider::usage`]
/// endpoint on every request. Without caching, a single slow or flaky usage
/// endpoint stalls every routing decision; with a short TTL the worst case is
/// one slow call per provider per `ttl` window. A `None` (provider can't
/// report) is cached just like a `Some(_)`, since the underlying answer
/// changes on the same timescale either way.
#[derive(Clone)]
pub struct UsageCache {
    inner: Arc<RwLock<HashMap<String, CacheEntry>>>,
    ttl: Duration,
}

#[derive(Clone)]
struct CacheEntry {
    fetched_at: Instant,
    snapshot: Option<UsageSnapshot>,
}

impl Default for UsageCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(60))
    }
}

impl UsageCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl,
        }
    }

    /// Return the cached snapshot for `key` if it is still fresh; otherwise
    /// invoke `fetch`, store its result (success or `Ok(None)`) under `key`
    /// with the current timestamp, and return it.
    ///
    /// On `fetch` error nothing is cached and the error is propagated; the
    /// usage-aware router treats that as "weight 0.5" so a transient endpoint
    /// failure neither poisons future picks nor falsely-prefers/excludes the
    /// provider.
    pub async fn get_or_fetch<F, Fut>(
        &self,
        key: &str,
        fetch: F,
    ) -> Result<Option<UsageSnapshot>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<UsageSnapshot>>>,
    {
        if let Some(snap) = self.peek_fresh(key) {
            return Ok(snap);
        }
        let snap = fetch().await?;
        self.store(key, snap.clone());
        Ok(snap)
    }

    fn peek_fresh(&self, key: &str) -> Option<Option<UsageSnapshot>> {
        let guard = self.inner.read().expect("usage cache poisoned");
        let entry = guard.get(key)?;
        if entry.fetched_at.elapsed() < self.ttl {
            Some(entry.snapshot.clone())
        } else {
            None
        }
    }

    fn store(&self, key: &str, snapshot: Option<UsageSnapshot>) {
        let mut guard = self.inner.write().expect("usage cache poisoned");
        guard.insert(
            key.to_string(),
            CacheEntry {
                fetched_at: Instant::now(),
                snapshot,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn snap(f: f64) -> UsageSnapshot {
        UsageSnapshot {
            fraction_used: Some(f),
            window: "monthly".to_string(),
            label: None,
        }
    }

    #[tokio::test]
    async fn second_call_within_ttl_uses_cache() {
        let cache = UsageCache::new(Duration::from_secs(60));
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let calls = calls.clone();
            let result = cache
                .get_or_fetch("p", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(Some(snap(0.4)))
                })
                .await
                .unwrap();
            assert_eq!(result, Some(snap(0.4)));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "TTL should coalesce calls");
    }

    #[tokio::test]
    async fn expired_entry_triggers_refetch() {
        let cache = UsageCache::new(Duration::from_millis(10));
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = calls.clone();
            cache
                .get_or_fetch("p", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(Some(snap(0.4)))
                })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "stale entries must refetch"
        );
    }

    #[tokio::test]
    async fn fetch_error_is_not_cached() {
        let cache = UsageCache::new(Duration::from_secs(60));
        let _ = cache
            .get_or_fetch::<_, _>("p", || async { Err(anyhow::anyhow!("boom")) })
            .await
            .unwrap_err();

        // Next call hits the closure again because the previous error wasn't
        // cached as a successful "None" entry.
        let result = cache
            .get_or_fetch("p", || async { Ok(Some(snap(0.1))) })
            .await
            .unwrap();
        assert_eq!(result, Some(snap(0.1)));
    }

    #[tokio::test]
    async fn none_snapshot_is_cached() {
        let cache = UsageCache::new(Duration::from_secs(60));
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let calls = calls.clone();
            let result = cache
                .get_or_fetch("p", || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(None)
                })
                .await
                .unwrap();
            assert!(result.is_none());
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "Ok(None) is a real answer and must be cached",
        );
    }
}
