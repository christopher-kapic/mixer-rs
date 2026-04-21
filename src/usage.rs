//! Shared usage-snapshot type reported by providers that can introspect their
//! current subscription consumption. Consumed by the usage-aware router.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

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
///
/// Concurrent misses for the same key are coalesced behind a per-provider
/// async lock: a burst of simultaneous requests after startup (or after TTL
/// expiry) issues exactly one fetch, and the rest wait on the lock and read
/// the populated cache entry after it lands.
#[derive(Clone)]
pub struct UsageCache {
    inner: Arc<RwLock<HashMap<String, CacheEntry>>>,
    locks: Arc<RwLock<HashMap<String, Arc<AsyncMutex<()>>>>>,
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
            locks: Arc::new(RwLock::new(HashMap::new())),
            ttl,
        }
    }

    /// Return the cached snapshot for `key` if it is still fresh; otherwise
    /// invoke `fetch`, store its result (success or `Ok(None)`) under `key`
    /// with the current timestamp, and return it.
    ///
    /// Concurrent callers observing a stale/missing entry for the same `key`
    /// are serialized on a per-key async lock: the winner runs `fetch`, stores
    /// the result, and releases; queued callers then find a fresh entry and
    /// return it without re-fetching.
    ///
    /// On `fetch` error nothing is cached and the error is propagated; the
    /// usage-aware router treats that as "weight 0.5" so a transient endpoint
    /// failure neither poisons future picks nor falsely-prefers/excludes the
    /// provider. A queued caller behind a failed fetch will retry the fetch
    /// itself (still serialized — we avoid parallel hammering, not sequential
    /// retries).
    pub async fn get_or_fetch<F, Fut>(&self, key: &str, fetch: F) -> Result<Option<UsageSnapshot>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<UsageSnapshot>>>,
    {
        if let Some(snap) = self.peek_fresh(key) {
            return Ok(snap);
        }

        let lock = self.lock_for(key);
        let _guard = lock.lock().await;

        // Another caller may have populated the cache while we waited.
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

    fn lock_for(&self, key: &str) -> Arc<AsyncMutex<()>> {
        if let Some(lock) = self
            .locks
            .read()
            .expect("usage cache lock map poisoned")
            .get(key)
        {
            return lock.clone();
        }
        let mut guard = self.locks.write().expect("usage cache lock map poisoned");
        guard
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
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
    async fn concurrent_misses_coalesce_to_single_fetch() {
        let cache = UsageCache::new(Duration::from_secs(60));
        let calls = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let calls = calls.clone();
            let release = release.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_fetch("p", || async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        // Block so peer tasks can queue up on the per-key lock.
                        release.notified().await;
                        Ok(Some(snap(0.4)))
                    })
                    .await
                    .unwrap()
            }));
        }

        // Give all tasks a chance to reach the fetch / queue for the lock.
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_waiters();

        for h in handles {
            assert_eq!(h.await.unwrap(), Some(snap(0.4)));
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "concurrent misses on the same key must coalesce to one fetch",
        );
    }

    #[tokio::test]
    async fn concurrent_misses_on_distinct_keys_do_not_serialize() {
        // Two keys fetched in parallel should both make progress without
        // blocking on each other's per-key lock.
        let cache = UsageCache::new(Duration::from_secs(60));
        let gate = Arc::new(tokio::sync::Barrier::new(2));

        let c1 = cache.clone();
        let g1 = gate.clone();
        let h1 = tokio::spawn(async move {
            c1.get_or_fetch("a", || async move {
                g1.wait().await;
                Ok(Some(snap(0.1)))
            })
            .await
            .unwrap()
        });

        let c2 = cache.clone();
        let g2 = gate.clone();
        let h2 = tokio::spawn(async move {
            c2.get_or_fetch("b", || async move {
                g2.wait().await;
                Ok(Some(snap(0.2)))
            })
            .await
            .unwrap()
        });

        // Both fetches must reach the barrier concurrently; if the cache
        // serialized across keys, this would deadlock (timeout below).
        let result =
            tokio::time::timeout(Duration::from_secs(1), async move { tokio::join!(h1, h2) })
                .await
                .expect("per-key locks must not serialize distinct keys");
        assert_eq!(result.0.unwrap(), Some(snap(0.1)));
        assert_eq!(result.1.unwrap(), Some(snap(0.2)));
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
