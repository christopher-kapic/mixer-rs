//! Backend selection: given a mixer model + an inbound request, pick which
//! concrete `(provider, provider_model)` should service it.
//!
//! Selection is a two-phase process:
//!
//! 1. **Filter** the backend pool down to candidates that are actually
//!    usable for this request: the provider must be registered, enabled in
//!    config, authenticated, and — if the request carries an image — the
//!    backend model must report `supports_images = true`.
//! 2. **Pick** one candidate. When the mixer model has a [`StickyConfig`]
//!    enabled and the caller supplied a sticky key derived from the request,
//!    we route via consistent hashing so the same key pins to the same
//!    backend. Otherwise the model's [`RoutingStrategy`] takes over:
//!    uniformly random, weighted random, or usage-aware.
//!
//! Stickiness is a *preference*, not a hard pin — if the hashed backend is
//! filtered out (auth expired, request needs images but the pinned backend is
//! text-only, etc.) we fall back to the normal strategy so the request still
//! completes. The per-request failover loop in `server.rs` (plan.md §5.2.1)
//! can likewise move a sticky request to a different backend on a retryable
//! error; the key just biases the initial choice.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::{Result, anyhow, bail};
use axum::http::HeaderMap;
use rand::Rng;

use crate::config::{Backend, Config, MixerModel, RoutingStrategy, StickyKey};
use crate::credentials::CredentialStore;
use crate::openai::{ChatMessage, ChatRequest, MessageContent};
use crate::providers::ProviderRegistry;

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub provider_id: String,
    pub provider_model: String,
}

/// Pick a backend for the given mixer model + request characteristics.
pub async fn pick(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
    mixer_model: &MixerModel,
    requires_images: bool,
    sticky_hash: Option<u64>,
) -> Result<RouteDecision> {
    pick_excluding(
        config,
        registry,
        credentials,
        mixer_model,
        requires_images,
        &[],
        sticky_hash,
    )
    .await
}

/// Like [`pick`], but rules out specific `(provider, model)` backends. Used by
/// the failover path in the server: when the first attempt fails with a
/// retryable error, we re-pick from the same mixer model's pool excluding the
/// backend that just failed.
pub async fn pick_excluding(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
    mixer_model: &MixerModel,
    requires_images: bool,
    excluded: &[Backend],
    sticky_hash: Option<u64>,
) -> Result<RouteDecision> {
    let mut candidates = filter_candidates(
        config,
        registry,
        credentials,
        &mixer_model.backends,
        requires_images,
    );
    if !excluded.is_empty() {
        candidates.retain(|b| {
            !excluded
                .iter()
                .any(|e| e.provider == b.provider && e.model == b.model)
        });
    }

    if candidates.is_empty() {
        if excluded.is_empty() {
            bail!(
                "no eligible backends for this request (try `mixer providers list` \
to see which providers are authenticated{}",
                if requires_images {
                    "; the request has images so only vision-capable models qualify)"
                } else {
                    ")"
                }
            );
        }
        bail!(
            "no eligible backends remain after excluding {} failed backend(s)",
            excluded.len()
        );
    }

    // Sticky path: consistent hashing over the already-filtered pool. If no
    // key was computed (sticky disabled, or the chosen key type produced
    // nothing — e.g. absent header) we fall straight through to the normal
    // strategy.
    let idx = if let Some(h) = sticky_hash {
        sticky_pick(&candidates, h)
    } else {
        match mixer_model.strategy {
            RoutingStrategy::Random => uniform_pick(candidates.len()),
            RoutingStrategy::Weighted => weighted_pick(&candidates, |b| {
                *mixer_model.weights.get(&b.provider).unwrap_or(&1.0)
            })?,
            RoutingStrategy::UsageAware => {
                usage_aware_pick(config, registry, credentials, &candidates).await?
            }
        }
    };

    let chosen = &candidates[idx];
    Ok(RouteDecision {
        provider_id: chosen.provider.clone(),
        provider_model: chosen.model.clone(),
    })
}

/// Derive a sticky-routing key from the request body and headers for the
/// given mixer model. Returns `None` when stickiness is disabled, the chosen
/// key source isn't present (e.g. absent header), or the derived prefix is
/// empty. Callers pass the result to [`pick`]/[`pick_excluding`].
pub fn compute_sticky_hash(
    mixer_model: &MixerModel,
    req: &ChatRequest,
    headers: &HeaderMap,
) -> Option<u64> {
    let sticky = mixer_model.sticky.as_ref()?;
    if !sticky.enabled {
        return None;
    }
    match &sticky.key {
        StickyKey::MessagesHash => hash_messages_prefix(&req.messages),
        StickyKey::Header(name) => {
            let value = headers.get(name).and_then(|v| v.to_str().ok())?;
            if value.is_empty() {
                return None;
            }
            Some(hash_bytes(value.as_bytes()))
        }
    }
}

/// Hash the "stable" prefix of a conversation: every message except the
/// trailing run of user-role messages (usually just the final user turn).
/// Returns `None` when the prefix is empty so the router falls through to
/// its normal strategy rather than pinning every single-turn request to the
/// same backend.
fn hash_messages_prefix(messages: &[ChatMessage]) -> Option<u64> {
    let mut end = messages.len();
    while end > 0 && messages[end - 1].role == "user" {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut hasher = DefaultHasher::new();
    for m in &messages[..end] {
        m.role.hash(&mut hasher);
        match &m.content {
            MessageContent::Text(s) => {
                0u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            MessageContent::Parts(parts) => {
                1u8.hash(&mut hasher);
                // Serialize deterministically via serde_json so we capture the
                // full structure (text + image parts) without reaching into
                // each variant. Any serialisation failure collapses to a
                // tagged empty value — the hash stays deterministic for
                // identical inputs, which is all we need.
                let s = serde_json::to_string(parts).unwrap_or_else(|_| "<unhashable>".to_string());
                s.hash(&mut hasher);
            }
        }
    }
    Some(hasher.finish())
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Consistent-hashing pick: each candidate sits on a ring at
/// `hash(provider/model)`; the sticky key lands at `sticky_hash` and gets the
/// first candidate clockwise (wrapping around). Adding or removing a
/// candidate only reshuffles the keys that mapped to the removed slot — every
/// other key keeps its previous owner, which is the whole point of using a
/// ring instead of `sticky_hash % n`.
fn sticky_pick(candidates: &[&Backend], sticky_hash: u64) -> usize {
    debug_assert!(!candidates.is_empty(), "caller guarantees non-empty pool");
    let mut positions: Vec<(u64, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, b)| (backend_position(b), idx))
        .collect();
    positions.sort_by_key(|(h, _)| *h);
    positions
        .iter()
        .find(|(pos, _)| *pos >= sticky_hash)
        .copied()
        .unwrap_or(positions[0])
        .1
}

fn backend_position(b: &Backend) -> u64 {
    let mut hasher = DefaultHasher::new();
    b.provider.hash(&mut hasher);
    // Null byte separator so "a" + "bc" and "ab" + "c" don't collide.
    0u8.hash(&mut hasher);
    b.model.hash(&mut hasher);
    hasher.finish()
}

fn filter_candidates<'a>(
    config: &'a Config,
    registry: &'a ProviderRegistry,
    credentials: &'a CredentialStore,
    backends: &'a [Backend],
    requires_images: bool,
) -> Vec<&'a Backend> {
    backends
        .iter()
        .filter(|b| {
            if let Some(s) = config.providers.get(&b.provider)
                && !s.enabled
            {
                return false;
            }
            let Ok(provider) = registry.get(&b.provider) else {
                return false;
            };
            let settings = config
                .providers
                .get(&b.provider)
                .cloned()
                .unwrap_or_default();
            if !provider.is_authenticated(credentials, &settings) {
                return false;
            }
            if requires_images {
                let supports = provider
                    .models()
                    .into_iter()
                    .find(|m| m.id == b.model)
                    .map(|m| m.supports_images)
                    .unwrap_or(false);
                if !supports {
                    return false;
                }
            }
            true
        })
        .collect()
}

fn uniform_pick(len: usize) -> usize {
    rand::thread_rng().gen_range(0..len)
}

fn weighted_pick<F>(candidates: &[&Backend], weight_of: F) -> Result<usize>
where
    F: Fn(&Backend) -> f64,
{
    let weights: Vec<f64> = candidates.iter().map(|b| weight_of(b).max(0.0)).collect();
    weighted_index(&weights)
}

/// Sample an index from `[0, weights.len())` with probability proportional to
/// `weights[i]`. Falls back to uniform if the sum is zero or non-finite.
pub(crate) fn weighted_index(weights: &[f64]) -> Result<usize> {
    if weights.is_empty() {
        return Err(anyhow!("cannot pick from an empty candidate set"));
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Ok(uniform_pick(weights.len()));
    }
    let mut rng = rand::thread_rng();
    let mut target: f64 = rng.r#gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        target -= *w;
        if target <= 0.0 {
            return Ok(i);
        }
    }
    Ok(weights.len() - 1)
}

/// Usage-aware strategy: favour providers that have burned the *least* of
/// their plan so far, so consumption tends to even out.
///
/// For each candidate we ask the provider for a [`UsageSnapshot`]:
///   - `fraction_used = Some(f)` → weight = `max(1 - f, 0.05)` (5% floor so
///     exhausted plans still have a tail probability).
///   - `fraction_used = None`    → weight = `0.5` (halfway, so providers with
///     no telemetry don't crowd out providers we *know* are underused).
async fn usage_aware_pick(
    config: &Config,
    registry: &ProviderRegistry,
    credentials: &CredentialStore,
    candidates: &[&Backend],
) -> Result<usize> {
    let mut weights = Vec::with_capacity(candidates.len());
    for b in candidates {
        let settings = config
            .providers
            .get(&b.provider)
            .cloned()
            .unwrap_or_default();
        let w = match registry.get(&b.provider) {
            Ok(p) => match p.usage(credentials, &settings).await {
                Ok(Some(snap)) => match snap.fraction_used {
                    Some(f) => (1.0 - f).max(0.05),
                    None => 0.5,
                },
                Ok(None) => 0.5,
                Err(_) => 0.5,
            },
            Err(_) => 0.0,
        };
        weights.push(w);
    }
    weighted_index(&weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MixerModel, StickyConfig};
    use crate::openai::{ChatMessage, ChatRequest, MessageContent};
    use std::collections::HashMap;

    #[test]
    fn weighted_index_empty_errors() {
        assert!(weighted_index(&[]).is_err());
    }

    #[test]
    fn weighted_index_zero_total_falls_back_to_uniform() {
        let out = weighted_index(&[0.0, 0.0, 0.0]).unwrap();
        assert!(out < 3);
    }

    #[test]
    fn weighted_index_biases_toward_heavier_weight() {
        // Probability that 1000 draws from [100, 0.001] never pick index 0
        // is astronomically small.
        let mut zeros = 0;
        for _ in 0..200 {
            if weighted_index(&[100.0, 0.001]).unwrap() == 0 {
                zeros += 1;
            }
        }
        assert!(zeros > 180, "expected index 0 to dominate, got {zeros}/200");
    }

    // ── Sticky routing ──────────────────────────────────────────────────────

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(text.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: MessageContent::Text(text.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn system(text: &str) -> ChatMessage {
        ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(text.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn req_from(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: "mixer".to_string(),
            messages,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
            extra: Default::default(),
        }
    }

    fn mixer_model_with_sticky(key: StickyKey) -> MixerModel {
        MixerModel {
            description: String::new(),
            backends: Vec::new(),
            strategy: RoutingStrategy::Random,
            weights: HashMap::new(),
            sticky: Some(StickyConfig { enabled: true, key }),
        }
    }

    #[test]
    fn sticky_disabled_returns_none_key() {
        let model = MixerModel {
            sticky: Some(StickyConfig {
                enabled: false,
                key: StickyKey::MessagesHash,
            }),
            ..mixer_model_with_sticky(StickyKey::MessagesHash)
        };
        let req = req_from(vec![system("sys"), user("hi")]);
        assert!(compute_sticky_hash(&model, &req, &HeaderMap::new()).is_none());
    }

    #[test]
    fn sticky_unset_returns_none_key() {
        let mut model = mixer_model_with_sticky(StickyKey::MessagesHash);
        model.sticky = None;
        let req = req_from(vec![system("sys"), user("hi")]);
        assert!(compute_sticky_hash(&model, &req, &HeaderMap::new()).is_none());
    }

    #[test]
    fn messages_hash_is_deterministic_for_identical_inputs() {
        let model = mixer_model_with_sticky(StickyKey::MessagesHash);
        let req = req_from(vec![
            system("sys"),
            user("original ask"),
            assistant("reply"),
            user("follow up"),
        ]);
        let h1 = compute_sticky_hash(&model, &req, &HeaderMap::new()).unwrap();
        let h2 = compute_sticky_hash(&model, &req, &HeaderMap::new()).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn messages_hash_ignores_trailing_user_turn() {
        let model = mixer_model_with_sticky(StickyKey::MessagesHash);
        // Two requests with the same prefix but different trailing user
        // messages should hash to the same value — that's how we keep the
        // backend pinned as a conversation progresses.
        let prefix = vec![system("sys"), user("a"), assistant("b")];
        let mut m1 = prefix.clone();
        m1.push(user("first ask"));
        let mut m2 = prefix;
        m2.push(user("second ask, completely different"));
        let h1 = compute_sticky_hash(&model, &req_from(m1), &HeaderMap::new()).unwrap();
        let h2 = compute_sticky_hash(&model, &req_from(m2), &HeaderMap::new()).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn messages_hash_only_user_message_returns_none() {
        let model = mixer_model_with_sticky(StickyKey::MessagesHash);
        let req = req_from(vec![user("single turn")]);
        assert!(compute_sticky_hash(&model, &req, &HeaderMap::new()).is_none());
    }

    #[test]
    fn messages_hash_differs_for_different_prefixes() {
        let model = mixer_model_with_sticky(StickyKey::MessagesHash);
        let a = req_from(vec![system("sys A"), user("hi")]);
        let b = req_from(vec![system("sys B"), user("hi")]);
        let ha = compute_sticky_hash(&model, &a, &HeaderMap::new()).unwrap();
        let hb = compute_sticky_hash(&model, &b, &HeaderMap::new()).unwrap();
        assert_ne!(ha, hb);
    }

    #[test]
    fn header_key_extracts_named_header() {
        let model = mixer_model_with_sticky(StickyKey::Header("X-Session-Id".to_string()));
        let req = req_from(vec![user("hi")]);
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", "abc-123".parse().unwrap());
        let h1 = compute_sticky_hash(&model, &req, &headers).unwrap();

        // Same session id → same hash.
        let mut headers2 = HeaderMap::new();
        headers2.insert("X-Session-Id", "abc-123".parse().unwrap());
        let h2 = compute_sticky_hash(&model, &req, &headers2).unwrap();
        assert_eq!(h1, h2);

        // Different session id → different hash (astronomically unlikely to collide).
        let mut headers3 = HeaderMap::new();
        headers3.insert("x-session-id", "other".parse().unwrap());
        let h3 = compute_sticky_hash(&model, &req, &headers3).unwrap();
        assert_ne!(h1, h3);
    }

    #[test]
    fn header_key_missing_returns_none() {
        let model = mixer_model_with_sticky(StickyKey::Header("X-Session-Id".to_string()));
        let req = req_from(vec![user("hi")]);
        assert!(compute_sticky_hash(&model, &req, &HeaderMap::new()).is_none());
    }

    #[test]
    fn header_key_empty_returns_none() {
        let model = mixer_model_with_sticky(StickyKey::Header("X-Session-Id".to_string()));
        let req = req_from(vec![user("hi")]);
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", "".parse().unwrap());
        assert!(compute_sticky_hash(&model, &req, &headers).is_none());
    }

    #[test]
    fn sticky_pick_is_deterministic_for_same_key() {
        let pool = vec![
            Backend {
                provider: "a".to_string(),
                model: "m1".to_string(),
            },
            Backend {
                provider: "b".to_string(),
                model: "m2".to_string(),
            },
            Backend {
                provider: "c".to_string(),
                model: "m3".to_string(),
            },
        ];
        let refs: Vec<&Backend> = pool.iter().collect();
        let key = 12345_u64;
        let first = sticky_pick(&refs, key);
        for _ in 0..10 {
            assert_eq!(sticky_pick(&refs, key), first);
        }
    }

    #[test]
    fn sticky_pick_distributes_across_pool() {
        // Drive 1000 random-ish keys through a 3-backend pool and assert we
        // hit every backend at least once. Not a distribution test, just a
        // "consistent hashing doesn't collapse onto one backend" sanity check.
        let pool = vec![
            Backend {
                provider: "alpha".to_string(),
                model: "m".to_string(),
            },
            Backend {
                provider: "beta".to_string(),
                model: "m".to_string(),
            },
            Backend {
                provider: "gamma".to_string(),
                model: "m".to_string(),
            },
        ];
        let refs: Vec<&Backend> = pool.iter().collect();
        let mut counts = [0usize; 3];
        for i in 0..1000_u64 {
            let idx = sticky_pick(&refs, hash_bytes(&i.to_le_bytes()));
            counts[idx] += 1;
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "expected every backend to be hit at least once: {counts:?}",
        );
    }

    #[test]
    fn sticky_pick_single_backend_always_returns_index_zero() {
        let pool = vec![Backend {
            provider: "only".to_string(),
            model: "m".to_string(),
        }];
        let refs: Vec<&Backend> = pool.iter().collect();
        for k in [0_u64, 1, u64::MAX / 2, u64::MAX] {
            assert_eq!(sticky_pick(&refs, k), 0);
        }
    }

    // End-to-end: drive `pick` through the full filter+sticky flow with a
    // real provider registry and credential store, and assert 10 consecutive
    // identical requests all route to the same backend. This is the explicit
    // acceptance criterion for the sticky-routing feature.
    #[tokio::test]
    async fn pick_routes_ten_consecutive_identical_requests_to_same_backend() {
        use crate::providers::{AuthKind, ChatStream, ModelInfo, Provider, ProviderRegistry};
        use async_trait::async_trait;
        use std::sync::Arc;
        use tempfile::TempDir;

        struct AlwaysAuth(&'static str);

        #[async_trait]
        impl Provider for AlwaysAuth {
            fn id(&self) -> &'static str {
                self.0
            }
            fn display_name(&self) -> &'static str {
                self.0
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![ModelInfo {
                    id: "m",
                    display_name: "M",
                    supports_images: false,
                }]
            }
            fn auth_kind(&self) -> AuthKind {
                AuthKind::ApiKey
            }
            fn is_authenticated(
                &self,
                _store: &CredentialStore,
                _settings: &crate::config::ProviderSettings,
            ) -> bool {
                true
            }
            async fn login(&self, _store: &CredentialStore) -> anyhow::Result<()> {
                Ok(())
            }
            async fn chat_completion(
                &self,
                _store: &CredentialStore,
                _settings: &crate::config::ProviderSettings,
                _req: ChatRequest,
            ) -> anyhow::Result<ChatStream> {
                unreachable!("router tests never dispatch")
            }
        }

        let mut registry = ProviderRegistry::new();
        for id in ["alpha", "beta", "gamma"] {
            registry.register(Arc::new(AlwaysAuth(id)));
        }

        let mixer_model = MixerModel {
            description: String::new(),
            backends: vec![
                Backend {
                    provider: "alpha".to_string(),
                    model: "m".to_string(),
                },
                Backend {
                    provider: "beta".to_string(),
                    model: "m".to_string(),
                },
                Backend {
                    provider: "gamma".to_string(),
                    model: "m".to_string(),
                },
            ],
            strategy: RoutingStrategy::Random,
            weights: HashMap::new(),
            sticky: Some(crate::config::StickyConfig {
                enabled: true,
                key: StickyKey::MessagesHash,
            }),
        };

        let mut config = Config::default();
        config.providers.clear();
        for id in ["alpha", "beta", "gamma"] {
            config.providers.insert(
                id.to_string(),
                crate::config::ProviderSettings::default_enabled(),
            );
        }

        let tmp = TempDir::new().unwrap();
        let store = CredentialStore::with_dir_for_tests(tmp.path().to_path_buf());

        let req = req_from(vec![
            system("you are a helpful agent"),
            user("start session"),
            assistant("ok"),
            user("latest ask"),
        ]);
        let sticky_hash = compute_sticky_hash(&mixer_model, &req, &HeaderMap::new());
        assert!(
            sticky_hash.is_some(),
            "messages_hash should produce a key for a multi-turn conversation",
        );

        let mut seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let decision = pick(&config, &registry, &store, &mixer_model, false, sticky_hash)
                .await
                .expect("pick should succeed");
            seen.insert((decision.provider_id, decision.provider_model));
        }
        assert_eq!(
            seen.len(),
            1,
            "ten sticky picks with the same hash must pin to exactly one backend, got {seen:?}",
        );
    }

    #[test]
    fn sticky_pick_preserves_assignment_when_nonmatching_backend_removed() {
        // Consistent hashing property: removing a backend only reshuffles the
        // keys that previously pointed at *that* backend. Keys pointing at
        // other backends keep their mapping.
        let a = Backend {
            provider: "a".to_string(),
            model: "m".to_string(),
        };
        let b = Backend {
            provider: "b".to_string(),
            model: "m".to_string(),
        };
        let c = Backend {
            provider: "c".to_string(),
            model: "m".to_string(),
        };
        let full: Vec<&Backend> = vec![&a, &b, &c];

        // Find a key whose owner isn't the one we're about to drop.
        let mut hit = None;
        for k in 0..10_000_u64 {
            let key = hash_bytes(&k.to_le_bytes());
            let idx = sticky_pick(&full, key);
            if full[idx].provider != "c" {
                hit = Some((key, full[idx].provider.clone()));
                break;
            }
        }
        let (key, owner) = hit.expect("expected at least one key owned by a or b");

        let pruned: Vec<&Backend> = vec![&a, &b];
        let new_idx = sticky_pick(&pruned, key);
        assert_eq!(
            pruned[new_idx].provider, owner,
            "removing an unrelated backend must not reshuffle this key",
        );
    }
}
