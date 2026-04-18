//! Backend selection: given a mixer model + an inbound request, pick which
//! concrete `(provider, provider_model)` should service it.
//!
//! Selection is a two-phase process:
//!
//! 1. **Filter** the backend pool down to candidates that are actually
//!    usable for this request: the provider must be registered, enabled in
//!    config, authenticated, and — if the request carries an image — the
//!    backend model must report `supports_images = true`.
//! 2. **Pick** one candidate using the mixer model's [`RoutingStrategy`]:
//!    uniformly random, weighted random, or usage-aware (weight inversely
//!    proportional to how much of each plan has been consumed, so we drain
//!    plans evenly).

use anyhow::{Result, anyhow, bail};
use rand::Rng;

use crate::config::{Backend, Config, MixerModel, RoutingStrategy};
use crate::credentials::CredentialStore;
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
) -> Result<RouteDecision> {
    pick_excluding(
        config,
        registry,
        credentials,
        mixer_model,
        requires_images,
        &[],
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

    let idx = match mixer_model.strategy {
        RoutingStrategy::Random => uniform_pick(candidates.len()),
        RoutingStrategy::Weighted => weighted_pick(&candidates, |b| {
            *mixer_model.weights.get(&b.provider).unwrap_or(&1.0)
        })?,
        RoutingStrategy::UsageAware => {
            usage_aware_pick(config, registry, credentials, &candidates).await?
        }
    };

    let chosen = &candidates[idx];
    Ok(RouteDecision {
        provider_id: chosen.provider.clone(),
        provider_model: chosen.model.clone(),
    })
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
}
