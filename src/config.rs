use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Top-level mixer configuration, stored at `~/.config/mixer/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// Address the HTTP server binds to.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Name of the default mixer model used when a request does not specify one,
    /// or specifies a model name that isn't defined.
    #[serde(default = "default_default_model")]
    pub default_model: String,

    /// Virtual mixer models, keyed by the name an OpenAI-compatible client
    /// will pass in the `model` field.
    #[serde(default)]
    pub models: HashMap<String, MixerModel>,

    /// Per-provider settings (authentication is stored separately under
    /// `credentials/`; these are non-secret behavioural overrides).
    #[serde(default)]
    pub providers: HashMap<String, ProviderSettings>,

    /// Name of an environment variable holding a shared-secret bearer token
    /// that clients must present in `Authorization: Bearer <token>` on every
    /// `/v1/*` request. Only the env var *name* is stored here — never the
    /// token itself (same rationale as `api_key_env`, plan.md §3.6.2). When
    /// unset, or when the env var resolves to an empty string, the local
    /// endpoint accepts unauthenticated requests (preserves today's default
    /// loopback-only behaviour).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listen_bearer_token_env: Option<String>,
}

/// A virtual mixer model — a pool of concrete provider/model backends plus
/// a routing strategy that picks one per request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MixerModel {
    /// Human-readable description shown in `mixer models list`.
    #[serde(default)]
    pub description: String,

    /// The pool of backends this mixer model may route to.
    pub backends: Vec<Backend>,

    /// Routing strategy used to pick one backend per request.
    #[serde(default)]
    pub strategy: RoutingStrategy,

    /// Optional per-provider weights used when `strategy = "weighted"`.
    /// Keyed by provider id. Providers absent from this map fall back to 1.0.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub weights: HashMap<String, f64>,

    /// Optional sticky-session mode: pin a given conversation (or a
    /// client-supplied session header) to a single backend via consistent
    /// hashing over the eligible pool. Preserves KV-cache locality at the
    /// cost of even distribution. Unset / `enabled: false` preserves today's
    /// stateless per-request routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sticky: Option<StickyConfig>,
}

/// Sticky-session policy for a mixer model. See [`MixerModel::sticky`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StickyConfig {
    /// When false, sticky routing is off and the model uses its normal
    /// strategy.
    #[serde(default)]
    pub enabled: bool,

    /// What to derive the sticky key from. See [`StickyKey`].
    pub key: StickyKey,
}

/// How to compute a sticky-session key from the incoming request.
///
/// Serialised as a string to match the config shape documented in plan.md §10:
///   * `"messages_hash"` — hash the conversation prefix (every message except
///     the trailing run of user messages). Deterministic for a given input,
///     which lets identical conversation snapshots route to the same backend.
///   * `"header:<Name>"` — pull a named header from the request (e.g.
///     `"header:X-Session-Id"`); absent or empty header falls through to the
///     model's normal strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StickyKey {
    MessagesHash,
    Header(String),
}

impl Serialize for StickyKey {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            StickyKey::MessagesHash => ser.serialize_str("messages_hash"),
            StickyKey::Header(name) => ser.serialize_str(&format!("header:{name}")),
        }
    }
}

impl<'de> Deserialize<'de> for StickyKey {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(de)?;
        if raw == "messages_hash" {
            return Ok(StickyKey::MessagesHash);
        }
        if let Some(name) = raw.strip_prefix("header:") {
            if name.is_empty() {
                return Err(D::Error::custom(
                    "sticky key `header:` requires a header name after the colon",
                ));
            }
            return Ok(StickyKey::Header(name.to_string()));
        }
        Err(D::Error::custom(format!(
            "unknown sticky key `{raw}` (expected `messages_hash` or `header:<Name>`)"
        )))
    }
}

/// A single concrete backend: a provider plus one of that provider's models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Backend {
    /// Provider id (must match a registered provider, e.g. `codex`).
    pub provider: String,

    /// The provider-native model identifier (e.g. `gpt-5.2`, `glm-4.6`).
    pub model: String,
}

/// How a mixer model picks a backend per request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum RoutingStrategy {
    /// Uniformly random across all *available* backends (auth'd + capable
    /// of handling the request).
    #[default]
    Random,

    /// Weighted random using `weights`.
    Weighted,

    /// Weight each available backend by how *underused* its subscription is:
    /// providers further from exhausting their monthly quota are preferred,
    /// so consumption naturally evens out. Providers that don't report usage
    /// fall back to equal weight.
    UsageAware,
}

/// Non-secret, per-provider settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderSettings {
    /// When false, the provider is skipped even if authenticated.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Optional override for the provider's base URL (useful for self-hosted
    /// endpoints or regional mirrors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Caps the number of in-flight requests mixer will dispatch to this
    /// provider concurrently. Requests beyond the cap queue inside the
    /// mixer server; they are not rejected. Useful for self-hosted models
    /// that can only service N requests at a time. `None` means unlimited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests: Option<u32>,

    /// Seconds before a request to this provider is aborted. `None` uses
    /// the client default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timeout_secs: Option<u64>,

    /// Name of an environment variable to read the provider's API key from.
    /// When set and non-empty at request time, this takes precedence over any
    /// stored credential file. Only meaningful for API-key providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
}

fn default_listen_addr() -> String {
    "127.0.0.1:4141".to_string()
}

fn default_default_model() -> String {
    "mixer".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert(
            "mixer".to_string(),
            MixerModel {
                description: "Default pool — randomly routes to every authenticated subscription"
                    .to_string(),
                backends: vec![
                    Backend {
                        provider: "codex".to_string(),
                        model: "gpt-5.2".to_string(),
                    },
                    Backend {
                        provider: "minimax".to_string(),
                        model: "MiniMax-M2".to_string(),
                    },
                    Backend {
                        provider: "glm".to_string(),
                        model: "glm-4.6".to_string(),
                    },
                    Backend {
                        provider: "opencode".to_string(),
                        model: "anthropic/claude-sonnet-4-6".to_string(),
                    },
                    Backend {
                        provider: "kimi-code".to_string(),
                        model: "kimi-k2.6".to_string(),
                    },
                ],
                strategy: RoutingStrategy::Random,
                weights: HashMap::new(),
                sticky: None,
            },
        );

        let mut providers = HashMap::new();
        for id in ["codex", "minimax", "glm", "opencode", "kimi-code"] {
            providers.insert(id.to_string(), ProviderSettings::default_enabled());
        }
        // `kimi-api` is the pay-per-token backup to `kimi-code`. Off by
        // default so the subscription path is the advertised one; users flip
        // `enabled` if they want to add pay-per-token Kimi to their pool.
        providers.insert(
            "kimi-api".to_string(),
            ProviderSettings {
                enabled: false,
                ..ProviderSettings::default_enabled()
            },
        );
        // Self-hosted ollama is opt-in: disabled by default, with a sensible
        // concurrency cap for GPU-constrained hosts. Users flip `enabled` once
        // they have a local ollama server running.
        providers.insert(
            "ollama".to_string(),
            ProviderSettings {
                enabled: false,
                max_concurrent_requests: Some(2),
                ..ProviderSettings::default_enabled()
            },
        );

        Self {
            listen_addr: default_listen_addr(),
            default_model: default_default_model(),
            models,
            providers,
            listen_bearer_token_env: None,
        }
    }
}

impl ProviderSettings {
    pub fn default_enabled() -> Self {
        Self {
            enabled: true,
            base_url: None,
            max_concurrent_requests: None,
            request_timeout_secs: None,
            api_key_env: None,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        if contents.trim().is_empty() {
            return Ok(Self::default());
        }
        let config: Config = serde_json::from_str(&contents)
            .with_context(|| format!("parsing {}", path.display()))?;
        Ok(config)
    }

    pub fn load_or_default() -> Result<Self> {
        let path = crate::paths::config_file()?;
        if path.exists() {
            Self::load(&path)
        } else {
            Ok(Self::default())
        }
    }

    /// Atomic save — write to a temp file in the same directory, then rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path
            .parent()
            .context("config path has no parent directory")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory {}", parent.display()))?;

        let json = serde_json::to_string_pretty(self).context("serializing config")?;

        let mut tmp = tempfile::NamedTempFile::new_in(parent)
            .with_context(|| format!("creating temp file in {}", parent.display()))?;
        tmp.write_all(json.as_bytes())
            .context("writing config to temp file")?;
        tmp.flush().context("flushing config temp file")?;
        tmp.persist(path)
            .with_context(|| format!("persisting config to {}", path.display()))?;
        Ok(())
    }

    /// Resolve a mixer model name, falling back to `default_model` when the
    /// caller-supplied name is empty or not defined.
    pub fn resolve_model(&self, requested: &str) -> Option<(&str, &MixerModel)> {
        if let Some((k, v)) = self.models.get_key_value(requested) {
            return Some((k.as_str(), v));
        }
        self.models
            .get_key_value(self.default_model.as_str())
            .map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_includes_mixer_model() {
        let c = Config::default();
        assert!(c.models.contains_key("mixer"));
        assert_eq!(c.default_model, "mixer");
    }

    #[test]
    fn default_enables_subscription_providers() {
        let c = Config::default();
        for id in ["codex", "minimax", "glm", "opencode", "kimi-code"] {
            let s = c
                .providers
                .get(id)
                .unwrap_or_else(|| panic!("missing {id}"));
            assert!(s.enabled, "{id} should be enabled by default");
            assert!(
                s.max_concurrent_requests.is_none(),
                "{id} should be uncapped by default",
            );
        }
    }

    #[test]
    fn default_disables_kimi_api_pay_per_token_path() {
        let c = Config::default();
        let kimi_api = c
            .providers
            .get("kimi-api")
            .expect("kimi-api should be in default providers map");
        assert!(
            !kimi_api.enabled,
            "kimi-api is opt-in (backup to kimi-code)"
        );
    }

    #[test]
    fn default_disables_self_hosted_ollama_with_concurrency_cap() {
        let c = Config::default();
        let ollama = c
            .providers
            .get("ollama")
            .expect("ollama should be in default providers map");
        assert!(!ollama.enabled, "ollama is opt-in");
        assert_eq!(
            ollama.max_concurrent_requests,
            Some(2),
            "default cap guards GPU-constrained hosts",
        );
    }

    #[test]
    fn routing_strategy_default_is_random() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::Random);
    }

    #[test]
    fn roundtrip_json() {
        let c = Config::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn resolve_model_falls_back_to_default() {
        let c = Config::default();
        let (name, _) = c.resolve_model("nonexistent").unwrap();
        assert_eq!(name, "mixer");
    }

    #[test]
    fn resolve_model_returns_exact_match() {
        let mut c = Config::default();
        c.models.insert(
            "vision".to_string(),
            MixerModel {
                description: String::new(),
                backends: vec![Backend {
                    provider: "codex".to_string(),
                    model: "gpt-5.2".to_string(),
                }],
                strategy: RoutingStrategy::Random,
                weights: HashMap::new(),
                sticky: None,
            },
        );
        let (name, _) = c.resolve_model("vision").unwrap();
        assert_eq!(name, "vision");
    }

    #[test]
    fn sticky_key_messages_hash_round_trips() {
        let raw = r#""messages_hash""#;
        let key: StickyKey = serde_json::from_str(raw).unwrap();
        assert_eq!(key, StickyKey::MessagesHash);
        assert_eq!(serde_json::to_string(&key).unwrap(), raw);
    }

    #[test]
    fn sticky_key_header_round_trips() {
        let raw = r#""header:X-Session-Id""#;
        let key: StickyKey = serde_json::from_str(raw).unwrap();
        assert_eq!(key, StickyKey::Header("X-Session-Id".to_string()));
        assert_eq!(serde_json::to_string(&key).unwrap(), raw);
    }

    #[test]
    fn sticky_key_rejects_unknown_and_empty_header_name() {
        assert!(serde_json::from_str::<StickyKey>(r#""nope""#).is_err());
        assert!(serde_json::from_str::<StickyKey>(r#""header:""#).is_err());
    }

    #[test]
    fn mixer_model_parses_sticky_block() {
        let raw = r#"{
            "backends": [{"provider":"codex","model":"gpt-5.2"}],
            "sticky": {"enabled": true, "key": "messages_hash"}
        }"#;
        let m: MixerModel = serde_json::from_str(raw).unwrap();
        let s = m.sticky.expect("sticky should be present");
        assert!(s.enabled);
        assert_eq!(s.key, StickyKey::MessagesHash);
    }

    #[test]
    fn parses_max_concurrent_requests() {
        let raw = r#"{
            "providers": {
                "selfhost": { "enabled": true, "max_concurrent_requests": 2 }
            }
        }"#;
        let c: Config = serde_json::from_str(raw).unwrap();
        assert_eq!(c.providers["selfhost"].max_concurrent_requests, Some(2));
    }
}
