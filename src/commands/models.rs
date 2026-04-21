use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::json;

use crate::cli::{ModelsCommand, StrategyArg};
use crate::config::{Backend, Config, MixerModel, RoutingStrategy};
use crate::paths;
use crate::providers::builtin_registry;

pub fn run(cmd: &ModelsCommand) -> Result<()> {
    match cmd {
        ModelsCommand::List { verbose, json } => list(*verbose, *json),
        ModelsCommand::Show { name, json } => show(name, *json),
        ModelsCommand::Create {
            name,
            description,
            strategy,
        } => create(name, description.as_deref(), *strategy),
        ModelsCommand::Delete { name } => delete(name),
        ModelsCommand::AddBackend {
            name,
            provider,
            model,
        } => add_backend(name, provider, model),
        ModelsCommand::RemoveBackend {
            name,
            provider,
            model,
        } => remove_backend(name, provider, model),
        ModelsCommand::SetStrategy { name, strategy } => set_strategy(name, *strategy),
    }
}

fn list(verbose: bool, as_json: bool) -> Result<()> {
    let config = Config::load_or_default()?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&config.models)?);
        return Ok(());
    }

    let mut names: Vec<_> = config.models.keys().collect();
    names.sort();
    for n in names {
        let m = &config.models[n];
        let default_marker = if *n == config.default_model {
            " (default)"
        } else {
            ""
        };
        let strategy = strategy_label(m.strategy);
        println!(
            "{n:16} {strategy:12} {} backend(s){default_marker}",
            m.backends.len()
        );
        if verbose {
            if !m.description.is_empty() {
                println!("  {}", m.description);
            }
            for b in &m.backends {
                println!("  - {}/{}", b.provider, b.model);
            }
        }
    }
    Ok(())
}

fn show(name: &str, as_json: bool) -> Result<()> {
    let config = Config::load_or_default()?;
    let model = config
        .models
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("no mixer model named `{name}`"))?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "name": name,
                "is_default": name == config.default_model,
                "model": model,
            }))?
        );
    } else {
        println!("{}", serde_json::to_string_pretty(model)?);
    }

    // Surface zero-backend mixer models immediately.
    if model.backends.is_empty() {
        bail!("mixer model `{name}` has no backends configured");
    }
    Ok(())
}

fn create(name: &str, description: Option<&str>, strategy: Option<StrategyArg>) -> Result<()> {
    let (path, mut config) = load_for_mutation()?;
    apply_create(&mut config, name, description, strategy)?;
    config.save(&path)?;
    println!("Created mixer model `{name}`");
    Ok(())
}

fn delete(name: &str) -> Result<()> {
    let (path, mut config) = load_for_mutation()?;
    apply_delete(&mut config, name)?;
    config.save(&path)?;
    println!("Deleted mixer model `{name}`");
    Ok(())
}

fn add_backend(name: &str, provider: &str, model: &str) -> Result<()> {
    let (path, mut config) = load_for_mutation()?;
    let registry = builtin_registry();
    let prov = registry.get(provider)?;
    let valid: Vec<String> = prov
        .models()
        .into_iter()
        .map(|m| m.id.to_string())
        .collect();
    apply_add_backend(&mut config, name, provider, model, &valid)?;
    config.save(&path)?;
    println!("Added backend {provider}/{model} to mixer model `{name}`");
    Ok(())
}

fn remove_backend(name: &str, provider: &str, model: &str) -> Result<()> {
    let (path, mut config) = load_for_mutation()?;
    apply_remove_backend(&mut config, name, provider, model)?;
    config.save(&path)?;
    println!("Removed backend {provider}/{model} from mixer model `{name}`");
    Ok(())
}

fn set_strategy(name: &str, strategy: StrategyArg) -> Result<()> {
    let (path, mut config) = load_for_mutation()?;
    apply_set_strategy(&mut config, name, strategy)?;
    config.save(&path)?;
    println!(
        "Set strategy for mixer model `{name}` to {}",
        strategy_label(strategy.into())
    );
    Ok(())
}

/// Load the on-disk config (or the default if none exists) along with the path
/// the caller should write back to after mutating. Mirrors the pattern used by
/// `src/commands/config_cmd.rs::set`.
fn load_for_mutation() -> Result<(std::path::PathBuf, Config)> {
    let path = paths::config_file()?;
    let config = if path.exists() {
        Config::load(&path)?
    } else {
        Config::default()
    };
    Ok((path, config))
}

fn strategy_label(s: RoutingStrategy) -> &'static str {
    match s {
        RoutingStrategy::Random => "random",
        RoutingStrategy::Weighted => "weighted",
        RoutingStrategy::UsageAware => "usage-aware",
    }
}

// ---------------------------------------------------------------------------
// Pure mutators — split out so tests can exercise the logic without touching
// the real config file on disk.
// ---------------------------------------------------------------------------

fn apply_create(
    config: &mut Config,
    name: &str,
    description: Option<&str>,
    strategy: Option<StrategyArg>,
) -> Result<()> {
    if name.is_empty() {
        bail!("mixer model name must not be empty");
    }
    if config.models.contains_key(name) {
        bail!("mixer model `{name}` already exists (use `mixer models show {name}` to inspect it)");
    }
    config.models.insert(
        name.to_string(),
        MixerModel {
            description: description.unwrap_or("").to_string(),
            backends: Vec::new(),
            strategy: strategy.map(Into::into).unwrap_or_default(),
            weights: HashMap::new(),
            sticky: None,
        },
    );
    Ok(())
}

fn apply_delete(config: &mut Config, name: &str) -> Result<()> {
    if !config.models.contains_key(name) {
        bail!("no mixer model named `{name}`");
    }
    if config.default_model == name {
        bail!(
            "refusing to delete mixer model `{name}` because it is the current default_model — \
             set another default first with `mixer config set default_model <other>`"
        );
    }
    config.models.remove(name);
    Ok(())
}

fn apply_add_backend(
    config: &mut Config,
    name: &str,
    provider: &str,
    model: &str,
    valid_models: &[String],
) -> Result<()> {
    if !valid_models.iter().any(|id| id == model) {
        let list = if valid_models.is_empty() {
            "(none)".to_string()
        } else {
            valid_models.join(", ")
        };
        bail!("provider `{provider}` has no model `{model}`; valid ids for this provider: {list}");
    }
    let mm = config
        .models
        .get_mut(name)
        .ok_or_else(|| anyhow::anyhow!("no mixer model named `{name}`"))?;
    if mm
        .backends
        .iter()
        .any(|b| b.provider == provider && b.model == model)
    {
        bail!("mixer model `{name}` already routes to {provider}/{model}");
    }
    mm.backends.push(Backend {
        provider: provider.to_string(),
        model: model.to_string(),
    });
    Ok(())
}

fn apply_remove_backend(
    config: &mut Config,
    name: &str,
    provider: &str,
    model: &str,
) -> Result<()> {
    let mm = config
        .models
        .get_mut(name)
        .ok_or_else(|| anyhow::anyhow!("no mixer model named `{name}`"))?;
    let before = mm.backends.len();
    mm.backends
        .retain(|b| !(b.provider == provider && b.model == model));
    if mm.backends.len() == before {
        bail!("mixer model `{name}` has no backend {provider}/{model}");
    }
    Ok(())
}

fn apply_set_strategy(config: &mut Config, name: &str, strategy: StrategyArg) -> Result<()> {
    let mm = config
        .models
        .get_mut(name)
        .ok_or_else(|| anyhow::anyhow!("no mixer model named `{name}`"))?;
    mm.strategy = strategy.into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_models(provider: &str) -> Vec<String> {
        builtin_registry()
            .get(provider)
            .unwrap()
            .models()
            .into_iter()
            .map(|m| m.id.to_string())
            .collect()
    }

    #[test]
    fn create_inserts_with_defaults() {
        let mut c = Config::default();
        apply_create(&mut c, "fresh", None, None).unwrap();
        let m = c.models.get("fresh").unwrap();
        assert_eq!(m.strategy, RoutingStrategy::Random);
        assert!(m.backends.is_empty());
        assert_eq!(m.description, "");
    }

    #[test]
    fn create_honors_description_and_strategy() {
        let mut c = Config::default();
        apply_create(
            &mut c,
            "weighted-pool",
            Some("hand-tuned"),
            Some(StrategyArg::Weighted),
        )
        .unwrap();
        let m = c.models.get("weighted-pool").unwrap();
        assert_eq!(m.description, "hand-tuned");
        assert_eq!(m.strategy, RoutingStrategy::Weighted);
    }

    #[test]
    fn create_rejects_duplicate_name() {
        let mut c = Config::default();
        // `mixer` is seeded by Config::default
        let err = apply_create(&mut c, "mixer", None, None).unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err:#}");
    }

    #[test]
    fn create_rejects_empty_name() {
        let mut c = Config::default();
        assert!(apply_create(&mut c, "", None, None).is_err());
    }

    #[test]
    fn delete_removes_non_default_model() {
        let mut c = Config::default();
        apply_create(&mut c, "scratch", None, None).unwrap();
        apply_delete(&mut c, "scratch").unwrap();
        assert!(!c.models.contains_key("scratch"));
    }

    #[test]
    fn delete_refuses_current_default_model() {
        let mut c = Config::default();
        assert_eq!(c.default_model, "mixer");
        let err = apply_delete(&mut c, "mixer").unwrap_err();
        assert!(
            err.to_string().contains("default_model"),
            "expected guard message, got: {err:#}"
        );
        // And the model must still be present.
        assert!(c.models.contains_key("mixer"));
    }

    #[test]
    fn delete_errors_on_unknown_model() {
        let mut c = Config::default();
        let err = apply_delete(&mut c, "ghost").unwrap_err();
        assert!(err.to_string().contains("no mixer model named"));
    }

    #[test]
    fn add_backend_validates_against_catalogue() {
        let mut c = Config::default();
        apply_create(&mut c, "primary", None, None).unwrap();
        let valid = registry_models("codex");
        // Pick a model id we know isn't in the codex catalogue.
        let err = apply_add_backend(&mut c, "primary", "codex", "definitely-not-a-model", &valid)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("valid ids"), "got: {msg}");
        // And the pool must still be empty.
        assert!(c.models["primary"].backends.is_empty());
    }

    #[test]
    fn add_backend_happy_path_and_dedupe() {
        let mut c = Config::default();
        apply_create(&mut c, "primary", None, None).unwrap();
        let valid = registry_models("codex");
        let pick = valid.first().expect("codex has at least one model").clone();
        apply_add_backend(&mut c, "primary", "codex", &pick, &valid).unwrap();
        assert_eq!(c.models["primary"].backends.len(), 1);
        let err = apply_add_backend(&mut c, "primary", "codex", &pick, &valid).unwrap_err();
        assert!(err.to_string().contains("already routes"));
        assert_eq!(c.models["primary"].backends.len(), 1);
    }

    #[test]
    fn add_backend_errors_on_unknown_mixer_model() {
        let mut c = Config::default();
        let valid = registry_models("codex");
        let pick = valid.first().unwrap().clone();
        let err = apply_add_backend(&mut c, "ghost", "codex", &pick, &valid).unwrap_err();
        assert!(err.to_string().contains("no mixer model named"));
    }

    #[test]
    fn remove_backend_removes_and_leaves_empty_pool() {
        let mut c = Config::default();
        // `mixer` seeded with 5 backends by default.
        let start = c.models["mixer"].backends.len();
        apply_remove_backend(&mut c, "mixer", "codex", "gpt-5.4").unwrap();
        assert_eq!(c.models["mixer"].backends.len(), start - 1);
        // Drain the rest — the last removal must leave an empty pool, not
        // auto-delete the model.
        let remaining: Vec<_> = c.models["mixer"].backends.clone();
        for b in remaining {
            apply_remove_backend(&mut c, "mixer", &b.provider, &b.model).unwrap();
        }
        assert!(c.models.contains_key("mixer"));
        assert!(c.models["mixer"].backends.is_empty());
    }

    #[test]
    fn remove_backend_errors_on_missing_entry() {
        let mut c = Config::default();
        let err = apply_remove_backend(&mut c, "mixer", "codex", "not-a-real-model").unwrap_err();
        assert!(err.to_string().contains("has no backend"));
    }

    #[test]
    fn set_strategy_updates_and_errors_on_unknown() {
        let mut c = Config::default();
        apply_set_strategy(&mut c, "mixer", StrategyArg::UsageAware).unwrap();
        assert_eq!(c.models["mixer"].strategy, RoutingStrategy::UsageAware);
        assert!(apply_set_strategy(&mut c, "ghost", StrategyArg::Random).is_err());
    }
}
