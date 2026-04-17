use anyhow::{Result, bail};
use serde_json::json;

use crate::cli::ModelsCommand;
use crate::config::Config;

pub fn run(cmd: &ModelsCommand) -> Result<()> {
    match cmd {
        ModelsCommand::List { verbose, json } => list(*verbose, *json),
        ModelsCommand::Show { name, json } => show(name, *json),
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
        let strategy = match m.strategy {
            crate::config::RoutingStrategy::Random => "random",
            crate::config::RoutingStrategy::Weighted => "weighted",
            crate::config::RoutingStrategy::UsageAware => "usage-aware",
        };
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
