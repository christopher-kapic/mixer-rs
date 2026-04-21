use anyhow::{Context, Result};

use crate::config::Config;
use crate::paths;

pub fn run(non_interactive: bool) -> Result<()> {
    let _ = non_interactive; // reserved for future interactive prompts

    let config_path = paths::config_file()?;
    let creds_dir = paths::credentials_dir()?;

    let existed = config_path.exists();
    let config = if existed {
        Config::load(&config_path)?
    } else {
        Config::default()
    };
    config.save(&config_path)?;

    std::fs::create_dir_all(&creds_dir)
        .with_context(|| format!("creating {}", creds_dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&creds_dir, std::fs::Permissions::from_mode(0o700));
    }

    eprintln!("mixer initialized");
    eprintln!();
    if existed {
        eprintln!(
            "  Config:       {} (existing — not overwritten)",
            config_path.display()
        );
    } else {
        eprintln!("  Config:       {} (created)", config_path.display());
    }
    eprintln!("  Credentials:  {}", creds_dir.display());
    eprintln!("  Listen addr:  {}", config.listen_addr);
    eprintln!("  Default:      {}", config.default_model);
    eprintln!();
    eprintln!("Next steps:");
    eprintln!("  mixer auth login <provider>   — for each subscription you want to mix");
    eprintln!("  mixer providers list     — see auth state and usage");
    eprintln!("  mixer serve              — start the OpenAI-compatible endpoint");
    Ok(())
}
