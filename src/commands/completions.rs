use anyhow::Result;
use clap::CommandFactory;
use clap_complete::{Shell, generate};

use crate::cli::Cli;

pub fn run(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let mut out = std::io::stdout();
    generate(shell, &mut cmd, "mixer", &mut out);
    Ok(())
}
