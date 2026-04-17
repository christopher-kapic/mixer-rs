mod cli;
mod commands;
mod concurrency;
mod config;
mod credentials;
mod openai;
mod paths;
mod providers;
mod router;
mod server;
mod usage;

use clap::Parser;

use cli::{Cli, Command};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result: anyhow::Result<i32> = match cli.command {
        Command::Init { non_interactive } => commands::init::run(non_interactive).map(|()| 0),
        Command::Serve { addr, port, model } => {
            commands::serve::run(addr, port, model).await.map(|()| 0)
        }
        Command::Login { provider } => commands::login::run(&provider).await.map(|()| 0),
        Command::Logout { provider } => commands::logout::run(&provider).await.map(|()| 0),
        Command::Providers { command } => commands::providers::run(&command).await.map(|()| 0),
        Command::Models { command } => commands::models::run(&command).map(|()| 0),
        Command::Config { command } => commands::config_cmd::run(&command).map(|()| 0),
        Command::Completions { shell } => commands::completions::run(shell).map(|()| 0),
    };

    match result {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {:#}", e);
            std::process::exit(1);
        }
    }
}
