use clap::{Parser, Subcommand};
use clap_complete::Shell;

#[derive(Parser)]
#[command(
    name = "mixer",
    version,
    about = "Mix LLM inference across subscription-based providers",
    long_about = "mixer runs a local OpenAI-compatible HTTP endpoint that distributes chat \
completion requests across the LLM subscriptions you are logged in to (Codex, Minimax, GLM, \
opencode, etc.). Each request is routed to a randomly-chosen provider — optionally weighted by \
your configured policy or by how much of each monthly plan you've already consumed — so that \
agentic tools that hit the endpoint get the benefit of multi-model inference without managing \
keys or quotas themselves."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Initialize mixer (writes config, creates credential directory)
    Init {
        /// Skip interactive prompts
        #[arg(long)]
        non_interactive: bool,
    },

    /// Start the OpenAI-compatible HTTP server
    Serve {
        /// Override the listen address (e.g. `127.0.0.1:4141`)
        #[arg(long)]
        addr: Option<String>,

        /// Override the listen port (kept if `--addr` is not given)
        #[arg(long, short)]
        port: Option<u16>,

        /// Only serve this mixer model name (default serves all configured models)
        #[arg(long)]
        model: Option<String>,
    },

    /// Manage provider authentication (login, logout, status)
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },

    /// Inspect providers
    Providers {
        #[command(subcommand)]
        command: ProvidersCommand,
    },

    /// Inspect virtual mixer models (pools of provider/model backends)
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },

    /// View/edit configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
}

#[derive(Subcommand)]
pub enum AuthCommand {
    /// Log in to a provider subscription
    Login {
        /// Provider id (e.g. `codex`, `minimax`, `glm`, `opencode`)
        provider: String,
    },

    /// Log out of a provider subscription
    Logout {
        /// Provider id
        provider: String,
    },

    /// Show auth status for one or all providers
    Status {
        /// Provider id (omit to show all)
        provider: Option<String>,

        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ProvidersCommand {
    /// List all registered providers, highlighting which are authenticated
    List {
        #[arg(short, long)]
        verbose: bool,

        #[arg(long)]
        json: bool,
    },

    /// Show a provider's metadata, models, and current usage
    Show {
        /// Provider id
        name: String,

        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// List all virtual mixer models defined in config
    List {
        #[arg(short, long)]
        verbose: bool,

        #[arg(long)]
        json: bool,
    },

    /// Show a single mixer model's backend pool and routing strategy
    Show {
        /// Mixer model name
        name: String,

        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Print current config
    Show {
        #[arg(long)]
        json: bool,
    },

    /// Open config in $EDITOR
    Edit,

    /// Set a config value (dot-notation key)
    Set { key: String, value: String },

    /// Print config file path
    Path,
}
