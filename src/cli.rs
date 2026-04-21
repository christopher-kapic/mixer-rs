use clap::{Parser, Subcommand, ValueEnum};
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
    /// Log output format (text = human-readable, json = one JSON object per line).
    /// `RUST_LOG` controls the filter; default is `mixer=info`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text, global = true)]
    pub log_format: LogFormat,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
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

    /// Validate config + probe each authenticated provider
    Doctor {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
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

    /// List the models this provider currently serves, by calling its live
    /// `/v1/models` endpoint (when supported). Falls back to the mixer's
    /// hardcoded catalogue for providers without a listing endpoint.
    Models {
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

    /// Create a new virtual mixer model with an empty backend pool
    Create {
        /// Mixer model name
        name: String,

        /// Human-readable description (shown by `mixer models list -v`)
        #[arg(long)]
        description: Option<String>,

        /// Routing strategy — defaults to `random`
        #[arg(long, value_enum)]
        strategy: Option<StrategyArg>,
    },

    /// Delete a mixer model (must not be the current default_model)
    Delete {
        /// Mixer model name
        name: String,
    },

    /// Add a provider/model backend to a mixer model's pool
    AddBackend {
        /// Mixer model name
        name: String,

        /// Provider id (must be a registered provider, e.g. `codex`)
        #[arg(long)]
        provider: String,

        /// Provider-native model id (must appear in that provider's catalogue)
        #[arg(long)]
        model: String,
    },

    /// Remove a provider/model backend from a mixer model's pool
    RemoveBackend {
        /// Mixer model name
        name: String,

        /// Provider id
        #[arg(long)]
        provider: String,

        /// Provider-native model id
        #[arg(long)]
        model: String,
    },

    /// Change a mixer model's routing strategy
    SetStrategy {
        /// Mixer model name
        name: String,

        /// New routing strategy
        #[arg(value_enum)]
        strategy: StrategyArg,
    },
}

/// CLI-facing routing-strategy enum. Lives here (not in `config.rs`) so the
/// config layer stays free of clap as a dependency. Converts into
/// [`crate::config::RoutingStrategy`] at the boundary.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyArg {
    Random,
    Weighted,
    UsageAware,
}

impl From<StrategyArg> for crate::config::RoutingStrategy {
    fn from(s: StrategyArg) -> Self {
        use crate::config::RoutingStrategy;
        match s {
            StrategyArg::Random => RoutingStrategy::Random,
            StrategyArg::Weighted => RoutingStrategy::Weighted,
            StrategyArg::UsageAware => RoutingStrategy::UsageAware,
        }
    }
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
