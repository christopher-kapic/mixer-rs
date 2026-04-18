# mixer-rs — Multi-provider OpenAI-compatible LLM mixer

A Rust CLI that runs a local OpenAI-compatible HTTP endpoint and distributes each inference request across the LLM subscriptions the user is logged in to. Designed so clients (e.g. opencode) can transparently benefit from multi-model rotation and so users can drain multiple monthly plans at roughly the same rate.

## Tech Stack

- **Language:** Rust (edition 2024)
- **CLI:** clap v4 with derive macros + clap_complete
- **HTTP server:** axum (OpenAI-compatible endpoints)
- **HTTP client:** reqwest (rustls) for outbound provider calls
- **Async:** tokio (full runtime)
- **Trait objects:** async-trait (dyn-compatible `Provider`)
- **Serialization:** serde + serde_json
- **Platform dirs:** dirs crate (XDG-compliant)
- **Error handling:** anyhow

## Project Structure

```
src/
  main.rs               — Entry point, tokio runtime, clap dispatch
  cli.rs                — Clap command/arg definitions
  paths.rs              — XDG path resolution (config, credentials)
  config.rs             — Config + MixerModel + Backend + RoutingStrategy
  credentials.rs        — Per-provider credential storage (0600 JSON files)
  openai.rs             — OpenAI-compatible wire types; request_has_images
  usage.rs              — UsageSnapshot shared type
  concurrency.rs        — Per-provider tokio::sync::Semaphore caps
  router.rs             — Backend selection: filter + strategy dispatch
  server.rs             — Axum app: /v1/chat/completions, /v1/models
  providers/
    mod.rs              — Provider trait, ProviderRegistry, builtin_registry
    codex.rs            — Codex / ChatGPT Plus/Pro provider (stub)
    minimax.rs          — Minimax provider (stub)
    glm.rs              — GLM / z.ai provider (stub)
    opencode.rs         — opencode subscription provider (stub)
  commands/
    mod.rs              — Re-exports
    init.rs             — `mixer init`
    serve.rs            — `mixer serve`
    auth_cmd.rs         — `mixer auth login|logout|status`
    providers.rs        — `mixer providers list|show`
    models.rs           — `mixer models list|show`
    config_cmd.rs       — `mixer config show|edit|set|path`
    completions.rs      — `mixer completions <shell>`
```

## Key Design Decisions

- **XDG paths on every platform** (no `~/Library/Application Support` on macOS). Config lives at `~/.config/mixer/config.json`, credentials at `~/.config/mixer/credentials/<provider>.json`.
- **Credentials stored per-provider** as opaque `serde_json::Value` blobs (0600 on Unix). Each provider owns its own shape.
- **Provider trait is dyn-compatible via async-trait.** `Provider` is the plugin point — one file per provider under `src/providers/`.
- **Two-phase routing.** `router::pick` filters the pool to "eligible" backends (enabled + authenticated + capability-compatible) then applies the strategy. Image-bearing requests drop non-vision backends automatically.
- **Three strategies:** `random`, `weighted`, `usage-aware`. Usage-aware weights each backend by `max(1 - fraction_of_plan_used, 0.05)`, with a 0.5 fallback for providers that don't report usage.
- **Per-provider concurrency caps.** `max_concurrent_requests` in config maps to a `tokio::sync::Semaphore`; requests beyond the cap queue inside the server rather than being rejected. Useful for self-hosted endpoints.
- **Streaming is not yet wired.** `POST /v1/chat/completions` with `stream: true` returns 501 until SSE support is added.
- **`req.model` is a *mixer model name*, not a provider model.** The router rewrites it to the provider's native model id before dispatch.
- **OpenAI-style error bodies.** `AppError` renders `{"error": {"message", "type"}}` so OpenAI SDKs surface errors idiomatically.

## Adding a Provider

1. Create `src/providers/<name>.rs` with a struct implementing `Provider`.
2. Register it in `builtin_registry()` in `src/providers/mod.rs`.
3. Add it to the default `Config::default` providers map (optional — lazy inserts work too).

Everything else — filtering, weighting, usage-awareness, image routing, concurrency caps — flows from the trait automatically.

## Build & Test

```bash
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt -- --check
```

## Releasing

Releases are automated via `.github/workflows/release.yml`: bump `version` in `Cargo.toml`, merge to `master`, the workflow creates a GitHub Release and uploads cross-compiled binaries (`x86_64/aarch64 × linux-gnu/apple-darwin`) plus SHA-256 checksums. `scripts/install.sh` is the user-facing installer.

## Related Projects

- **kctx-local** (sibling at `../kctx-local/`) — Q&A CLI for codebases. Same Rust patterns.
- **ralph-rs** (sibling at `../ralph-rs/`) — Deterministic execution planner. Same Rust patterns.
