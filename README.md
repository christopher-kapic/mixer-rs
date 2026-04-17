# mixer-rs

A Rust CLI that runs a local OpenAI-compatible HTTP endpoint and distributes chat completion requests across the LLM subscriptions you are logged in to (Codex / ChatGPT, Minimax, GLM / z.ai, opencode, …). The idea: point any agentic tool that speaks OpenAI (e.g. opencode) at `http://127.0.0.1:4141` and let mixer fan each request out to a random provider — weighted by policy, by plan consumption, or by what's left in each monthly cap.

## Why

- **Multi-model diversity.** There's a theory that agent traces improve when the underlying LLMs are rotated across each turn. mixer makes that rotation transparent to the client.
- **Burn plans evenly.** If you pay for four subscriptions, you want to consume them at similar rates. mixer's usage-aware strategy weights each request by how much of each plan is left.
- **One endpoint, many backends.** Clients keep using `https://api.openai.com/v1/chat/completions`-shaped URLs — no client-side plugin wrangling.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/christopher-kapic/mixer-rs/master/scripts/install.sh | bash
```

Or build from source:

```bash
cargo install --path .
```

## Quick Start

```bash
mixer init                          # write ~/.config/mixer/config.json
mixer login codex                   # + login minimax / glm / opencode
mixer providers list                # check who's authenticated
mixer serve                         # http://127.0.0.1:4141
```

Then point your OpenAI-compatible client at `http://127.0.0.1:4141/v1` with `model: "mixer"` (or any other mixer model you've defined).

## Mixer models

A *mixer model* is a virtual model name clients request (`model: "mixer"`) that maps to a pool of `(provider, provider_model)` backends plus a routing strategy. Define as many as you want in `config.json`:

```json
{
  "default_model": "mixer",
  "models": {
    "mixer": {
      "description": "Rotate across every authenticated subscription",
      "strategy": "usage-aware",
      "backends": [
        { "provider": "codex", "model": "gpt-5.2" },
        { "provider": "minimax", "model": "MiniMax-M2" },
        { "provider": "glm", "model": "glm-4.6" },
        { "provider": "opencode", "model": "anthropic/claude-sonnet-4-6" }
      ]
    },
    "vision": {
      "description": "Image-capable only",
      "strategy": "weighted",
      "weights": { "codex": 2, "opencode": 1 },
      "backends": [
        { "provider": "codex", "model": "gpt-5.2" },
        { "provider": "opencode", "model": "anthropic/claude-sonnet-4-6" }
      ]
    }
  }
}
```

### Routing strategies

- `random` — uniform random across available backends.
- `weighted` — weighted random using per-provider weights you set.
- `usage-aware` — weighted random where weight ∝ `(1 − fraction_of_plan_used)`, so providers further from exhausting their monthly plan are preferred. Providers that don't report usage fall back to a neutral weight. Lets you naturally drain all your plans at similar rates.

### Request-driven filtering

Before applying the strategy, mixer filters the backend pool to what can actually handle the request:

- If any message carries an `image_url` content part, backends whose model doesn't support images are dropped.
- Backends whose provider is `enabled: false`, disabled, or not logged in are dropped.

## Per-provider concurrency caps

Each provider entry in config can set `max_concurrent_requests` — useful when one of your backends is a self-hosted model that only wants to service N requests at a time. Requests beyond the cap queue inside mixer (via a `tokio::sync::Semaphore`); they aren't rejected.

```bash
mixer config set providers.selfhost.max_concurrent_requests 2
mixer config set providers.selfhost.max_concurrent_requests none   # clear
```

## Adding a provider

One file per provider under `src/providers/`. Drop in `src/providers/myprovider.rs`, implement the `Provider` trait (models, login, chat_completion, optional usage), then register it in `builtin_registry()` in `src/providers/mod.rs`. That's it — routing, auth storage, usage weighting, image filtering, and concurrency caps work automatically.

```rust
pub struct MyProvider;

#[async_trait::async_trait]
impl Provider for MyProvider {
    fn id(&self) -> &'static str { "myprovider" }
    fn display_name(&self) -> &'static str { "My Provider" }
    fn models(&self) -> Vec<ModelInfo> { /* ... */ vec![] }
    async fn login(&self, store: &CredentialStore) -> Result<()> { /* ... */ Ok(()) }
    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatResponse> { /* ... */ todo!() }
}
```

PRs for new providers welcome.

## Build & Test

```bash
cargo build
cargo test
cargo clippy -- -D warnings
```

## License

MIT
