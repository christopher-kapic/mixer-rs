# mixer-rs

A Rust CLI that runs a local OpenAI-compatible HTTP endpoint and distributes chat completion requests across the LLM subscriptions you are logged in to (Codex / ChatGPT, Minimax, GLM / z.ai, opencode, …). The idea: point any agentic tool that speaks OpenAI (e.g. opencode) at `http://127.0.0.1:4141` and let mixer fan each request out to a random provider — weighted by policy, by plan consumption, or by what's left in each monthly cap.

## Why

- **Multi-model diversity.** There's a theory that agent traces improve when the underlying LLMs are rotated across each turn. mixer makes that rotation transparent to the client.
- **Burn plans evenly.** If you pay for four subscriptions, you want to consume them at similar rates. mixer's usage-aware strategy weights each request by how much of each plan is left, on a best-effort basis — each provider reports as much plan telemetry as its API exposes (see the provider table below).
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
mixer auth login codex              # + login minimax / glm / opencode
mixer auth status                   # check who's authenticated
mixer doctor                        # validate config + authenticated backends
mixer providers list                # provider metadata + usage
mixer serve                         # http://127.0.0.1:4141
```

`mixer doctor` avoids live provider calls by default. To include chat and usage probes against authenticated providers, run it with `MIXER_DOCTOR_LIVE=1`.

Then point your OpenAI-compatible client at `http://127.0.0.1:4141/v1` with `model: "mixer"` (or any other mixer model you've defined). A minimal smoke test:

```bash
curl http://127.0.0.1:4141/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mixer","messages":[{"role":"user","content":"Say ok"}]}'
```

## Providers

| Provider | Auth | Default models | Vision | Usage telemetry | Notes |
|---|---|---|---|---|---|
| `codex` | OAuth device flow (ChatGPT Plus/Pro) | `gpt-5.2`, `gpt-5.2-mini` | yes | yes (official) | `/backend-api/wham/usage` |
| `minimax` | API key | `MiniMax-M2`, `MiniMax-M2-vl` | M2-vl only | yes (Coding Plan, official) | Token Plan users fall back to neutral weight |
| `glm` | API key | `glm-4.6`, `glm-4.5v` | 4.5v only | **unofficial** | Dashboard-internal endpoint — may break without notice |
| `opencode` | API key | `anthropic/claude-sonnet-4-6`, … | yes | no | No client-facing quota API as of April 2026 |
| `ollama` | none (self-hosted) | discovered at runtime | per-model | n/a | Disabled by default; `max_concurrent_requests: 2` |

"Usage telemetry: yes" means the `usage-aware` routing strategy has real plan-consumption data to work with. Providers marked "no" (or when any probe fails) fall back to a neutral 0.5 weight so they still participate in routing.

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
- `usage-aware` — weighted random where weight ∝ `(1 − fraction_of_plan_used)`, so providers further from exhausting their monthly plan are preferred. Providers that don't report usage (or whose probe fails) fall back to a neutral 0.5 weight — see the provider table above for which providers actually report plan telemetry. Best-effort: with only one or two telemetry-enabled providers in a pool, the effect is real but partial.

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

The built-in `ollama` provider ships disabled with `max_concurrent_requests: 2` so GPU-constrained hosts can serialize requests out of the box. Enable it with `mixer config set providers.ollama.enabled true` once you have a local ollama server running; tune the cap to whatever your hardware can handle.

## Securing the local endpoint

By default mixer binds to `127.0.0.1` and accepts unauthenticated requests — fine for single-user laptops. If you bind to a non-loopback interface (e.g. `0.0.0.0:4141` for a home-LAN shared gateway, or exposing mixer over Tailscale), gate the endpoint with a shared-secret bearer token:

```bash
mixer config set listen_bearer_token_env MIXER_BEARER
export MIXER_BEARER=$(openssl rand -hex 32)
mixer serve
```

Only the *name* of the environment variable lives in `config.json`; the token itself never touches disk. When the env var resolves to a non-empty value, mixer requires `Authorization: Bearer <token>` on every `/v1/*` request (`/healthz` is always exempt so liveness probes keep working). Mismatches return a 401 with an OpenAI-style body. Mixer logs a startup warning if the bind address is non-loopback and no token is configured.

Point your OpenAI-compatible client at mixer as usual; most SDKs expose an `api_key` / `auth_token` setting that maps to `Authorization: Bearer ...`.

## Config reference

A curated subset of fields is settable via `mixer config set`. Everything else — in particular mixer-model definitions, backends, routing strategy, and sticky-session config — is edited directly in `config.json` (use `mixer config edit`, which opens `$EDITOR` and re-validates on save).

**Settable via `mixer config set`:**

| Key | Type | Notes |
|---|---|---|
| `listen_addr` | string | `127.0.0.1:4141` by default |
| `default_model` | string | Used when a client sends an unknown `model` name |
| `listen_bearer_token_env` | string | Name of an env var holding the shared-secret token (never the token itself) |
| `providers.<id>.enabled` | bool | |
| `providers.<id>.base_url` | string | Regional mirrors / self-hosted endpoints |
| `providers.<id>.api_key_env` | string | Name of an env var; takes precedence over stored credentials |
| `providers.<id>.max_concurrent_requests` | int or `none` | Per-provider in-flight cap |
| `providers.<id>.request_timeout_secs` | int or `none` | Upstream request timeout |

Use `none`, `null`, or the empty string to clear an optional integer field.

**JSON-only (use `mixer config edit`):**

- `models.<name>` — backend pools, routing strategy, weights, sticky config
- `models.<name>.sticky` — sticky-session policy (`enabled`, `key`)

Storing raw secrets (bearer tokens, API keys) in `config.json` is intentionally rejected by `config set` — only env-var *names* live on disk, and actual credentials live under `~/.config/mixer/credentials/` (0600) via `mixer auth login`.

## Adding a provider

One file per provider under `src/providers/`. Drop in `src/providers/myprovider.rs`, implement the `Provider` trait (models, login, chat_completion, optional usage), then register it in `builtin_registry()` in `src/providers/mod.rs`. That's it — routing, auth storage, usage weighting, image filtering, and concurrency caps work automatically.

```rust
pub struct MyProvider;

#[async_trait::async_trait]
impl Provider for MyProvider {
    fn id(&self) -> &'static str { "myprovider" }
    fn display_name(&self) -> &'static str { "My Provider" }
    fn models(&self) -> Vec<ModelInfo> { /* ... */ vec![] }
    fn auth_kind(&self) -> AuthKind { AuthKind::ApiKey }
    async fn login(&self, store: &CredentialStore) -> Result<()> { /* ... */ Ok(()) }
    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatStream> { /* stream ChatCompletionChunks */ todo!() }
}
```

Providers always produce a stream of `ChatCompletionChunk`s; the server forwards them as SSE when the client asks for `stream: true` and accumulates them into a single response otherwise.

PRs for new providers welcome.

## Build & Test

```bash
cargo build
cargo test
cargo clippy -- -D warnings
```

## License

MIT
