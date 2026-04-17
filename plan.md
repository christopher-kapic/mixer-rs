# mixer-rs — Design Plan

Captures what mixer-rs is, what it's for, and every feature we want it to ship with. Serves as the authoritative spec; the code reflects a scaffolded subset of this plan (see **Status** at the end of each section).

## 1. Vision

mixer is a Rust CLI that runs a local OpenAI-compatible HTTP server on the user's machine. It accepts chat completion requests from any OpenAI-compatible client (opencode, custom agentic tools, SDKs) and fans them out — one request at a time — across the LLM subscriptions the user is logged in to.

Two ideas converging:

1. **Multi-model diversity.** There's a working theory that agent traces improve when the underlying LLM is rotated turn-to-turn. mixer makes that rotation transparent to the client: the client sees one endpoint and one virtual model name; mixer decides which real provider answers each request.
2. **Fixed-cost subscription economics.** Users already pay for multiple all-you-can-eat LLM plans (Codex / ChatGPT Plus, Minimax, GLM / z.ai, opencode Pro, …). mixer lets them pool those plans behind a single endpoint and drain them at roughly the same rate.

The tradeoff we accept: rotating providers across turns sacrifices KV-cache locality. That's a known cost; the upside of multi-model rotation is worth it for the workloads we care about.

## 2. Non-goals

- **Not a training-time tool.** mixer is strictly inference-side.
- **Not a prompt router.** mixer does not inspect prompt *content* to match models to tasks. (Image/no-image is the only content-driven routing rule.)
- **Not a harness.** mixer doesn't know about tools, plans, or agent loops; it speaks OpenAI chat completions and nothing else.
- **Not a usage database.** mixer reads current subscription usage from each provider on demand; it does not persist rolling usage history.

## 3. User-facing model

### 3.1 The endpoint

A local HTTP server exposing the OpenAI-compatible surface:

- `POST /v1/chat/completions` — the routed endpoint.
- `GET  /v1/models` — lists the *mixer models* the user has configured.
- `GET  /healthz` — liveness.

Default bind address: `127.0.0.1:4141`. Configurable via `listen_addr` in config or `--addr` / `--port` flags on `mixer serve`.

### 3.2 Mixer models (virtual models)

A **mixer model** is a virtual model name a client sends as the `model` field. It maps to a pool of concrete `(provider, provider_model)` *backends* plus a routing strategy. Users can define as many mixer models as they want — `mixer`, `vision`, `cheap`, `claude-only`, etc. — each with its own pool and strategy. `mixer serve --model <name>` restricts the server to exposing a single mixer model.

```json
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
```

### 3.3 Routing strategies

1. **`random`** — uniform over the eligible pool.
2. **`weighted`** — per-provider explicit weights (`weights: { codex: 2, minimax: 1 }`). Providers absent from `weights` default to 1.0.
3. **`usage-aware`** — weight each eligible backend by how *underused* its plan is. Concretely, `weight = max(1 − fraction_used, 0.05)` when the provider reports usage; providers that don't report usage use `0.5`. The floor ensures exhausted plans still get a tail of traffic and the default keeps silent providers from crowding out providers we *know* are underused. Net effect: plans get drained at similar rates.

### 3.4 Eligibility filter (applied before every strategy)

A backend in a mixer model's pool is dropped from the candidate set for a given request when any of the following hold:

- The provider is not registered (bad config).
- The provider's `enabled` flag is `false`.
- The user is not authenticated to the provider.
- The request includes an image part (`image_url`) and the backend's model reports `supports_images = false`.

If the filtered set is empty, the server returns an OpenAI-style error explaining why (with hints about missing auth or image capability).

### 3.5 Per-provider concurrency caps

Each provider has an optional `max_concurrent_requests` (integer). mixer enforces it with a `tokio::sync::Semaphore`: in-flight requests to that provider cannot exceed the cap. Requests beyond the cap *queue inside the server* (they are not rejected). Uncapped providers skip the semaphore entirely.

Use cases: self-hosted models that only serve N requests at a time; provider endpoints with strict concurrency limits; throttling an expensive provider without fully disabling it.

### 3.6 Authentication (per-provider)

Every provider owns its own login flow, credential shape, and persistence:

- `mixer login <provider>` invokes the provider's interactive flow.
- Credentials land in `~/.config/mixer/credentials/<provider>.json`, chmod'd `0600` on Unix.
- Each file is an opaque `serde_json::Value` — mixer doesn't know the shape; the provider reads and writes it.
- `mixer logout <provider>` deletes the file (and any server-side session, if applicable).

### 3.7 CLI surface

```
mixer init [--non-interactive]
mixer serve [--addr <host:port>] [--port <n>] [--model <name>]

mixer login <provider>
mixer logout <provider>

mixer providers list [--verbose] [--json]
mixer providers show <name> [--json]

mixer models list [--verbose] [--json]
mixer models show <name> [--json]

mixer config show [--json]
mixer config edit
mixer config set <key> <value>
mixer config path

mixer completions <shell>
```

Config `set` paths supported: `listen_addr`, `default_model`, `providers.<id>.enabled`, `providers.<id>.max_concurrent_requests`, `providers.<id>.base_url`, `providers.<id>.request_timeout_secs`.

## 4. Providers

### 4.1 Initial set (subscription-based)

| id         | display name             | plan                         | status   |
|------------|--------------------------|------------------------------|----------|
| `codex`    | Codex (ChatGPT Plus/Pro) | OpenAI ChatGPT subscription  | scaffold |
| `minimax`  | Minimax                  | Minimax monthly plan         | scaffold |
| `glm`      | GLM (z.ai)               | z.ai monthly plan            | scaffold |
| `opencode` | opencode                 | opencode Pro / opencode Go   | scaffold |

Model catalogues and image capabilities for each are hard-coded in `src/providers/<id>.rs` and should be kept current as providers release new models.

### 4.2 Future providers

- Usage-based providers (OpenAI direct API, Anthropic direct API, Google Gemini, Groq, DeepInfra, Fireworks, Together, …).
- Self-hosted endpoints (Ollama, vLLM, TGI, llama.cpp-server). These are the main motivator for `max_concurrent_requests`.
- Other subscriptions as they emerge (Pi, Cursor bridge, Copilot bridge, …).

### 4.3 Adding a provider

One file per provider under `src/providers/`. The PR surface is:

1. Create `src/providers/<id>.rs` with a struct implementing `Provider`.
2. Register it in `builtin_registry()` in `src/providers/mod.rs`.
3. (Optional) Add it to `Config::default().providers`.

Everything else — filtering, weighting, usage-aware picks, image-capability routing, concurrency caps, credential storage — flows through the generic server/router for free.

### 4.4 The Provider trait

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    fn models(&self) -> Vec<ModelInfo>;                  // includes supports_images
    fn is_authenticated(&self, store: &CredentialStore) -> bool;
    async fn login(&self, store: &CredentialStore) -> Result<()>;
    async fn logout(&self, store: &CredentialStore) -> Result<()>;
    async fn usage(&self, store: &CredentialStore) -> Result<Option<UsageSnapshot>>;
    async fn chat_completion(
        &self,
        store: &CredentialStore,
        settings: &ProviderSettings,
        req: ChatRequest,
    ) -> Result<ChatResponse>;
}
```

`req.model` is the provider-native model id at this point (the router rewrites it from the mixer model name before dispatch).

### 4.5 Usage reporting

`UsageSnapshot { fraction_used: Option<f64>, window: String, label: Option<String> }`.

Providers that can introspect plan consumption return a `fraction_used` in `[0, 1]`; providers that can't return `None`. The `window` (`"monthly"`, `"daily"`, …) and free-form `label` (`"1.2M / 5M tokens"`) are for display only. Usage is queried lazily on each usage-aware routing decision — no background polling, no on-disk cache.

## 5. Request handling

### 5.1 Lifecycle

1. Axum decodes the JSON body into `ChatRequest`.
2. Reject streaming (`stream: true`) with 501 until SSE is wired.
3. Enforce `--model` pin if the server was started that way.
4. Resolve the request's `model` name to a `MixerModel` (fall back to `default_model`).
5. Inspect messages for `image_url` parts → `requires_images`.
6. `router::pick` filters the pool and applies the strategy.
7. Log the route (`mixer_model → provider/model (images=…)`).
8. Rewrite `req.model` to the provider-native model id.
9. Acquire the provider's concurrency permit (if any).
10. Dispatch `Provider::chat_completion` and return its response verbatim.

### 5.2 OpenAI compatibility

- Known fields (`model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `tools`, `tool_choice`) are parsed explicitly.
- Everything else (`response_format`, `seed`, `logit_bias`, provider-specific extensions) is captured in a flattened `extra` map and forwarded verbatim.
- Message content supports both the string form (`"content": "hi"`) and the content-parts form (`"content": [{"type": "text", ...}, {"type": "image_url", ...}]`).
- Errors return OpenAI-style bodies: `{ "error": { "message": ..., "type": ... } }`.

### 5.3 Streaming (planned)

Currently returns 501 for `stream: true`. Plan: forward the provider's SSE stream straight back to the client, re-emitting `data: {...}` chunks. This requires each provider to expose a streaming variant of `chat_completion` returning `impl Stream<Item = Result<ChatStreamChunk>>`. Add once at least one real provider is implemented.

## 6. Configuration

### 6.1 Location

- Config file: `~/.config/mixer/config.json` (XDG-style on every OS, including macOS).
- Credentials: `~/.config/mixer/credentials/<provider>.json` (`0700` dir, `0600` files on Unix).
- Overridable via `$XDG_CONFIG_HOME`.

### 6.2 Schema

```json
{
  "listen_addr": "127.0.0.1:4141",
  "default_model": "mixer",
  "models": {
    "<mixer_model_name>": {
      "description": "...",
      "strategy": "random" | "weighted" | "usage-aware",
      "weights": { "<provider_id>": <number> },
      "backends": [
        { "provider": "<provider_id>", "model": "<provider_model_id>" }
      ]
    }
  },
  "providers": {
    "<provider_id>": {
      "enabled": true,
      "base_url": "https://...",
      "max_concurrent_requests": 2,
      "request_timeout_secs": 120
    }
  }
}
```

Atomic writes: save goes to a temp file in the same dir, then renames. Missing files load `Default::default()` silently.

## 7. Tech stack

- Rust edition 2024, `rust-version = 1.85`.
- CLI: `clap` v4 derive + `clap_complete`.
- Async: `tokio` (full runtime).
- HTTP server: `axum` 0.8.
- HTTP client: `reqwest` 0.12 with `rustls-tls`.
- Trait objects: `async-trait` (for dyn-compatible `Provider`).
- Serde: `serde` + `serde_json`.
- RNG: `rand` 0.8 for routing decisions.
- Platform dirs: `dirs` 6 (XDG).
- Errors: `anyhow`.
- Timestamps: `chrono`.
- Atomic writes: `tempfile`.

## 8. Project layout

```
src/
  main.rs               Entry point, tokio runtime, clap dispatch
  cli.rs                Clap command/arg definitions
  paths.rs              XDG path resolution
  config.rs             Config + MixerModel + Backend + RoutingStrategy
  credentials.rs        Per-provider credential storage
  openai.rs             OpenAI wire types + request_has_images
  usage.rs              UsageSnapshot shared type
  concurrency.rs        Per-provider semaphore caps
  router.rs             Backend selection: filter + strategy dispatch
  server.rs             Axum app + AppError
  providers/
    mod.rs              Provider trait + registry + builtin_registry
    codex.rs            Codex / ChatGPT Plus/Pro provider
    minimax.rs          Minimax provider
    glm.rs              GLM / z.ai provider
    opencode.rs         opencode subscription provider
  commands/
    mod.rs              Re-exports
    init.rs             mixer init
    serve.rs            mixer serve
    login.rs            mixer login
    logout.rs           mixer logout
    providers.rs        mixer providers list|show
    models.rs           mixer models list|show
    config_cmd.rs       mixer config show|edit|set|path
    completions.rs      mixer completions <shell>
```

## 9. Distribution

- GitHub Releases triggered on `Cargo.toml` version bump in `.github/workflows/release.yml`.
- Cross-compiled binaries: `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`.
- `scripts/install.sh` for `curl | bash` installs (reads `INSTALL_DIR`, defaults to `/usr/local/bin`).
- `.github/workflows/ci.yml` runs `cargo build`, `cargo test`, `cargo clippy -D warnings`, `cargo fmt --check`.

## 10. Status (as of scaffolding)

- [x] CLI structure + all subcommands wired.
- [x] Config types, defaults, atomic save, `resolve_model`, JSON round-trip tests.
- [x] Credential store with `0600` permissions.
- [x] `Provider` trait + registry + `builtin_registry()` + 4 stub provider files.
- [x] Router with `random`, `weighted`, and `usage-aware` strategies.
- [x] Image-capability filter driven by `request_has_images`.
- [x] Per-provider `max_concurrent_requests` semaphores (`ConcurrencyLimits`).
- [x] Axum server with `/v1/chat/completions`, `/v1/models`, `/healthz`.
- [x] OpenAI-style error bodies.
- [x] CI workflow (build/test/clippy/fmt).
- [x] Release workflow + `install.sh`.
- [x] 26 unit tests, `clippy -D warnings` clean, `fmt --check` clean.

Outstanding work (in rough priority order):

- [ ] Real `login` + `chat_completion` bodies for each of the four initial providers.
- [ ] Real `usage` implementations where the provider exposes an endpoint; document "usage unknown" behaviour for the rest.
- [ ] SSE streaming end-to-end (`stream: true`).
- [ ] `Provider::is_authenticated` that does a credential *freshness* check (refresh tokens, expiry), not just file existence.
- [ ] `mixer doctor` — validate config, probe each authenticated provider, report state.
- [ ] First self-hosted provider (Ollama or vLLM) to exercise `max_concurrent_requests`.
- [ ] Metrics / structured logs (tracing + `--log-format json`).
- [ ] Client-level auth: optional bearer-token gate on the local HTTP endpoint for multi-user machines.
- [ ] Per-request retry / failover (if the picked provider errors, optionally try another from the pool).
- [ ] Configurable "sticky session" mode to preserve KV-cache locality within a conversation when desired.
