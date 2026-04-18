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

If the filtered set is empty, the server returns `503 Service Unavailable` with an OpenAI-style body:

```json
{ "error": {
    "type": "no_backend_available",
    "message": "no authenticated backend for mixer model 'vision' supports image inputs — run `mixer auth login <provider>` or add an image-capable backend"
} }
```

The message should be specific enough to name the failing mixer model and the actionable fix (missing auth, missing vision capability, provider disabled, etc.).

### 3.5 Per-provider concurrency caps

Each provider has an optional `max_concurrent_requests` (integer). mixer enforces it with a `tokio::sync::Semaphore`: in-flight requests to that provider cannot exceed the cap. Requests beyond the cap *queue inside the server* (they are not rejected). Uncapped providers skip the semaphore entirely.

Use cases: self-hosted models that only serve N requests at a time; provider endpoints with strict concurrency limits; throttling an expensive provider without fully disabling it.

### 3.6 Authentication (per-provider)

Every provider owns its own login flow, credential shape, and persistence:

- `mixer auth login <provider>` invokes the provider's interactive flow.
- Credentials land in `~/.config/mixer/credentials/<provider>.json`, chmod'd `0600` on Unix.
- Each file is an opaque `serde_json::Value` — mixer doesn't know the shape; the provider reads and writes it.
- `mixer auth logout <provider>` deletes the file (and any server-side session, if applicable).
- `mixer auth status [<provider>]` reports whether credentials exist and pass a freshness check (e.g. refresh-token-not-expired).

#### 3.6.1 Supported auth flow shapes

Providers pick whichever flow their upstream supports. The initial set we need to cover:

1. **OAuth 2.0 device authorization (RFC 8628)** — the `mixer auth login` command prints the `user_code` and `verification_uri` (ideally also `verification_uri_complete`), optionally opens the URL in the user's browser, then polls the token endpoint until the user approves. The returned access/refresh token pair is persisted. This is the flow used by **Codex** (ChatGPT Plus/Pro) and by **GitHub Copilot** (the canonical reference implementations for mixer live in the opencode source tree at `packages/opencode/src/plugin/codex.ts` and `packages/opencode/src/plugin/copilot.ts`).
2. **OAuth 2.0 authorization code + PKCE** — opens a browser, runs a one-shot loopback HTTP listener to receive the redirect, exchanges the code for tokens. Used by Anthropic in the `opencode-anthropic-auth` plugin; likely needed if we ever add direct Anthropic.
3. **API key paste** — the simplest case. User pastes a key and it's stored verbatim. This is what **opencode Pro / opencode Go (Zen)** uses (create key at `https://opencode.ai/auth`), and what direct OpenAI/Minimax/GLM API keys look like.
4. **Provider-specific login** — anything weirder (e.g. username/password, magic link) stays behind the `Provider::login` trait method.

A shared `src/auth/device_flow.rs` helper should handle the RFC 8628 polling loop (respecting `interval`, `slow_down`, `expires_in`) so each device-flow provider just supplies client id / scopes / endpoint URLs.

#### 3.6.2 Credential sources: stored file vs. environment variable

Credentials resolve from two sources, in this order:

1. **Environment variable** — if the provider's config block sets `api_key_env: "<VAR_NAME>"` and that variable is non-empty, its value is used directly as the API key.
2. **Stored file** — `~/.config/mixer/credentials/<provider>.json` (written by `mixer auth login`).

**Env first, file second.** The intent: env vars are the more deliberate per-run declaration (CI, containers, shell sessions that want to override stored creds without touching disk), so they win. A stored credential is the fallback "I already logged in once" case.

**No well-known defaults.** mixer never auto-probes `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. The user must opt in by setting `api_key_env` explicitly in config. Rationale: silent env pickup makes it unclear to the user *why* a provider appears authenticated, and collides with tools that set those vars for other purposes.

**No literal `api_key` in config.** Secrets never land in `config.json`. The only two places a key can live are env vars or the `0600` credentials file.

**Only API-key providers consult env.** Device-flow providers (e.g. `codex`) persist a full token blob — refresh token, access token, expiry — that doesn't fit a single env var. They ignore `api_key_env` entirely.

**CredentialStore surface** exposes two lookups, and only the second consults env:

```rust
impl CredentialStore {
    fn load_blob(&self, provider_id: &str) -> Result<Option<serde_json::Value>>;  // file-only
    fn load_api_key(&self, provider_id: &str, settings: &ProviderSettings) -> Option<String>;
    //   ^ tries env (settings.api_key_env) first, then reads "api_key" field from stored blob
}
```

**UX consequences:**

- `mixer auth status` reports the resolved source per provider: `env:OPENAI_API_KEY`, `file:credentials/minimax.json`, `file:credentials/codex.json (oauth, expires in 12d)`, or `missing`.
- `mixer auth logout <provider>` deletes the stored file and warns `"env var <VAR> is still set; provider remains authenticated via env"` when relevant.
- `mixer auth login <provider>` for an API-key provider that has `api_key_env` set still works — it writes the file — and prints the same warning if the env var is also set (env will shadow the freshly stored key).

#### 3.6.3 Token lifecycle for OAuth device-flow providers

OAuth access tokens expire (codex's are hours-scale). mixer refreshes transparently. Strategy:

1. **Proactive expiry check at request time.** Before dispatching an outbound request, decode the access token's `exp` claim (the JWT is already being parsed to extract `chatgpt-account-id`, so this is free). If it's within **60 seconds** of expiry, refresh first using the stored refresh token, write the new tokens back to `credentials/<provider>.json`, then dispatch.
2. **401 safety net.** If the upstream returns 401 anyway (token was rotated server-side, clock skew, etc.), refresh once and retry the request a single time. No infinite retry loop — a second 401 surfaces as an error.
3. **Refresh failure UX.** If the refresh endpoint itself rejects the refresh token (revoked, user changed password, session expired), return an OpenAI-style 401 to the client:
   ```json
   { "error": { "type": "authentication_error",
                "message": "codex credentials expired — run `mixer auth login codex`" } }
   ```
   Client sees a real auth error; user sees an actionable fix. mixer does **not** try to initiate a fresh device flow automatically — that requires user interaction on a terminal that may not be attached to the running server.

Concurrency: refreshes are guarded by a per-provider `tokio::sync::Mutex` so simultaneous in-flight requests don't double-refresh. The first waiter performs the refresh; the rest read the freshly written credentials.

No background refresher task. For a localhost CLI that can sleep/wake arbitrarily (laptop closed), proactive-at-request-time + 401 fallback is simpler and correct under suspend/resume.

### 3.7 CLI surface

```
mixer init [--non-interactive]   # writes default config.json if absent, exits. Does NOT trigger login — that's `mixer auth login <provider>`.
mixer serve [--addr <host:port>] [--port <n>] [--model <name>]

mixer auth login <provider>
mixer auth logout <provider>
mixer auth status [<provider>] [--json]

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

Config `set` paths supported: `listen_addr`, `default_model`, `providers.<id>.enabled`, `providers.<id>.max_concurrent_requests`, `providers.<id>.base_url`, `providers.<id>.request_timeout_secs`, `providers.<id>.api_key_env`.

## 4. Providers

### 4.1 Initial set (subscription-based)

| id         | display name             | plan                         | auth flow                | wire protocol            | status   |
|------------|--------------------------|------------------------------|--------------------------|--------------------------|----------|
| `codex`    | Codex (ChatGPT Plus/Pro) | OpenAI ChatGPT subscription  | OAuth 2.0 device flow    | Responses API (SSE-only) | scaffold |
| `minimax`  | Minimax                  | Minimax monthly plan         | API key paste            | OpenAI-compatible        | scaffold |
| `glm`      | GLM (z.ai)               | z.ai monthly plan            | API key paste            | OpenAI-compatible        | scaffold |
| `opencode` | opencode (Zen)           | opencode Pro / opencode Go   | API key paste            | OpenAI-compatible        | scaffold |

#### 4.1.1 Codex wire protocol notes

Codex against a ChatGPT subscription does **not** hit `/v1/chat/completions`. The `codex` provider is a translator, not a passthrough. Details derived from the Codex CLI Rust source (`codex-rs/`):

- **Endpoint:** `POST https://chatgpt.com/backend-api/codex/responses`.
- **Request shape:** OpenAI **Responses API** — `input: [{type: "input_text", text}]`, `instructions`, `reasoning: {effort}`, `tools`, `tool_choice`, `parallel_tool_calls`, `store`, `stream`, `prompt_cache_key`, `client_metadata`. *Not* the Chat Completions `messages` array. Inbound Chat Completions requests must be translated.
- **Response:** SSE stream only. Event types include `response.created`, `response.output_item.added`, `response.output_item.done`, `response.completed`. No non-streaming JSON variant is exposed — even when the upstream caller wants a single `ChatCompletion` object, `codex` must consume the SSE stream and accumulate.
- **Required headers** (beyond `Content-Type`):
  - `Authorization: Bearer <access_token>`
  - `chatgpt-account-id: <id>` — parsed from the ID token JWT claim `https://api.openai.com/auth.chatgpt_account_id`. Requires a JWT decode step at login time; the account id is stored alongside the tokens in the credentials file.
  - `X-OpenAI-Fedramp: true` — only for FedRAMP accounts; optional.

Reference: `codex-rs/model-provider/src/bearer_auth_provider.rs`, `codex-rs/codex-api/src/endpoint/responses.rs`, `codex-rs/codex-api/src/sse/responses.rs`, `codex-rs/login/src/token_data.rs` (via the `codex` kctx dependency).

The two translation directions (OpenAI Chat Completions ↔ OpenAI Responses API, for both request and SSE response) live in a shared `src/providers/common/responses_api.rs` so any future "OpenAI Responses API" provider can reuse them.

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

`chat_completion` returns a `Stream<Item = Result<ChatCompletionChunk>>` unconditionally. The server layer collapses the stream into a single `ChatCompletion` when the caller sent `stream: false`. This shape is forced by `codex`, whose upstream is SSE-only — making non-streaming providers "fake" a single-chunk stream is cheaper than asking SSE-only providers to double-buffer.

### 4.5 Usage reporting

`UsageSnapshot { fraction_used: Option<f64>, window: String, label: Option<String> }`.

Providers that can introspect plan consumption return a `fraction_used` in `[0, 1]`; providers that can't return `None`. The `window` (`"monthly"`, `"daily"`, …) and free-form `label` (`"1.2M / 5M tokens"`) are for display only. Usage is queried lazily on each usage-aware routing decision — no background polling, no on-disk cache.

## 5. Request handling

### 5.1 Lifecycle

1. Axum decodes the JSON body into `ChatRequest`.
2. Enforce `--model` pin if the server was started that way.
3. Resolve the request's `model` name to a `MixerModel` (fall back to `default_model`).
4. Inspect messages for `image_url` parts → `requires_images`.
5. `router::pick` filters the pool and applies the strategy.
6. Log the route (`mixer_model → provider/model (images=…)`).
7. Rewrite `req.model` to the provider-native model id.
8. Acquire the provider's concurrency permit (if any).
9. Dispatch `Provider::chat_completion` — always returns a `Stream<ChatCompletionChunk>`.
10. If the client sent `stream: true`, pipe the stream to an SSE response. Otherwise, accumulate chunks and return a single `ChatCompletion` JSON object.

### 5.2 OpenAI compatibility

- Known fields (`model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `tools`, `tool_choice`) are parsed explicitly.
- Everything else (`response_format`, `seed`, `logit_bias`, provider-specific extensions) is captured in a flattened `extra` map and forwarded verbatim.
- Message content supports both the string form (`"content": "hi"`) and the content-parts form (`"content": [{"type": "text", ...}, {"type": "image_url", ...}]`).
- Errors return OpenAI-style bodies: `{ "error": { "message": ..., "type": ... } }`.

### 5.2.1 Failover on provider errors

When the picked backend fails, mixer retries once against a *different* backend from the same mixer model's eligible pool, per-error-class. Retry budget is hardcoded to **1** — a second failure surfaces to the client.

**Retry on:**
- HTTP `429` (rate-limited)
- HTTP `500`, `502`, `503`, `504`
- Connection errors (DNS, TCP refused, TLS, read/write timeout)
- `401` after a refresh-and-retry already failed (treat as provider-unavailable, not request-broken)

**Do not retry on:**
- HTTP `400`, `403`, `404`, `422` — these are about the *request*, not the provider. Retrying wastes a second backend's quota for the same failure.
- HTTP `401` on the first attempt — handled by §3.6.3's refresh logic, not by failover.

**Mid-stream failure.** Once the first `ChatCompletionChunk` has been emitted to the client, retry is impossible (the client already has partial output). In that case mixer emits one final frame matching OpenAI's own behavior:

```
data: {"error": {"message": "...", "type": "upstream_error"}}

```

Then closes the connection without sending `data: [DONE]`. The OpenAI Python SDK and all compatible SDKs parse `data:` frames and raise on the `error` key, so this surfaces as a real exception client-side. `event: error` SSE frames are **not** used — OpenAI doesn't emit them and most SDKs ignore them.

Retry therefore only applies *before* the first chunk has been flushed to the client. A pre-first-chunk failure is a clean retry (the client hasn't seen anything yet).

**Not in v1:** temporary down-weighting of a recently-429'd backend (usage-aware already drains the exhausted plan down; short-lived 429s usually resolve faster than any sensible cooldown). Revisit if hot-loop retries show up in practice.

**Not in v1:** configurable retry budget or per-provider retry toggles. Hardcoded `1` until someone needs more.

### 5.2.2 Tool calling (`tools` / `tool_choice`)

mixer's internal request shape uses OpenAI **Chat Completions** `tools` / `tool_choice` verbatim. Inbound requests pass through unchanged to OpenAI-compatible providers (`minimax`, `glm`, `opencode`). For `codex`, the `common/responses_api.rs` translator converts:

- **Request:** Chat Completions `tools: [{"type": "function", "function": {"name", "description", "parameters"}}]` → Responses API `tools: [{"type": "function", "name", "description", "parameters"}]` (function fields hoisted up one level). Same for `tool_choice`.
- **Response:** Responses API `function_call` items (with `call_id`) → Chat Completions tool-call chunks (with `tool_call_id`). IDs must map consistently across streaming deltas so the client can correlate call → result.

**Why this design (single internal shape, translate at the boundary):**

1. Chat Completions `tools` is the de facto client standard. Every tool-using client that mixer cares about (opencode, openai SDKs, LangChain, direct HTTP) emits this shape.
2. Most providers advertise OpenAI Chat Completions compat, so pass-through *should* work without per-provider adapters.
3. Keeping one internal shape means `router::pick`, logging, and request validation all see a single format.

**Known downsides we're accepting in v1:**

- **"OpenAI-compatible" diverges in practice.** Providers sometimes differ on `tool_choice: "required"` vs `"auto"` semantics, parallel tool calls, unknown-field rejection, or streaming tool-call delta shapes. Minimax/GLM/opencode-Zen will pass through *raw*; if any of them disagree with OpenAI's interpretation, the client will see wrong results and we won't know until a bug report.
- **No cross-provider tool test coverage at ship time.** Shipping on shape-matching alone.
- **Image parts + tool calls in the same request** — exotic combined shape, untested against any provider.
- **Tool-call ID semantics differ between Chat Completions and Responses API** (`tool_call_id` vs `call_id`). The codex translator has to preserve consistent IDs across the request and every streaming chunk; subtle bugs here break round-tripping.

**Changes we'd likely make later if pain shows up:**

- Per-provider tool-format adapter modules (`providers/common/minimax_tools.rs`, etc.) when one diverges visibly.
- A `mixer doctor --check-tools` command that sends a canonical tool call through every authenticated provider and reports format discrepancies.
- If the Responses API becomes the dominant upstream shape (OpenAI's own direction appears to be this), flip mixer's *internal* representation to Responses API and translate Chat Completions at ingress instead. That inversion is a ~2-day refactor but would be worth it if 3+ providers adopt Responses API natively.

### 5.3 Streaming

Streaming is **v1**, not deferred — the `codex` provider's upstream is SSE-only, so an SSE parser/translator is mandatory regardless of what the client asks for.

- `Provider::chat_completion` returns `Stream<Item = Result<ChatCompletionChunk>>` unconditionally.
- When the client sent `stream: true`, the server forwards the stream as `text/event-stream` with `data: {...}` chunks plus a trailing `data: [DONE]`.
- When the client sent `stream: false`, the server accumulates the stream and returns a single `ChatCompletion` JSON object.
- Non-SSE upstreams (Minimax, GLM, opencode — whichever expose a plain JSON chat completion) are adapted with a one-chunk stream.

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
      "request_timeout_secs": 120,
      "api_key_env": "OPENAI_API_KEY"
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
- JWT decode: `jsonwebtoken` — used by `codex` to pull `chatgpt-account-id` out of the ID token's claims at login.
- SSE: `eventsource-stream` (or `reqwest-eventsource`) for parsing upstream SSE streams; axum's `Sse` response for emitting them.
- Async streams: `futures` / `async-stream` for the `Stream<ChatCompletionChunk>` surface.

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
  auth/
    mod.rs              Shared auth helpers
    device_flow.rs      RFC 8628 device-authorization polling loop
    pkce.rs             PKCE helpers (for future OAuth code-flow providers)
  providers/
    mod.rs              Provider trait + registry + builtin_registry
    common/
      mod.rs
      responses_api.rs  OpenAI Chat Completions ↔ Responses API translators (request + SSE response)
    codex.rs            Codex / ChatGPT Plus/Pro provider (uses common/responses_api)
    minimax.rs          Minimax provider
    glm.rs              GLM / z.ai provider
    opencode.rs         opencode subscription provider
  commands/
    mod.rs              Re-exports
    init.rs             mixer init
    serve.rs            mixer serve
    auth_cmd.rs         mixer auth login|logout|status
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

- [ ] Reshape CLI from `mixer login|logout <provider>` to `mixer auth login|logout|status <provider>` subcommand group.
- [ ] `src/auth/device_flow.rs` — shared RFC 8628 device-authorization helper (poll `device_authorization` → `token`, respect `interval` / `slow_down` / `expires_in`). First consumer: `codex`. Reference implementations: opencode's `plugin/codex.ts` and `plugin/copilot.ts`.
- [ ] `api_key_env` support in `ProviderSettings` + `CredentialStore::load_api_key` with env→file precedence, and env-source reporting in `mixer auth status`.
- [ ] Real `login` + `chat_completion` bodies for each of the four initial providers.
- [ ] Real `usage` implementations where the provider exposes an endpoint; document "usage unknown" behaviour for the rest.
- [ ] `Provider::chat_completion` returns `Stream<ChatCompletionChunk>` unconditionally; server collapses to a `ChatCompletion` when `stream: false` and forwards SSE when `stream: true`.
- [ ] `src/providers/common/responses_api.rs` — Chat Completions ↔ Responses API translators, including the `response.output_item.*` SSE event parser. First consumer: `codex`.
- [ ] JWT decode step in `codex` login to extract `chatgpt-account-id` from the ID token and persist it with the tokens.
- [ ] `Provider::is_authenticated` that does a credential *freshness* check (refresh tokens, expiry), not just file existence.
- [ ] Token-refresh plumbing for device-flow providers: expiry-check-at-request-time (60s threshold) with 401-retry fallback, per-provider `Mutex` guarding simultaneous refreshes, OpenAI-style 401 with "run `mixer auth login <provider>`" message when refresh itself fails.
- [ ] `mixer doctor` — validate config, probe each authenticated provider, report state.
- [ ] First self-hosted provider (Ollama or vLLM) to exercise `max_concurrent_requests`.
- [ ] Metrics / structured logs (tracing + `--log-format json`).
- [ ] Client-level auth: optional bearer-token gate on the local HTTP endpoint for multi-user machines.
- [ ] Per-request failover (§5.2.1): retry once on a different eligible backend for 429 / 5xx / connection errors / post-refresh 401; surface 4xx immediately; emit a final `data: {"error": {...}}` SSE frame on mid-stream failure (no `[DONE]`).
- [ ] Configurable "sticky session" mode to preserve KV-cache locality within a conversation when desired.
