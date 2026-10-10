# GLM-5.3-Flash — containerized llama-server

[![image](https://github.com/onixldlc/glm53f-server/actions/workflows/build.yml/badge.svg)](https://github.com/onixldlc/glm53f-server/actions/workflows/build.yml)

GLM-5.3-Flash served by **stock upstream llama.cpp** — no fork, no patched ggml types. The
weights are [Unsloth Dynamic](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) GGUFs
(`UD-*`), ordinary K-quants that an official release tarball loads as-is. The image is the
upstream release binaries plus an entrypoint that fetches the weights and starts the server.

This is the same stack as [`qwen38-server`](https://github.com/onixldlc/qwen38-server) pointed
at a much larger model, and the size is the whole story. A 27B dense model fits on one 24 GB
card; **no single GPU runs this one** — the smallest quant is 86.69 GiB and the one worth using
is 112.10 GiB. Plan on 2x A100-80 or 4x RTX 5090 and up. In exchange you get the #4 open-weights
model on the Artificial Analysis Intelligence Index v4.3.2 (41.81) and DeepSWE v1.1 63.4, ahead
of Claude Opus 4.8's 58.0 on that benchmark.

| Piece | Value |
|---|---|
| Base model | GLM-5.3-Flash, 320B-A18B MoE (`glm5-next` arch, 34 KDA linear + 11 DSA sparse attention layers, 1048576 native context), MIT |
| Modality | natively multimodal — text + vision, `ENABLE_VISION=1` fetches the 1.08 GB tower |
| Weights | [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) |
| llama.cpp release | [`b11461`](https://github.com/ggml-org/llama.cpp/releases/tag/b11461), asset `bin-ubuntu-cuda-12.8-x64` (gpu) / `bin-ubuntu-x64` (cpu) |
| Arch support | [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773), merged 2026-09-30 as `649dcb1036fd`; `b11461` (2026-10-07) contains it |
| Image | [`ghcr.io/onixldlc/glm53f-server`](https://github.com/onixldlc/glm53f-server/pkgs/container/glm53f-server) |
| API | OpenAI-compatible on `:8080` (`/v1/chat/completions`), web UI at `/` |

Links: [weights on HuggingFace](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) ·
[GHCR package](https://github.com/onixldlc/glm53f-server/pkgs/container/glm53f-server) ·
[build workflow](.github/workflows/build.yml)

## Layout

```
docker/        Dockerfile + entrypoint.sh + verify.sh + mcp-web.sh + mcp-files.sh
docker/gateway/  nginx + njs front door: Dockerfile, nginx.conf.template, upload.js
gpu/           compose.yaml + compose-dev.yaml + .env  — CUDA
cpu/           compose.yaml + compose-dev.yaml + .env  — CPU-only, for a VPS
VERSION        the release number; changing it is what triggers a publish
.github/       workflow that builds all three images and pushes them to GHCR
```

`compose.yaml` pulls the published image — `:gpu` or `:cpu` — so a plain `up` downloads a
build instead of making one:

```bash
cd gpu && podman compose up -d     # CUDA card
cd cpu && podman compose up -d     # VPS
```

`compose-dev.yaml` is the override that builds from this checkout instead. Use it whenever
you are changing anything under `docker/`, since a published tag cannot contain edits you
have not released yet:

```bash
cd gpu && podman compose -f compose.yaml -f compose-dev.yaml up -d --build
```

It replaces nothing but the image source: ports, volume, devices and `env_file` still come
from `compose.yaml`, and the local tag (`localhost/glm53f-server:dev-gpu`) is distinct so a
dev build never shadows the pulled one. The Dockerfile takes `BASE_IMAGE` and
`LLAMA_FLAVOR` build args, which is how one file yields the CUDA image (`nvidia/cuda` base,
172 MB release asset) and the CPU image (`ubuntu:24.04`, 17.7 MB asset); the cpu dev
override sets them, gpu uses the defaults.

Both declare a volume named `glm53f-models`, but compose prefixes it with the folder name,
so on one host they are `gpu_glm53f-models` and `cpu_glm53f-models` — separate copies of a
112 GiB download. That matters a great deal more here than it did at 18.5 GB: do not run both
on one machine unless you mean to pay for the weights twice.

## Prebuilt images

Skip the build and pull from GHCR. One image name, the variant is the tag:

| Tag | What it is |
|---|---|
| `ghcr.io/onixldlc/glm53f-server:gateway` | newest nginx + njs front door — rolling |
| `ghcr.io/onixldlc/glm53f-server:gpu` | newest CUDA build — rolling |
| `ghcr.io/onixldlc/glm53f-server:cpu` | newest CPU-only build — rolling |
| `ghcr.io/onixldlc/glm53f-server:v0.1.0-gpu` | that release's CUDA build — pinned |
| `ghcr.io/onixldlc/glm53f-server:v0.1.0-cpu` | that release's CPU-only build — pinned |

```bash
podman run -d --name glm53f \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -v glm53f-models:/models -p 8080:8080 \
  -e MODEL_FILE=UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  -e CTX_SIZE=131072 \
  ghcr.io/onixldlc/glm53f-server:gpu
```

## Agent tools

`EXTRA_ARGS` in both `.env.example` files turns on llama.cpp's own built-in tools:

```
--tools read_file,write_file,edit_file,file_glob_search,grep_search,get_info
```

`exec_shell_command` is available upstream and deliberately left out here — it runs in the
container with the server's privileges.

**Where they operate:** a relative path is resolved against the server's working directory,
which this image sets to `/workspace`
([`server-tools.cpp:303-316`](https://github.com/ggml-org/llama.cpp/blob/b11461/tools/server/server-tools.cpp#L303-L316)).
There is no CLI flag for a tool root — the only override is the `x-tool-cwd` header on a
`POST /tools/...` call
([`server-tools.cpp:2089`](https://github.com/ggml-org/llama.cpp/blob/b11461/tools/server/server-tools.cpp#L2089)),
which the model itself cannot set. Both `compose.yaml` files carry a commented-out
workspace mount; uncomment it to give the tools a project:

```yaml
    volumes:
      - glm53f-models:/models
      - ./workspace:/workspace:z
```

Note that an **absolute** path is passed through unchanged
([`server-tools.cpp:162`](https://github.com/ggml-org/llama.cpp/blob/b11461/tools/server/server-tools.cpp#L162)) —
`/workspace` is a default, not a jail. The tools can reach anything the container can.

## Uploads and the gateway

The agent tools above can read anything on `/workspace`, but only once the model knows the
path. Nothing tells it that a zip arrived, where it landed, or what is inside. That is the
gap the gateway and the `uploads_*` tools close.

**One port, two things behind it.** `gateway` is the service that publishes `8080`;
`llama-server` keeps its port inside the compose network and is proxied. Everything this
image serves lives under `/manage/`, and everything outside it is llama.cpp:

```
:8080/manage/                the file manager page
:8080/manage/files           JSON listing       ?path= &depth=
:8080/manage/upload          PUT a raw body     ?name=
:8080/manage/remove          DELETE one upload  ?name=
:8080/manage/download/<path> read a file back out
:8080/manage/health          gateway liveness, no upstream involved

:8080/                       llama.cpp's WebUI and API (/v1/*, /health, /props, ...)
```

**The page and the API are the same endpoints.** `/manage/` is a file manager — a listing in
the middle, drop files anywhere on it to upload, click a file to download it, click a folder
to walk into it — and every action it takes is one of the URLs above. There is no private
API behind the UI, so anything the page can do, `curl` and the MCP server can do with the
same call.

The single prefix is also why the upstream can grow any route it likes without colliding
with this image.

nginx and njs are **one** service, not two: njs is an nginx module, so the upload handler and
the reverse proxy are the same process. The image is `alpine` plus the distro `nginx` and
`nginx-mod-http-js`, deliberately not `nginx:alpine` — `ngx_http_js_module` is a dynamic
module and only loads into the exact nginx build it was compiled against, and Alpine main
ships the pair at a matching `-r` revision.

Uploads are a **raw body, never multipart**. nginx buffers the body to a temp file on the same
filesystem as the destination and `upload.js` renames it into place, so a multi-gigabyte zip
never touches the njs heap:

```bash
curl -T report.zip 'http://127.0.0.1:8080/manage/upload?name=report.zip'
```

```json
{
  "stored": "report.zip",
  "bytes": 19283746,
  "workspace_path": "inbox/report.zip",
  "download": "/manage/download/inbox/report.zip"
}
```

**`workspace_path` is the point.** The gateway writes to the volume that is also
`llama-server`'s working directory, so a path reported here is already relative to what the
built-in file tools resolve against. Nothing translates paths between the two halves, because
there is nothing to translate.

### Scripting it

```bash
curl -s   'http://127.0.0.1:8080/manage/files?path=inbox&depth=1'
curl -T x.zip 'http://127.0.0.1:8080/manage/upload?name=x.zip'
curl -O   'http://127.0.0.1:8080/manage/download/inbox/x.zip'
curl -X DELETE 'http://127.0.0.1:8080/manage/remove?name=x.zip'
```

`depth=1` lists one level, which is what the page walks; left off, the listing recurses to
`UPLOADS_MAX_DEPTH`. Both forms cap out at 2000 entries and say so with `"truncated": true`.

### What the model sees

`docker/mcp-files.sh` is a second MCP stdio server — `sh`, `jq`, `find`, `unzip`, `tar` —
exposing five tools:

| Tool | What it answers |
|---|---|
| `uploads_list` | what is on the volume, newly uploaded archives first |
| `uploads_peek` | what is inside an archive, **without** extracting it |
| `uploads_extract` | unpack into `extracted/<name>/` and report every path |
| `uploads_tree` | structure under a path, depth-limited |
| `uploads_stat` | size, type, and whether it needs extracting first |

Every path they return is relative to `/workspace` and can be handed to `read_file` or
`grep_search` unchanged. `uploads_extract` also collapses an archive's single top-level
directory, so `report.zip` becomes `extracted/report/README.md` rather than
`extracted/report/report/README.md` — one less path for the model to get wrong.

Extraction exists as a tool because `exec_shell_command` is deliberately **not** in the
shipped `EXTRA_ARGS`: the model has no shell and cannot unpack an archive itself. `unzip` and
`zstd` ship in the image for exactly this.

The two MCP servers have independent switches under one master:

```bash
MCP_ENABLE=1           # master; 0 means no MCP config at all
MCP_FILES_ENABLE=1     # uploads_*
MCP_WEB_ENABLE=0       # web_search / web_fetch
```

Both are stdio children of `llama-server`, not services. This release parses only
`command`, `cwd`, `args`, `env` and `timeout_ms` from its MCP config
([`server-mcp.cpp:147`](https://github.com/ggml-org/llama.cpp/blob/b11461/tools/server/server-mcp.cpp#L147))
and warns-and-skips any entry without a `command` — there is no HTTP transport, so there is
nothing to point at another container.

### Token, and what it is worth

`UPLOAD_TOKEN` is **empty by default**, which disables the check: anyone who can reach the
port can write to the volume the model reads. That is the intended posture here because
reaching the port already means an SSH tunnel. Set it and uploads need `X-Upload-Token` (the
`/manage/` page has a field that keeps it in the browser only):

```bash
UPLOAD_TOKEN=$(openssl rand -hex 16)
```

Upload names are validated as leaf names and anything with a separator, a `..` or a leading
dot is **refused rather than sanitised** — quietly rewriting a name would make the path
reported back to the model wrong, which is worse than an error it can read.

`/manage/download/` is gated the same way, but by njs rather than by the `alias`: the handler
checks the token and the path and then `internalRedirect`s to an `internal` location, so the
bytes stream out with `sendfile` instead of passing through the JS heap. Serving the alias
directly would have left every uploaded file readable to anyone who guessed its name while
listing and upload still honoured the token.

### One wrinkle: uploaded files are mode 0600

An upload inherits the mode of the nginx body temp file it was renamed from, which is `0600`
owned by the gateway worker. Both containers run as root by default, so the model reads them
fine. It only bites if you change either container to a non-root user — njs has no `chmod`,
so there is nothing in `upload.js` that could widen it. Run both as the same uid if you move
off root.

### One ordering constraint

nginx resolves an upstream server name when it **parses** its config, not per request, so the
gateway cannot start before `glm53f` resolves on the compose network. That is why
`depends_on` is present and set to `service_started` — not `service_healthy`, because the
model server spends up to 15 minutes loading weights and everything under `/manage/` should
answer throughout. Only `/` returns 502 until the upstream is up.

## Web tools

The model can also search and read the web. `llama-server` speaks MCP over stdio, so this is
one script in the image rather than a service: `docker/mcp-web.sh` is an MCP server exposing
`web_search` and `web_fetch`, built out of `curl` and `jq` and nothing else. It shares the
master `MCP_ENABLE` switch with the uploads server above and has its own, which ships **off**
since this is the one that reaches the internet:

```bash
MCP_ENABLE=1              # master
MCP_WEB_ENABLE=1          # this server; 0 in the shipped .env.example
SEARXNG_URL=              # empty = DuckDuckGo lite, no key, no account
SEARCH_RESULTS=8
FETCH_MAX_CHARS=20000
```

The entrypoint renders `/etc/glm53f/mcp.json.template` into a temp dir at start, deletes the
servers whose switch is off with `jq`, and passes `--mcp-servers-config` to `llama-server`
(one template holds every server, rather than a second template that would drift from the
first). If `MCP_ENABLE=1` but every server is off, the flag is not passed at all. The tools
then appear as `web_search` /
`web_fetch` in `GET /tools`, in the built-in Web UI and to any OpenAI client that sends a
`tools` array — llama.cpp prefixes MCP tools with the server name, so they show up under the
server named `web`.

`web_search` returns a numbered list of titles and URLs; `web_fetch` returns a page's
visible text with script, style and tags stripped, truncated at `FETCH_MAX_CHARS`. Point
`SEARXNG_URL` at your own SearXNG to keep queries on your network — it needs `json` in its
`search.formats`.

Worth knowing before you turn it on:

- the MCP server is a child process of `llama-server` **with the same privileges**. This one
  only shells out to `curl`, but the mechanism is as trusted as what you declare
- enabling tools of either kind makes llama.cpp default `--cors-origins` to localhost, which
  changes who can reach `:8080` from a browser
- tool-call formatting is the first thing to degrade as the quant gets smaller. `UD-IQ3_XXS`
  is the floor for agentic use; below that, prose still reads fine while the calls stop parsing.
  The `IQ2_XXS -> IQ3_XXS` step halves KL divergence, which is the measurable version of that
  claim
- **tool-call reliability on this model is not yet measured at any quant.** Reviewers have
  tested it for reasoning and coding quality, but nobody has published a tool-calling pass rate
  the way they have for `DeepSeek-V4-Flash` (6/6, 0 broken calls at IQ3_XXS). If your workload
  is agentic, verify this yourself before trusting it

## CI

> `.github/workflows/build.yml` is **not in this repo yet**. `.github/workflows/` holds only a
> `.temp` placeholder so the folder exists; add `build.yml` from the GitHub web editor ("Add
> file" inside that folder prefills the path), then delete the placeholder. Copy it from
> [`qwen38-server`](https://github.com/onixldlc/qwen38-server/blob/main/.github/workflows/build.yml)
> and change the image name — pushing workflow files needs a PAT with the `workflow` scope.

[`.github/workflows/build.yml`](.github/workflows/build.yml) builds the `gpu` and `cpu`
variants in a matrix off the one Dockerfile, builds the `gateway` image from
`docker/gateway/Dockerfile` in a job of its own, and pushes all three to GHCR.

The gateway's smoke test starts the real entrypoint and drives the njs handlers over HTTP —
`/manage/health`, `/manage/files`, a full upload round trip, the same file back out through
`/manage/download/`, a refused traversing filename and the page itself. That is the only
check that proves the dynamic module actually loaded into that nginx build. It overrides `UPSTREAM` to `127.0.0.1:9`, because nginx resolves an upstream name when
it parses the config and the default `glm53f` does not exist on a runner.

**The `VERSION` file is the trigger.** The workflow runs only on a push to `main` that
changes that file — any other commit costs zero Actions minutes. So a release is one edit,
made from anywhere: the GitHub web editor, github.dev, a phone. No git tag, no local clone,
no credentialed machine.

```
VERSION: 0.1.0 -> 0.2.0        commit to main
  => ghcr.io/onixldlc/glm53f-server:v0.2.0-gpu       (pinned)
     ghcr.io/onixldlc/glm53f-server:v0.2.0-cpu       (pinned)
     ghcr.io/onixldlc/glm53f-server:v0.2.0-gateway   (pinned)
     ghcr.io/onixldlc/glm53f-server:gpu          (moved to this build)
     ghcr.io/onixldlc/glm53f-server:cpu          (moved to this build)
```

The file holds the bare number (`0.2.0`); the workflow adds the `v`. Pushing the same
VERSION twice republishes over those tags, so bump it to keep a pinned build.

No registry secrets to set up — it logs in with the built-in `GITHUB_TOKEN`. Actions tab →
*Build and Push* → **Run workflow** forces a run without touching the file, with an optional
`version` override and a `publish` checkbox you can clear to build and smoke test only.

## Auto-scaling to the host

`AUTO_SCALE=1` (the default) sums total VRAM across every visible card with `nvidia-smi` at
start and picks the quant, the context and the KV cache type to match:

| VRAM | `MODEL_FILE` quant | KV | Context | Used | Free |
|---|---|---|---|---|---|
| 288 GB | `UD-Q6_K_XL` | f16 | 131072 | 281122 MiB | 8878 MiB |
| 240 GB | `UD-Q5_K_XL` | f16 | 131072 | 231982 MiB | 8018 MiB |
| 192 GB | `UD-Q4_K_XL` | f16 | 131072 | 193264 MiB | 6736 MiB |
| 160 GB | `UD-IQ4_XS` | f16 | 131072 | 152365 MiB | 7635 MiB |
| 144 GB | `UD-Q3_K_XL` | f16 | 131072 | 143509 MiB | 6491 MiB |
| 128 GB | `UD-IQ3_XXS` | f16 | 131072 | 117599 MiB | 12401 MiB |
| 112 GB | `UD-Q2_K_XL` | f16 | 131072 | 106492 MiB | 5508 MiB |
| 104 GB | `UD-IQ2_XXS` | f16 | 131072 | 99935 MiB | 5065 MiB |
| 96 GB | `UD-IQ1_M` | f16 | 131072 | 95867 MiB | 4133 MiB |
| 88 GB | `UD-IQ1_S` | f16 | 131072 | 91583 MiB | 4917 MiB |

Figures are with `ENABLE_VISION=1` (mmproj 1110 MiB), measured at each row's *threshold*, so
"Free" is the worst case for that row and still includes the 4096 MiB `FIT_MARGIN_MIB`.

Two things differ from the qwen image and both are worth reading twice:

**Every row gets 131072 at f16.** There is no q8_0 tier. Of 45 layers only 11 are DSA sparse
attention over an MLA path and hold a per-token cache; the other 34 are KDA linear attention,
whose state is a fixed size no matter how long the context gets. So:

```
11 layers x (kv_lora_rank 512 + qk_rope_head_dim 64)  =  6336 elements/token
  f16:  6336 x 2      = 12672 B/token  ->  12.375 MiB per 1024 tokens
  q8_0: 6336 x 1.0625 =  6732 B/token  ->   6.573 MiB per 1024 tokens
```

128K of f16 KV is about **1.6 GiB**. Dropping to q8_0 would save 0.8 GiB against a 112 GiB
model, which is not a trade worth making. `autoscale.sh` still implements the rule — f16 unless
`CTX_TARGET` will not fit in f16, only then q8_0 — so it self-corrects if the estimate is off.

> These KV constants are **derived from the published architecture, not read out of the GGUF
> header** the way the qwen numbers were. Treat them as a starting point and cross-check with
> `autoscale.sh --measure` once the weights are on disk.

**There is a floor, and it is high.** Below roughly 95 GiB of total VRAM no quant of a 320B
model fits on the GPU at all. `autoscale.sh` picks `UD-IQ1_S` and says so loudly rather than
pretending a tier exists:

```
[autoscale] WARNING 81920 MiB is below this model's floor.
[autoscale]   UD-IQ1_S alone is 88775 MiB. Layers will spill to host RAM;
[autoscale]   drop N_GPU_LAYERS until it fits, or rent more VRAM.
```

Each threshold is the real minimum for its row rather than a round number:
`weights + FIT_MARGIN_MIB(4096) + KDA state(34) + mmproj(1110) + 128K f16 KV(1664)`, i.e.
`weights + 6904`. The tier *label* is the rough VRAM class it targets. Both matter, because no
card reports its box number: a 5090 reports 32607 MiB, so four of them are 130428 and not
131072, and comparing against the round number would drop every host a tier.

The logic lives in `docker/autoscale.sh`, not buried in the entrypoint, so you can ask it what
it would do without starting anything:

```bash
$ autoscale.sh                       # decide from nvidia-smi
$ VRAM_MIB=163840 autoscale.sh       # decide for a host you do not have
$ autoscale.sh --measure             # ask llama-fit-params (needs the weights)
```

It prints `KEY=VALUE` lines on stdout and its reasoning on stderr. `AUTO_SCALE=0` pins
`MODEL_FILE` and `CTX_SIZE` from the environment; `VRAM_MIB=N` overrides detection.

### Sharded weights

Every quant here is larger than HuggingFace's per-file limit, so each one is 3-7 shards inside
a per-quant subdirectory:

```
UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf    9 MiB   (index)
UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf   47 GiB
UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00003-of-00004.gguf   46 GiB
UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00004-of-00004.gguf   21 GiB
```

`MODEL_FILE` always names **shard 1**. The entrypoint reads the `-of-0000N` suffix off the
filename, fetches the rest into the same directory, and llama.cpp finds shards 2..N there by
itself. There is no shard-count variable to keep in sync — a hand-pinned `MODEL_FILE` at
`AUTO_SCALE=0` needs nothing extra.

### Pin the quant, auto-size the context

These two compose, and the combination is usually what you want on a host the table
under-fills. `AUTO_SCALE=1` is all-or-nothing — it overwrites `MODEL_FILE`, `CTX_SIZE` and the
KV type together — so pinning the quant while still filling the cards means turning the table
off and delegating only the context:

```bash
AUTO_SCALE=0
MODEL_FILE=UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
CTX_SIZE=auto
```

At `AUTO_SCALE=0` the autoscaler never runs, so nothing can overwrite the quant; `CTX_SIZE=auto`
hands only the context to llama.cpp's own fitter, which measures free memory per device
instead of estimating a pooled total from a table. Two things to keep in mind:

- **You do *not* need to pin `--cache-type-k/v` here.** This is the opposite of the qwen image:
  llama.cpp defaults to `f16`, and on this architecture `f16` is what you want anyway.
- **The fitter will not spill layers to CPU.** `N_GPU_LAYERS` is passed explicitly and the
  fitter only adjusts arguments that are *not* passed, so it shrinks the context but never
  reduces `-ngl`. Pin a quant whose weights actually fit.

### Letting llama.cpp size the context instead

`CTX_SIZE=auto` drops `--ctx-size` from the command line entirely. llama.cpp then sizes the KV
pool itself — `fit_params` is **on by default** (`common/common.h:481`) and reserves
`--fit-target` MiB. It measures the compute buffers rather than estimating them, so it is more
accurate than any table here, and on this model that is worth more than it was on the qwen
image: 320B of MoE weights needs far more per-device scratch than a 27B dense model, which is
why `FIT_MARGIN_MIB` defaults to 4096 rather than llama.cpp's stock 1024.

## Pick the quant

All sizes are the **sum of every shard** of that quant, from the Unsloth repo:

| Min VRAM | Quant | Shards | Size | Perplexity vs BF16 | Same-top-p |
|---|---|---|---|---|---|
| 88 GB | `UD-IQ1_S` | 3 | 86.69 GiB | — | — |
| 96 GB | `UD-IQ1_M` | 3 | 90.88 GiB | — | — |
| 104 GB | `UD-IQ2_XXS` | 4 | 94.85 GiB | 1.883x | 73.2% |
| 112 GB | `UD-Q2_K_XL` | 4 | 101.25 GiB | — | 78% |
| **128 GB** | **`UD-IQ3_XXS`** | 4 | **112.10 GiB** | **1.341x** | **81.8%** |
| 144 GB | `UD-Q3_K_XL` | 4 | 137.40 GiB | — | — |
| 160 GB | `UD-IQ4_XS` | 5 | 146.05 GiB | — | — |
| 192 GB | `UD-Q4_K_XL` | 6 | 185.99 GiB | — | — |
| 240 GB | `UD-Q5_K_XL` | 6 | 223.80 GiB | — | — |
| 288 GB | `UD-Q6_K_XL` | 7 | 271.79 GiB | — | — |

Add 1.08 GB for the vision tower (`ENABLE_VISION=1`) and ~1.6 GiB of f16 KV at 128K on top.

**`UD-IQ3_XXS` is the one to aim at.** The `IQ2_XXS -> IQ3_XXS` step halves KL divergence
(0.7072 to 0.3557), gains 8.56 points of same-top-p agreement and drops the perplexity penalty
from 1.88x to 1.34x, for 17 GiB more weights. Below it the model gets measurably unreliable;
above it the gains are small until `UD-IQ4_XS`.

About 1 GiB of precision-sensitive tensors — the DSA indexer and the KDA gates — are kept
unquantized in every one of these, which is already counted in the sizes above.

## Run

```bash
cp .env.example .env     # pick MODEL_FILE + CTX_SIZE for your host
podman compose up -d     # starts glm53f AND the gateway in front of it
podman logs -f glm53f    # first start downloads 112 GiB into the glm53f-models volume
curl localhost:8080/v1/models
```

That download is the first thing to plan around. At 1 Gbit it is roughly 16 minutes of
transfer before llama.cpp starts reading anything, which is why the healthcheck's
`start_period` is 90m on the GPU stack. On a rented box, pull the weights onto a persistent
volume once and keep it.

`8080` is the **gateway**; `llama-server` is proxied behind it and publishes no port of its
own. The chat UI is still at `/`, and `/manage/` is the file manager. To reach `llama-server`
directly as well, uncomment its `ports:` block in `compose.yaml` — it maps to `8081` so the
two do not collide.

Plain podman:

```bash
podman run --rm --name glm53f \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -v glm53f-models:/models -p 8080:8080 \
  -e MODEL_FILE=UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  -e CTX_SIZE=131072 \
  ghcr.io/onixldlc/glm53f-server:gpu
```

That form skips the gateway, so there is no upload page — fine when you only want the API.
The `uploads_*` tools still work against whatever is on `/workspace`.

**`nvidia.com/gpu=all`, not `=0`.** No single card holds this model, so handing the container
one device drops it to the floor tier and spills layers to host RAM. `nvidia-ctk cdi list`
prints the valid names if you want to pin a subset.

Building them yourself instead:

```bash
podman build -t glm53f-server:dev         -f docker/Dockerfile .
podman build -t glm53f-server:dev-gateway -f docker/gateway/Dockerfile .
```

Both take the repo root as context, which is why the `-f` path is `docker/...` rather than a
`cd` into it. To build against a newer llama.cpp than the pinned `b11461`:

```bash
podman build --build-arg LLAMA_TAG=bNNNNN -f docker/Dockerfile .
```

Support for this architecture is only days old, so a newer tag may carry `glm5-next` fixes that
`b11461` does not.

## Renting a host

There is no "run it on the card you have" option here. Practical configurations, cheapest
first, with what `autoscale.sh` picks for each:

| Host | Total VRAM | Tier | Quant | Notes |
|---|---|---|---|---|
| 3x RTX 5090 | 97821 MiB | 88 | `UD-IQ1_S` | works, but 1-bit — not worth it |
| 4x RTX 5090 | 130428 MiB | 128 | `UD-IQ3_XXS` | the cheap sweet spot |
| 1x H200-141 | 144384 MiB | 128 | `UD-IQ3_XXS` | single card, no interconnect to worry about |
| **2x A100-80** | **163840 MiB** | **160** | **`UD-IQ4_XS`** | **best quality per dollar** |
| 2x H100-80 | 163840 MiB | 160 | `UD-IQ4_XS` | faster than A100, costs more |
| 8x RTX 5090 | 260856 MiB | 240 | `UD-Q5_K_XL` | diminishing returns above IQ4 |

A note on topology: llama.cpp **layer-splits** rather than tensor-paralleling, so each token
walks the layers in sequence across devices and you get roughly one card's memory bandwidth,
not the aggregate. Four 5090s at 1.79 TB/s each do not add up to 7 TB/s. That is why 2x A100-80
(HBM2e, ~2.0 TB/s, NVLink) is the better buy than 4x 5090 (GDDR7, 1.79 TB/s, PCIe) despite
having fewer cards: more VRAM for a better quant, slightly more per-card bandwidth, and fewer
hops. The 5090's FP4/FP8 hardware goes unused — GGUF does not touch it.

Only 18B of 320B parameters are active per token, so decode is far quicker than the parameter
count suggests. For calibration, someone measured **15.5-17.4 tok/s** running the comparable
`DeepSeek-V4-Flash` IQ3_XXS on a DGX Spark, whose unified memory is ~273 GB/s — roughly a
seventh of an A100's bandwidth.

## CPU-only stack

For a VPS with no GPU. Same weights, same server, `-ngl 0`.

```bash
cd cpu
podman compose up -d --build
```

A 320B MoE is a stranger case on CPU than a dense 27B was. Only 18B of parameters are active
per token, so the arithmetic per token is closer to an 18B dense model than to a 320B one — but
every one of the 320B has to be resident, because the router can reach any expert on any token.
This trades "needs a huge GPU" for "needs a huge amount of RAM".

What is tuned differently in `cpu/.env`:

| Knob | Value | Why |
|---|---|---|
| `N_GPU_LAYERS` | `0` | everything on host cores |
| `CPU_LIMIT` | `3.0` | 50% of a 6 vCPU box |
| `THREADS` / `THREADS_BATCH` | `3` | matched to the quota |
| `ENABLE_VISION` | `0` | the vision tower is slow without a GPU |
| `CTX_SIZE` | `32768` | prompt processing, not RAM, is the limit here |
| `MEM_LIMIT` | `124g` | 112.1 GiB weights + ~0.4 GiB f16 KV + overhead |
| `--reasoning-effort` | `low` | thinking tokens decode one at a time; on CPU they are the most expensive thing you can ask for |
| KV cache | `f16` | only 11 of 45 layers hold a per-token cache; nothing to reclaim |

### Capping CPU usage

`cpus:` is a CFS quota, not a core count. The kernel gives the container that much CPU *time*
per scheduling period and freezes it for the rest, so `CPU_LIMIT=3.0` on a 6 vCPU box is a hard
50% ceiling on total usage — it can never pin the whole machine, no matter how many threads
llama.cpp spawns. Scale it to the box: 4 vCPU → `2.0`, 8 vCPU → `4.0`.

Keep `THREADS` equal to `CPU_LIMIT`. Running 6 threads under a 3.0 quota is not faster than
running 3 — the same total CPU time gets split across twice as many threads, each of which is
frozen half the time, which wastes cache locality and adds latency.

`cpu_shares: 512` is a different dial: it only decides who wins when the host is contended,
leaving the hard cap untouched. `mem_limit` bounds RAM. Swap in `cpuset: "0-2"` for hard core
pinning if you want the inference confined to specific cores rather than given a time share.

Sizing: 112.1 GiB of weights at `UD-IQ3_XXS` + ~0.4 GiB of f16 KV at 32768 + overhead, so a
128 GB box is the realistic floor. `UD-IQ2_XXS` (94.85 GiB) brings that to about 104 GB and
`UD-IQ1_S` (86.69 GiB) to about 96 GB. It is bound by memory bandwidth rather than clock
speed, so a box with faster RAM beats one with more cores. Measure yours before committing:

```bash
podman exec glm53f-cpu llama-bench -ngl 0 -t 3 \
  -m /models/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
```

## Getting the weights in

The entrypoint pulls each shard with plain resumable `curl` straight from the HF CDN —
no `hf` CLI, no Python, no Xet. A stalled transfer resumes from the byte it reached
(`curl -C -` against a `.part` file) and retries up to `DOWNLOAD_ATTEMPTS` times; a transfer
that drops under 1 KB/s for 60 s is cut and resumed rather than left hanging. That matters
more here than on a single-file model: a quant is 3-7 files of up to 50 GB each.

**Or download them yourself** and drop them in. **A file that is already in the volume is
never downloaded again** — the entrypoint only checks that it is present, non-empty and
readable. The `GGUF` magic bytes are checked too, but a mismatch is a warning, not a reason
to spend 112 GiB of bandwidth again; delete the file if you want it refetched. A file the
container's uid cannot read is a hard error naming the fix (ownership, or `:Z` on the volume
mount if SELinux is relabelling it) rather than a silent re-download.

Shards must keep their per-quant subdirectory, because llama.cpp looks for shards 2..N in the
directory it loaded shard 1 from:

```bash
Q=UD-IQ3_XXS
mkdir -p "$Q" && cd "$Q"
for i in 1 2 3 4; do
  curl -L -C - -O "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/$Q/GLM-5.3-Flash-$Q-0000$i-of-00004.gguf"
done
cd ..
curl -L -C - -O https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/mmproj-BF16.gguf

podman run --rm -v glm53f-models:/models -v "$PWD":/host:z \
  docker.io/library/busybox sh -c 'cp -r /host/UD-* /host/*.gguf /models/'
```

Resume a half-finished host download by re-running the same `curl -C -` loop.

## Weights volume

Weights live in the named volume `glm53f-models`, so `compose down` / `up` never re-downloads.
It holds the per-quant subdirectories of `.gguf` shards and nothing else — `verify.sh` fails if
any non-`.gguf` file turns up in there.

```bash
podman volume inspect glm53f-models --format '{{.Mountpoint}}'
podman run --rm -v glm53f-models:/m docker.io/library/busybox find /m -name '*.gguf'
podman volume rm glm53f-models    # only if you want the 112 GiB back
```

## Verify

```bash
podman exec -it glm53f verify.sh
```

Checks GPU passthrough, that the binary runs, that the build is new enough for the `--tools`
in `EXTRA_ARGS`, **every shard** on disk, a weights-only volume, and `/health`. Checking only
shard 1 would pass on a half-finished download, because llama.cpp does not open shards 2..N
until it has read the index in shard 1.

## Sampling

```
--temp 0.95 --top-p 1.0 --reasoning-effort high
--tools read_file,write_file,edit_file,file_glob_search,grep_search,get_info
```

Set via `EXTRA_ARGS`. `temp 0.95 / top-p 1.0` is Z.ai's **DeepSWE** preset, the one meant for
code and analysis; their general-purpose default is `temp 1.0 / top-p 0.95`. No `top-k` or
`min-p` — the model card does not specify them.

`--reasoning-effort` maps onto the model's thinking modes; the template default is `max`.
`high` is shipped here because reviewers running it through OpenCode reported long traces and
needing to nudge it along with continuations. The flag parses
`default`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` (`common/arg.cpp:3718`), but this model
only documents `low`, `high` and `max`.

No `--cache-type-k/v` on purpose: llama.cpp defaults to `f16`, which is both affordable and
correct here. This is the opposite of the qwen image, where `q8_0` had to be pinned.

The model also ships an MTP (multi-token prediction) head for speculative decoding, which
llama.cpp drives through `--spec-type` / `--spec-draft-n-max`; the model card suggests stopping
at about 2 draft tokens. It is left off because it needs a second weights file this image does
not fetch.
