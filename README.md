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
docker/        Dockerfile + entrypoint.sh + verify.sh + gateway.sh + mcp-web.sh + mcp-files.sh
docker/gateway/  front door assets: nginx.conf.template, upload.js, index.html
gpu/           compose.yaml + compose-dev.yaml + .env  — CUDA
cpu/           compose.yaml + compose-dev.yaml + .env  — CPU-only, for a VPS
VERSION        the release number; changing it is what triggers a publish
.github/       workflow that builds both images and pushes them to GHCR
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

## Uploads and the front door

The agent tools above can read anything on `/workspace`, but only once the model knows the
path. Nothing tells it that a zip arrived, where it landed, or what is inside. That is the
gap the front door and the `uploads_*` tools close.

**One image, one port, two processes.** nginx runs **inside** this image, next to
`llama-server`: nginx owns the published `8080` and `llama-server` binds `127.0.0.1:8081`
behind it. There is no gateway image, no second service and no compose network between the
two — `docker/gateway.sh` renders the config and starts nginx, then `entrypoint.sh` goes on
to fetch weights and `exec`s `llama-server` as the container's main process. nginx is
started first and daemonised, so `/manage/` answers for the whole of a first run's download
while the container still lives and dies with the model.

Everything nginx serves itself lives under `/manage/`, and everything outside it is
llama.cpp:

```
:8080/manage/                the file manager page
:8080/manage/files           JSON listing       ?path= &depth=
:8080/manage/upload          PUT a raw body     ?name=
:8080/manage/remove          DELETE one upload  ?name=
:8080/manage/download/<path> read a file back out
:8080/manage/health          nginx liveness, no upstream involved

:8080/                       llama.cpp's WebUI and API (/v1/*, /health, /props, ...)
```

**The page and the API are the same endpoints.** `/manage/` is a file manager — a listing in
the middle, drop files anywhere on it to upload, click a file to download it, click a folder
to walk into it — and every action it takes is one of the URLs above. There is no private
API behind the UI, so anything the page can do, `curl` and the MCP server can do with the
same call.

The single prefix is also why the upstream can grow any route it likes without colliding
with this image.

njs is an nginx **module**, so the upload handler and the reverse proxy are the same process
— there is nothing here to supervise. `ngx_http_js_module` is a *dynamic* module and only
loads into the exact nginx build it was compiled against, which is why the packages come from
the Ubuntu archive the image is already built on: `nginx` 1.24.0 and `libnginx-mod-http-js`
0.8.2 resolve together, and `gateway.sh` runs `nginx -t` at startup so a mismatch fails
loudly instead of silently skipping every route above.

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

**`workspace_path` is the point.** nginx writes to the volume that is also
`llama-server`'s working directory — the same directory, in the same container — so a path
reported here is already relative to what the built-in file tools resolve against. Nothing translates paths between the two halves, because
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
owned by the nginx worker. Both nginx and `llama-server` run as root in this image, so the
model reads them fine. It only bites if you move either off root — njs has no `chmod`, so
there is nothing in `upload.js` that could widen it. Run both as the same uid if you do.

### Start order, and why it is no longer a constraint

nginx resolves an upstream server name when it **parses** its config, not per request. With a
gateway container that meant it could not start before `glm53f` resolved on the compose
network, and a `depends_on: service_started` to say so. In one container the upstream is
`127.0.0.1:8081`, a literal that always parses, so nginx starts unconditionally and starts
*first*: `/manage/` answers from a few seconds in, through the whole 87-272 GiB download and
load, and only `/` returns 502 until `llama-server` is listening.

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
> file" inside that folder prefills the path), then delete the placeholder. The content to
> paste is in this checkout at `.github/workflows/build.yml`, which is deliberately untracked
> — pushing workflow files needs a PAT with the `workflow` scope. It is *not* interchangeable
> with [`qwen38-server`](https://github.com/onixldlc/qwen38-server/blob/main/.github/workflows/build.yml)'s
> copy any more: that one still builds a separate gateway image, which this stack does not have.

[`.github/workflows/build.yml`](.github/workflows/build.yml) builds the `gpu` and `cpu`
variants in a matrix off the one Dockerfile and pushes both to GHCR. There is no third job:
the front door is part of these images, so it is built and tested with them.

Each variant gets two smoke tests. The first checks the binaries. The second starts the
**same image** with `--entrypoint /usr/local/bin/gateway.sh ... foreground`, which brings up
nginx with no model and no download in front of it, and drives the njs handlers over HTTP —
`/manage/health`, `/manage/files`, a full upload round trip, the same file back out through
`/manage/download/`, a refused traversing filename, the page itself, and finally that the
upload really landed in `/workspace`, the model's working directory. That is the only check
that proves the dynamic module actually loaded into that nginx build. It overrides `UPSTREAM`
to `127.0.0.1:9` because nothing in the test requests a proxied path.

**The `VERSION` file is the trigger.** The workflow runs only on a push to `main` that
changes that file — any other commit costs zero Actions minutes. So a release is one edit,
made from anywhere: the GitHub web editor, github.dev, a phone. No git tag, no local clone,
no credentialed machine.

```
VERSION: 0.1.0 -> 0.2.0        commit to main
  => ghcr.io/onixldlc/glm53f-server:v0.2.0-gpu       (pinned)
     ghcr.io/onixldlc/glm53f-server:v0.2.0-cpu       (pinned)
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
start, then picks the **largest quant that holds 131072 context**, using `f16` KV if it fits
and `q8_0` if that is what makes that quant fit.

**128K is a floor, not a target.** The context never shrinks — the quant is what flexes.

| Quant | Shards | Weights | Base, q8_0 KV | Base, f16 KV |
|---|---|---|---|---|
| `UD-Q6_K_XL` | 7 | 271.79 GiB | 283776 MiB | 287616 MiB |
| `UD-Q5_K_XL` | 6 | 223.80 GiB | 234636 MiB | 238476 MiB |
| `UD-Q4_K_XL` | 6 | 185.99 GiB | 195918 MiB | 199758 MiB |
| **`UD-IQ4_XS`** | 5 | **146.05 GiB** | **155019 MiB** | **158859 MiB** |
| `UD-Q3_K_XL` | 4 | 137.40 GiB | 146163 MiB | 150003 MiB |
| `UD-IQ3_XXS` | 4 | 112.10 GiB | 120253 MiB | 124093 MiB |
| `UD-Q2_K_XL` | 4 | 101.25 GiB | 109146 MiB | 112986 MiB |
| `UD-IQ2_XXS` | 4 | 94.85 GiB | 102589 MiB | 106429 MiB |
| `UD-IQ1_M` | 3 | 90.88 GiB | 98521 MiB | 102361 MiB |
| `UD-IQ1_S` | 3 | 86.69 GiB | 94237 MiB | 98077 MiB |

Those are `weights + 128K of KV + mmproj(1110)`. Drop `ENABLE_VISION` and each falls by 1110.

### The margin is per device, and that is not a detail

On top of the base figure, **every GPU keeps `FIT_MARGIN_MIB` (4096) free** — the same meaning
llama.cpp gives `--fit-target`. A 2-card host therefore gives up 8192 MiB of its total, a
4-card host 16384.

This is not bookkeeping. llama.cpp splits **layers** across devices, so a reserve held as one
pooled number leaves each card a fraction of it, and whichever card also carries the mmproj
less still. Treating it as a total is how a 163840 MiB host gets handed a quant that needs
158859: on paper 4981 MiB spare, in reality ~2 GB per card before a single compute buffer is
allocated, and an abort after the weights are already resident.

The KV numbers are the same ones the qwen image used — **64 MiB per 1024 tokens at f16, 34 MiB
at q8_0** — measured rather than inferred from this model's layer geometry. Overestimating KV
costs at most a quant tier; underestimating it costs a failed load.

`autoscale.sh` takes the biggest quant that holds 128K **with q8_0 KV**, then upgrades that row
to f16 if f16 also fits. That order matters: a bigger quant at `q8_0` beats a smaller one at
`f16`, because the KV type costs far less quality than a whole quant step. Trying f16 first did
the opposite — it spent 3840 MiB on the KV type and then had to drop a quant to pay for it.

### The vision tower is not split either

llama.cpp spreads the layers and their KV evenly, but the mmproj goes on **device 0 alone**.
So the condition that has to hold is per device:

```
(weights + kv) / N + mmproj + margin  <=  VRAM / N
```

which, multiplied out, is `weights + kv + N*mmproj + N*margin <= VRAM`. Both the margin **and
the vision tower** multiply by the card count. Charging the 1110 MiB tower once to the pool is
what left device 0 short on the 8-card box: `free ~4718 MiB per device` read fine, but 1110 of
those MiB only existed on the other seven cards. `autoscale.sh` now reports device 0 separately,
because device 0 is the card that runs out first:

```
[autoscale] 196608 MiB over 8 device(s) -> UD-IQ4_XS (5 shards) + 131072 ctx @ q8_0 KV, needs 190 GiB
[autoscale]   weights 149557 MiB + kv 4352 MiB + mmproj 1110 MiB = 155019 MiB of 196608 MiB
[autoscale]   free ~5338 MiB per device, 4228 MiB on device 0 (it holds the
[autoscale]   vision tower) — reserve is 4096 MiB per device
```

What that buys in practice:

| Host | Total VRAM | Cards | Picks | KV | Free on device 0 |
|---|---|---|---|---|---|
| 4x A100-80 | 327680 MiB | 4 | `UD-Q6_K_XL` | f16 | 9184 MiB |
| 2x H200-141 | 288768 MiB | 2 | `UD-Q5_K_XL` | f16 | 24591 MiB |
| 8x RTX 5090 | 260856 MiB | 8 | `UD-Q4_K_XL` | f16 | 6666 MiB |
| **8x RTX 4090** | **196608 MiB** | 8 | **`UD-IQ4_XS`** | **q8_0** | 4228 MiB |
| 2x A100-80 | 163840 MiB | 2 | `UD-Q3_K_XL` | f16 | 6364 MiB |
| 1x H200-141 | 144384 MiB | 1 | `UD-IQ3_XXS` | f16 | 20291 MiB |
| 4x RTX 5090 | 130428 MiB | 4 | `UD-Q2_K_XL` | q8_0 | 4488 MiB |
| 3x RTX 5090 | 97821 MiB | 3 | `UD-IQ1_S` | q8_0 | **does not fit** |

The 8-card row is the one to read: `UD-IQ4_XS` is kept, but at `q8_0` KV, which leaves device 0
4228 MiB — over the reserve by 132 MiB. With the tower charged once it took f16 KV instead, and
device 0 was left ~3600 MiB, under its own reserve. That is a load that dies after the weights
are already resident.

**There is a floor, and it is high.** Once each card keeps its reserve, 3x RTX 5090 holds
nothing at 128K. `autoscale.sh` says so loudly rather than quietly shrinking your context:

```
[autoscale] WARNING 97821 MiB cannot hold 131072 context at any quant.
[autoscale]   Smallest is UD-IQ1_S + q8_0 KV = 106525 MiB. Layers will spill to
[autoscale]   host RAM; drop N_GPU_LAYERS, lower CTX_TARGET, or rent more VRAM.
```

### The disk is a budget too, and filling it takes the host with it

A host can have the cards for a quant and nowhere to put the file. That is not a download
failure — on a rented box with the weights on the container disk, a **full filesystem stops
everything that needs to write, sshd included**, and the instance is gone rather than the file.

So `autoscale.sh` reads `df` at `MODEL_DIR` and sizes the quant to the disk as well as the VRAM,
keeping `DISK_RESERVE_MIB` (8192) free. A re-run counts what the directory already holds, so
existing shards are not mistaken for missing space.

Mind that **quants are quoted in GiB and disks are sold in GB** — a factor of 1.074 that is
large enough to turn "it fits" into a dead instance. `UD-IQ4_XS` is 146.05 GiB, which is
**156.8 GB**: a 150 GB container disk cannot hold it.

| Quant | Weights | = GB | Container disk to ask for |
|---|---|---|---|
| `UD-Q6_K_XL` | 271.79 GiB | 291.9 GB | 320 GB |
| `UD-Q5_K_XL` | 223.80 GiB | 240.3 GB | 270 GB |
| `UD-Q4_K_XL` | 185.99 GiB | 199.7 GB | 230 GB |
| **`UD-IQ4_XS`** | **146.05 GiB** | **156.8 GB** | **200 GB** |
| `UD-Q3_K_XL` | 137.40 GiB | 147.5 GB | 170 GB |
| `UD-IQ3_XXS` | 112.10 GiB | 120.4 GB | 150 GB |
| `UD-Q2_K_XL` | 101.25 GiB | 108.7 GB | 130 GB |
| `UD-IQ2_XXS` | 94.85 GiB | 101.9 GB | 130 GB |
| `UD-IQ1_M` | 90.88 GiB | 97.6 GB | 120 GB |
| `UD-IQ1_S` | 86.69 GiB | 93.1 GB | 120 GB |

The last column is weights + the 1.08 GB vision tower + the 8 GiB reserve, rounded up to a size
you can actually rent.

Given a disk, the quant steps down to fit it and says so:

```
[autoscale] disk capped the choice: UD-Q6_K_XL and larger do not fit in
[autoscale]   143051 MiB at /models (keeping 8192 MiB free)
[autoscale] 196608 MiB over 8 device(s) -> UD-IQ3_XXS (4 shards) + 131072 ctx @ f16 KV, needs 160 GiB
```

If not even `UD-IQ1_S` fits, it refuses to start rather than downloading into a wall — before a
single byte is fetched, exit code 2:

```
[autoscale] FAILED 38146 MiB of disk at /models is not enough for any quant.
[autoscale]   Smallest is UD-IQ1_S at 88775 MiB + 9302 MiB of mmproj and reserve.
[autoscale]   Give the container a bigger disk, or mount a volume at /models.
[autoscale]   Refusing to start: a download that fills this filesystem takes the host
[autoscale]   down with it, sshd included.
```

Each shard is checked individually as well: the entrypoint `HEAD`s the URL and compares the
`Content-Length` with what is free, so a disk that fills up mid-quant stops at the file boundary
instead of at the last byte of free space. A server that will not answer a `HEAD` just skips
that check.

### How long the download takes, and how to make it shorter

On a rented box the download runs on the GPU clock: Vast bills the card rate for
every minute the instance is up, whatever the cards are doing. So the entrypoint
prices the job before starting it, and reports progress against that total:

```
[glm53f] model is sharded: 5 files
[glm53f] to fetch: 5 of 5 files, 156.8 GB total
[glm53f]   at 100 MB/s that is 26m08s, at 250 MB/s 10m27s
[glm53f]   16 connections per file (DOWNLOAD_CONNS), progress every 30s
[glm53f] downloading ...-00002-of-00005.gguf — 50.0 GB over 16 connection(s) (attempt 1/1)
[glm53f] ...-00002-of-00005.gguf: 12.4 GB of 50.0 GB (24%) at 214 MB/s, file 2m55s left, all 8m12s left
```

`UD-IQ4_XS` is five shards and they are not even:

| shard | GB |
|---|---|
| `00001-of-00005` | 0.01 — metadata only, finishes instantly |
| `00002-of-00005` | 49.99 |
| `00003-of-00005` | 49.61 |
| `00004-of-00005` | 49.49 |
| `00005-of-00005` | 7.73 |
| **total** | **156.83** |

That first 9 MB shard is why a download can look like it stalled: shard 1 lands in
a second and then three 50 GB files follow.

**One connection is the bottleneck, not the link.** HF's CDN shapes per
connection — a single stream lands around 50-200 MB/s no matter what the host's
`inet_down` says — so the image ships `aria2` and pulls each shard over
`DOWNLOAD_CONNS` (16) connections. `DOWNLOAD_CONNS=1`, or an image without
aria2, falls back to single-stream curl. Resume works either way: an interrupted
transfer continues from the `.part` file rather than starting over.

To see what your host can actually do before trusting a number:

```sh
U=https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00005-of-00005.gguf
curl -sL -r 0-2000000000 -o /dev/null -w '1 conn: %{speed_download} B/s\n' "$U"
```

Divide 156,830,000,000 by that to get seconds for the whole quant.

And once the weights are down: **stop the instance, do not destroy it**. A
stopped Vast instance keeps its container disk at the storage rate (200 GB is
about $1.78/day) and starts again with the shards already there. Destroying it
means paying for the download twice.

### When the table is still wrong

A table cannot measure compute buffers, and llama.cpp's own fitter — which can — refuses to run
when `--n-gpu-layers` is passed explicitly (`n_gpu_layers already set by user to 99, abort`),
which this image does. So the estimate can still be optimistic.

The entrypoint therefore does not strand the host on a bad guess. If `llama-server` exits
within `GRACE_SECONDS` (180), it is restarted on the **same weights** with:

1. `q8_0` KV at full context
2. `q8_0` KV at half context
3. `q8_0` KV at a quarter

Nothing is re-downloaded — a retry costs seconds, not another 157 GB. If it stayed up past the
grace period and then exited, that is a real crash, not a sizing problem, and it is not retried.
When every rung fails the entrypoint stops nginx too, so the container exits instead of serving
`/manage/` and a permanent 502 while looking alive.

Four cards is the real entry point.

The logic lives in `docker/autoscale.sh`, not buried in the entrypoint, so you can ask it what
it would do without starting anything:

```bash
$ autoscale.sh                       # decide from nvidia-smi
$ VRAM_MIB=163840 autoscale.sh       # decide for a host you do not have
$ CTX_TARGET=262144 autoscale.sh     # raise the floor; the table shifts down
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

Note the catch: that fitter **declines to run when `--n-gpu-layers` is passed**, and this image
passes it (`N_GPU_LAYERS=99`). The log says so plainly — `common_fit_params: failed to fit
params to free device memory: n_gpu_layers already set by user to 99, abort`. So on a normal
boot nothing measures the real buffers, and `autoscale.sh`'s table is the only estimate in
play. That is why the per-device margin and the entrypoint's step-down ladder both exist.

## Pick the quant

The table under [Auto-scaling](#auto-scaling-to-the-host) is the sizing reference — it lists
every quant with its shard count, weight total and the minimum VRAM for 128K at each KV type.

**`UD-IQ3_XXS` is the one to aim at.** The `IQ2_XXS -> IQ3_XXS` step halves KL divergence
(0.7072 to 0.3557), gains 8.56 points of same-top-p agreement and drops the perplexity penalty
from 1.88x to 1.34x, for 17 GiB more weights:

| Quant | Perplexity vs BF16 | Same-top-p | KL divergence |
|---|---|---|---|
| `UD-IQ3_XXS` | 1.341x | 81.8% | 0.3557 |
| `UD-Q2_K_XL` | — | 78% | — |
| `UD-IQ2_XXS` | 1.883x | 73.2% | 0.7072 |

Below `IQ3_XXS` the model gets measurably unreliable; above it the gains are small until
`UD-IQ4_XS`. Nothing above `IQ4_XS` has published quality figures.

About 1 GiB of precision-sensitive tensors — the DSA indexer and the KDA gates — are kept
unquantized in every one of these, which is already counted in the weight totals.

## Run

```bash
cp .env.example .env     # pick MODEL_FILE + CTX_SIZE for your host
podman compose up -d     # one container: nginx on 8080, llama-server behind it
podman logs -f glm53f    # first start downloads 112 GiB into the glm53f-models volume
curl localhost:8080/v1/models
```

That download is the first thing to plan around. At 1 Gbit it is roughly 16 minutes of
transfer before llama.cpp starts reading anything, which is why the healthcheck's
`start_period` is 90m on the GPU stack. On a rented box, pull the weights onto a persistent
volume once and keep it.

`8080` is **nginx**; `llama-server` listens on `127.0.0.1:8081` inside the container and is
proxied. The chat UI is still at `/`, and `/manage/` is the file manager. To reach
`llama-server` without the proxy, publish its port too (`-p 8081:8081` with `HOST=0.0.0.0`),
or set `GATEWAY_ENABLE=0` with `HOST=0.0.0.0 PORT=8080` to drop nginx entirely — which also
drops `/manage/`.

Plain podman:

```bash
podman run --rm --name glm53f \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -v glm53f-models:/models -p 8080:8080 \
  -e MODEL_FILE=UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  -e CTX_SIZE=131072 \
  ghcr.io/onixldlc/glm53f-server:gpu
```

That form has the upload page too: nginx is in the image, so a plain `podman run` publishing
`8080` serves `/manage/` and proxies `/` with nothing extra to start. Add
`-e GATEWAY_ENABLE=0 -e HOST=0.0.0.0 -e PORT=8080` if you want `llama-server` alone on the
port; the `uploads_*` tools still work against whatever is on `/workspace` either way.

**`nvidia.com/gpu=all`, not `=0`.** No single card holds this model, so handing the container
one device drops it to the floor tier and spills layers to host RAM. `nvidia-ctk cdi list`
prints the valid names if you want to pin a subset.

Building them yourself instead:

```bash
podman build -t glm53f-server:dev -f docker/Dockerfile .
```

One image, nginx included. It takes the repo root as context, which is why the `-f` path is
`docker/...` rather than a `cd` into it. To build against a newer llama.cpp than the pinned `b11461`:

```bash
podman build --build-arg LLAMA_TAG=bNNNNN -f docker/Dockerfile .
```

Support for this architecture is only days old, so a newer tag may carry `glm5-next` fixes that
`b11461` does not.

## Renting a host

There is no "run it on the card you have" option here. Practical configurations, cheapest
first, with what `autoscale.sh` picks for each:

| Host | Total VRAM | Quant | KV | Disk | Notes |
|---|---|---|---|---|---|
| 3x RTX 5090 | 97821 MiB | `UD-IQ1_S` | q8_0 | 120 GB | does not hold 128K — see the floor above |
| 4x RTX 5090 | 130428 MiB | `UD-Q2_K_XL` | q8_0 | 130 GB | the cheap floor that actually runs |
| 1x H200-141 | 144384 MiB | `UD-IQ3_XXS` | f16 | 150 GB | single card, no interconnect to worry about |
| **2x A100-80** | **163840 MiB** | **`UD-Q3_K_XL`** | **f16** | **170 GB** | **best quality per dollar** |
| 2x H100-80 | 163840 MiB | `UD-Q3_K_XL` | f16 | 170 GB | faster than A100, costs more |
| 8x RTX 4090 | 196608 MiB | `UD-IQ4_XS` | q8_0 | 200 GB | 8 cards, so 8x4096 of reserve + the tower on card 0 |
| 8x RTX 5090 | 260856 MiB | `UD-Q4_K_XL` | f16 | 230 GB | diminishing returns above IQ4 |
| 4x A100-80 | 327680 MiB | `UD-Q6_K_XL` | f16 | 320 GB | as good as this image goes |

**Size the container disk from that column, not from the GiB figure.** Every row is a real
`autoscale.sh` run, and the disk column is weights + tower + reserve converted to the GB the
host is actually sold in. A 150 GB disk against a 146.05 GiB quant is 157 GB of file in 150 GB
of space, and what you lose is the whole instance, not the download.

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
| `MEM_LIMIT` | `124g` | 112.1 GiB weights + 1 GiB q8_0 KV at 32768 + overhead |
| `--reasoning-effort` | `low` | thinking tokens decode one at a time; on CPU they are the most expensive thing you can ask for |
| KV cache | `q8_0` | halves cache RAM, and RAM is the binding constraint here |

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

Sizing: 112.1 GiB of weights at `UD-IQ3_XXS` + 1 GiB of q8_0 KV at 32768 + overhead, so a
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
