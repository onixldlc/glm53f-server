# GLM-5.3-Flash server — stock upstream llama.cpp.
# 320B-A18B MoE, MIT, natively multimodal, 1M context. The GGUFs are ordinary
# K-quants (Unsloth Dynamic UD-*), so no fork is needed: the official release
# tarball already carries llama-server and its shared objects.
# Base and release asset switch together: CUDA for the GPU build, plain ubuntu for the
# CPU-only build (see cpu/compose-dev.yaml). Defaults are the GPU build.
ARG BASE_IMAGE=docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# Release of https://github.com/ggml-org/llama.cpp and which asset of it to install.
# b11461 (2026-10-07) postdates PR #27773 "add GLM-5.3-Flash (GLM5-Next) support",
# merged 2026-09-30 as 649dcb1036fd: LLM_ARCH_GLM5_NEXT / "glm5-next", the KDA
# metadata keys and the DSA indexer tensors are all in this tree. Same tag the
# qwen image used, so --tools and the MCP config behave identically.
# Support for this arch is days old; newer tags carry more glm5-next fixes:
#   podman build --build-arg LLAMA_TAG=bNNNNN ...
ARG LLAMA_TAG=b11461
ARG LLAMA_FLAVOR=ubuntu-cuda-12.8-x64

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends ca-certificates curl tar libgomp1 libcurl4 jq unzip zstd

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -fsSL -o /tmp/llama.tar.gz "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-${LLAMA_FLAVOR}.tar.gz"

RUN mkdir -p /app/bin

RUN tar -xzf /tmp/llama.tar.gz -C /app/bin --strip-components=1

RUN rm -f /tmp/llama.tar.gz

RUN /app/bin/llama-server --version || true

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/verify.sh /usr/local/bin/verify.sh
COPY docker/autoscale.sh /usr/local/bin/autoscale.sh

# MCP stdio server for web_search / web_fetch. llama-server spawns it as a
# child process, so it ships in the image rather than as a separate service.
COPY docker/mcp-web.sh /usr/local/bin/mcp-web

# MCP stdio server for the uploads volume. Tells the model what was uploaded
# and where it landed, so the built-in file tools have a path to work with.
COPY docker/mcp-files.sh /usr/local/bin/mcp-files

COPY docker/mcp.json.template /etc/glm53f/mcp.json.template

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/verify.sh /usr/local/bin/autoscale.sh /usr/local/bin/mcp-web /usr/local/bin/mcp-files

# The release tarball carries its own shared objects next to the binaries.
ENV LD_LIBRARY_PATH=/app/bin:/usr/local/cuda/lib64
ENV PATH=/app/bin:$PATH
# Runtime knobs, all overridable from compose / --env.
# Weights are pulled with plain resumable curl from the HF CDN.
ENV MODEL_BASE_URL=https://huggingface.co
ENV MODEL_REPO=unsloth/GLM-5.3-Flash-GGUF
# Shard 1 of the quant; the entrypoint reads "-of-0000N" off the name and fetches
# the rest into the same directory, which is where llama.cpp looks for them.
ENV MODEL_FILE=UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
ENV MMPROJ_FILE=mmproj-BF16.gguf
ENV ENABLE_VISION=1
ENV MODEL_DIR=/models
# 1 = read the card's VRAM at start and pick MODEL_FILE / CTX_SIZE / KV type to
# match (see entrypoint.sh). 0 = use MODEL_FILE and CTX_SIZE as given.
ENV AUTO_SCALE=1
# VRAM to assume instead of asking nvidia-smi, in MiB. Empty = detect.
ENV VRAM_MIB=
# Headroom left free on each device, in MiB. Matches llama.cpp's --fit-target.
# 320B of MoE weights needs far more scratch than a 27B dense model: 4x the
# llama.cpp stock 1024. autoscale.sh assumes this same number.
ENV FIT_MARGIN_MIB=4096
# A number, or 'auto' to drop --ctx-size and let llama.cpp's own fitter size the
# KV pool against free VRAM. Ignored when AUTO_SCALE=1 picks a value.
# 131072 is the floor autoscale.sh sizes the quant around, not a target: it picks
# the largest quant that holds this much context, f16 KV if it fits and q8_0 if
# that is what makes that quant fit. Override with CTX_TARGET.
ENV CTX_SIZE=131072
ENV N_GPU_LAYERS=99
# 0 on the CPU build; empty means "let llama.cpp pick" (cores - 1 is usually right).
ENV THREADS=
ENV THREADS_BATCH=
ENV FLASH_ATTN=on
ENV HOST=0.0.0.0
ENV PORT=8080
ENV DOWNLOAD_ATTEMPTS=20
ENV EXTRA_ARGS=
# Tool servers, off by default. 1 renders the MCP config and hands llama-server
# --mcp-servers-config. Which servers that config ends up holding is the two
# switches below; MCP_ENABLE=0 means none of them regardless.
ENV MCP_ENABLE=0
# web_search / web_fetch.
ENV MCP_WEB_ENABLE=1
# uploads_list / uploads_peek / uploads_extract / uploads_tree / uploads_stat.
ENV MCP_FILES_ENABLE=1
# Empty = DuckDuckGo's lite endpoint, no key. Point at your own SearXNG to
# keep queries on your network (its settings.yml needs `json` in search.formats).
ENV SEARXNG_URL=
ENV SEARCH_RESULTS=8
ENV FETCH_MAX_CHARS=20000
ENV HTTP_TIMEOUT=20
# The uploads volume, shared with the gateway. This is also WORKDIR below, so a
# path mcp-files reports is already relative to what the built-in file tools
# resolve against — that is the whole trick, and changing one without the other
# breaks it.
ENV UPLOADS_WORKSPACE=/workspace
ENV INBOX_SUBDIR=inbox
ENV EXTRACT_SUBDIR=extracted
# Caps on what one tool call may return, so an extracted node_modules cannot
# flood the context window.
ENV UPLOADS_MAX_DEPTH=12
ENV UPLOADS_MAX_ENTRIES=400

EXPOSE 8080

VOLUME ["/models"]

# The built-in file tools (--tools) resolve a relative path against the server's
# working directory, so the server runs from a directory meant to be mounted over
# rather than from /app next to the binaries.
RUN mkdir -p /workspace
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
