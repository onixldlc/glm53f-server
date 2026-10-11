#!/usr/bin/env bash
# Fetch the GGUF into the models volume on first start, then serve an
# OpenAI-compatible endpoint with llama-server from the upstream release.
#
# Downloads are plain resumable HTTPS against the HF CDN — no hf CLI, no Xet.
# Drop a .gguf into the volume yourself and this skips the download entirely.
set -euo pipefail

log() { printf '[glm53f] %s\n' "$*"; }

MODEL_DIR="${MODEL_DIR:-/models}"
DISK_RESERVE_MIB="${DISK_RESERVE_MIB:-8192}"
mkdir -p "$MODEL_DIR"

# --- front door -------------------------------------------------------------
# nginx first, llama-server second, both in this container. nginx publishes
# LISTEN_PORT (8080) and proxies to llama-server on 127.0.0.1:PORT (8081), so
# /manage/ answers for the whole of a first run's 87-272 GiB download while /
# returns 502 until the model is actually up. It is daemonised, so the container
# still lives and dies with llama-server below.
if [ "${GATEWAY_ENABLE:-1}" = "1" ]; then
    /usr/local/bin/gateway.sh start
else
    log "gateway disabled: llama-server is the only listener, on ${HOST}:${PORT}"
    log "  nothing serves /manage/ — set HOST=0.0.0.0 PORT=8080 to publish it directly"
fi

# --- VRAM autoscaling -------------------------------------------------------
# AUTO_SCALE=1 asks autoscale.sh which quant, context and KV type suit the card
# in front of us. The decision happens here, before anything is downloaded,
# because it is what decides WHICH file to download. AUTO_SCALE=0 uses the
# MODEL_FILE and CTX_SIZE already in the environment.
AUTO_SCALE="${AUTO_SCALE:-1}"
FIT_MARGIN_MIB="${FIT_MARGIN_MIB:-4096}"
AS_KV=""

if [ "$AUTO_SCALE" = "1" ] && [ "${N_GPU_LAYERS:-0}" != "0" ]; then
    set +e
    as_out="$(FIT_MARGIN_MIB="$FIT_MARGIN_MIB" /usr/local/bin/autoscale.sh)"
    as_rc=$?
    set -e
    # 2 is "no quant fits on this disk", which is fatal: carrying on with the
    # image default would download until the filesystem is full, and a full
    # filesystem takes the whole host with it.
    if [ "$as_rc" = "2" ]; then
        log "FAILED autoscale refused to start: not enough disk at ${MODEL_DIR}"
        log "  Nothing has been downloaded. Give the container a bigger disk or"
        log "  mount a volume there, then start again."
        exit 2
    fi
    if [ "$as_rc" = "0" ]; then
        # autoscale.sh emits plain KEY=VALUE lines and nothing else.
        eval "$as_out"
        AS_KV="$CACHE_TYPE"
        log "auto-scale: ${VRAM_MIB} MiB -> ${MODEL_FILE}, ctx ${CTX_SIZE}, KV ${AS_KV}"
    else
        log "auto-scale: no card detected, keeping MODEL_FILE=${MODEL_FILE} ctx=${CTX_SIZE}"
    fi
fi

# A finished GGUF starts with the magic bytes "GGUF".
is_gguf() {
    [ -s "$1" ] && [ "$(head -c 4 "$1" 2>/dev/null)" = "GGUF" ]
}

fetch() {
    local file="$1"
    local dest="${MODEL_DIR}/${file}"
    local part="${dest}.part"
    local url="${MODEL_BASE_URL}/${MODEL_REPO}/resolve/main/${file}"
    local attempt=1

    # Quants live in a per-quant subdirectory of the repo, and the shards have to
    # land next to each other because llama.cpp finds shards 2..N by looking in
    # the directory it loaded shard 1 from.
    mkdir -p "$(dirname "$dest")"

    # A file already in the volume is never downloaded again. The magic bytes are
    # only a warning here: 18 GB is far too much to re-fetch over a check that can
    # fail for reasons that have nothing to do with the file being bad.
    if [ -e "$dest" ]; then
        if [ ! -r "$dest" ]; then
            log "FAILED ${dest} exists but uid $(id -u) cannot read it"
            log "  chown the file to this uid, or mount the volume with :Z so SELinux relabels it"
            return 1
        fi
        if [ ! -s "$dest" ]; then
            log "FAILED ${dest} exists but is empty — delete it and start again"
            return 1
        fi
        if ! is_gguf "$dest"; then
            log "WARNING ${file} does not start with GGUF — keeping it anyway; llama-server"
            log "  will refuse it if it really is truncated. Delete it to force a fresh download."
        fi
        log "have ${file} ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")

    # Never start a transfer that cannot finish. A download that fills the
    # filesystem does not just fail: it takes down everything else on the host
    # that needs to write, sshd included, and on a rented box that means losing
    # the instance rather than losing the file. The size comes from a HEAD, so
    # a server that will not answer one simply skips the check.
    local free_mib size_mib
    free_mib="$(df -Pm "$(dirname "$dest")" 2>/dev/null | awk 'NR==2 { print $4 + 0 }')" || free_mib=""
    if [ -n "$free_mib" ] && [ "$free_mib" -gt 0 ] 2>/dev/null; then
        # A HEAD that fails is not an error here: set -o pipefail would make the
        # assignment itself fatal, so the size is simply treated as unknown.
        size_mib="$(curl -fsSLI "${auth[@]}" "$url" 2>/dev/null \
            | tr -d '\r' \
            | awk 'tolower($1) == "content-length:" { n = int($2 / 1048576) } END { if (n > 0) print n }')" || size_mib=""
        if [ -n "$size_mib" ] && [ $(( size_mib + DISK_RESERVE_MIB )) -gt "$free_mib" ]; then
            log "FAILED ${file} needs ${size_mib} MiB and only ${free_mib} MiB is free at ${MODEL_DIR}"
            log "  (keeping ${DISK_RESERVE_MIB} MiB in reserve). Refusing to start the transfer:"
            log "  filling this filesystem would take the host down with it, sshd included."
            log "  Give the container a bigger disk, or mount a volume at ${MODEL_DIR}."
            return 1
        fi
    fi

    while [ "$attempt" -le "${DOWNLOAD_ATTEMPTS}" ]; do
        log "downloading ${file} (attempt ${attempt}/${DOWNLOAD_ATTEMPTS})"
        # -C - resumes from whatever is already in .part, so a stalled transfer
        # costs only the bytes it had not reached yet.
        if curl -fL --progress-bar \
                -C - \
                --retry 5 --retry-delay 5 --retry-all-errors \
                --speed-limit 1024 --speed-time 60 \
                "${auth[@]}" \
                -o "$part" "$url"; then
            mv -f "$part" "$dest"
            log "done ${file} ($(du -h "$dest" | cut -f1))"
            return 0
        fi
        log "transfer interrupted, resuming in 5s"
        sleep 5
        attempt=$((attempt + 1))
    done

    log "FAILED after ${DOWNLOAD_ATTEMPTS} attempts: ${url}"
    log "download it yourself and copy it into the volume, then start again:"
    log "  podman run --rm -v glm53f-models:/models -v \"\$PWD\":/host:z \\"
    log "    docker.io/library/busybox cp \"/host/${file}\" /models/"
    return 1
}

# A GLM-5.3-Flash quant is 87-272 GiB, which HF stores as 3-7 shards named
# -00001-of-0000N.gguf. MODEL_FILE always names shard 1; the count comes out of
# the name itself rather than a second variable, so a hand-pinned MODEL_FILE at
# AUTO_SCALE=0 needs nothing extra to be fetched completely.
fetch_all() {
    local first="$1"
    local n i name

    case "$first" in
        *-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf)
            n="${first##*-of-}"
            n="${n%.gguf}"
            # 00004 -> 4, without octal-interpreting the leading zeros.
            n=$((10#$n))
            ;;
        *)
            fetch "$first"
            return $?
            ;;
    esac

    log "model is sharded: ${n} files"
    i=1
    while [ "$i" -le "$n" ]; do
        name="$(printf '%s%05d-of-%05d.gguf' "${first%%[0-9][0-9][0-9][0-9][0-9]-of-*}" "$i" "$n")"
        fetch "$name" || return 1
        i=$((i + 1))
    done
}

fetch_all "$MODEL_FILE"

args=(
    --model "${MODEL_DIR}/${MODEL_FILE}"
    --n-gpu-layers "$N_GPU_LAYERS"
    --flash-attn "$FLASH_ATTN"
    --host "$HOST"
    --port "$PORT"
    --jinja
    --alias glm-5.3-flash
)

# CTX_SIZE=auto leaves --ctx-size off entirely, which hands the decision to
# llama.cpp's own fitter: it is on by default (common.h: fit_params) and sizes
# the KV pool to free device memory, keeping --fit-target MiB (1024 by default)
# in reserve. It measures compute buffers instead of estimating them, so it
# beats any table here — but it only adjusts arguments we do not pass.
ctx_args=()
if [ "${CTX_SIZE}" != "auto" ]; then
    ctx_args=(--ctx-size "$CTX_SIZE")
else
    log "ctx: delegated to llama.cpp --fit (margin $(( FIT_MARGIN_MIB )) MiB per device)"
    ctx_args=(--fit-target "$FIT_MARGIN_MIB")
fi

# Only set by auto-scale, and never over an explicit choice the user made.
kv_args=()
KV_IS_OURS=0
if [ -n "$AS_KV" ] \
   && [ -z "${LLAMA_ARG_CACHE_TYPE_K:-}" ] \
   && [[ "${EXTRA_ARGS}" != *--cache-type-k* ]]; then
    kv_args=(--cache-type-k "$AS_KV" --cache-type-v "$AS_KV")
    KV_IS_OURS=1
fi

# Left empty, llama.cpp picks its own thread count.
if [ -n "${THREADS:-}" ]; then
    args+=(--threads "$THREADS")
fi
if [ -n "${THREADS_BATCH:-}" ]; then
    args+=(--threads-batch "$THREADS_BATCH")
fi

if [ "${ENABLE_VISION}" = "1" ] && [ -n "${MMPROJ_FILE}" ]; then
    fetch "$MMPROJ_FILE"
    args+=(--mmproj "${MODEL_DIR}/${MMPROJ_FILE}")
fi

# MCP tool servers. The config carries no secret, but it is still rendered from
# the template into an ephemeral dir at start so the values come from the
# environment rather than from a file baked into the image.
#
# Both servers are stdio children of llama-server, not compose services:
# server-mcp.cpp:147 parses only command/cwd/args/env/timeout_ms, so this
# release has no HTTP transport to point at a separate container.
if [ "${MCP_ENABLE:-0}" = "1" ]; then
    mcp_dir="$(mktemp -d)"
    mcp_raw="${mcp_dir}/mcp.raw.json"
    mcp_config="${mcp_dir}/mcp.json"

    sed -e "s#__SEARXNG_URL__#${SEARXNG_URL:-}#" \
        -e "s#__SEARCH_RESULTS__#${SEARCH_RESULTS:-8}#" \
        -e "s#__FETCH_MAX_CHARS__#${FETCH_MAX_CHARS:-20000}#" \
        -e "s#__HTTP_TIMEOUT__#${HTTP_TIMEOUT:-20}#" \
        -e "s#__UPLOADS_WORKSPACE__#${UPLOADS_WORKSPACE:-/workspace}#" \
        -e "s#__INBOX_SUBDIR__#${INBOX_SUBDIR:-inbox}#" \
        -e "s#__EXTRACT_SUBDIR__#${EXTRACT_SUBDIR:-extracted}#" \
        -e "s#__UPLOADS_MAX_DEPTH__#${UPLOADS_MAX_DEPTH:-12}#" \
        -e "s#__UPLOADS_MAX_ENTRIES__#${UPLOADS_MAX_ENTRIES:-400}#" \
        /etc/glm53f/mcp.json.template > "$mcp_raw"

    if grep -q '__[A-Z_]*__' "$mcp_raw"; then
        log "FAILED mcp.json still has unrendered placeholders:"
        grep -o '__[A-Z_]*__' "$mcp_raw" | sort -u | sed 's/^/  /'
        exit 1
    fi

    # One template holds every server and the ones switched off are deleted here,
    # rather than kept as a second template that would drift from the first.
    mcp_filter='.'
    [ "${MCP_WEB_ENABLE:-1}" = "1" ]   || mcp_filter="${mcp_filter} | del(.mcpServers.web)"
    [ "${MCP_FILES_ENABLE:-1}" = "1" ] || mcp_filter="${mcp_filter} | del(.mcpServers.files)"
    jq "$mcp_filter" "$mcp_raw" > "$mcp_config"

    mcp_on="$(jq -r '.mcpServers | keys | join(", ")' "$mcp_config")"
    if [ -z "$mcp_on" ]; then
        log "mcp: MCP_ENABLE=1 but every server is off, not passing --mcp-servers-config"
    else
        args+=(--mcp-servers-config "$mcp_config")
        log "mcp servers: ${mcp_on}"
        if [ "${MCP_WEB_ENABLE:-1}" = "1" ]; then
            log "  web:   search via ${SEARXNG_URL:-duckduckgo}"
        fi
        if [ "${MCP_FILES_ENABLE:-1}" = "1" ]; then
            log "  files: uploads under ${UPLOADS_WORKSPACE:-/workspace}"
        fi
    fi
fi

# nginx writes here and the built-in file tools read from here. gateway.sh makes
# these too; repeated for GATEWAY_ENABLE=0, where an empty volume should still
# list cleanly instead of erroring.
mkdir -p "${UPLOADS_WORKSPACE:-/workspace}/${INBOX_SUBDIR:-inbox}" \
         "${UPLOADS_WORKSPACE:-/workspace}/${EXTRACT_SUBDIR:-extracted}" 2>/dev/null || true

log "volume holds: $(ls -1A "$MODEL_DIR" | tr '\n' ' ')"

# EXTRA_ARGS lives in its own array so it can be appended LAST. llama.cpp takes
# the final occurrence of a repeated flag, so anything the user put in here has
# to come after the --ctx-size / --cache-type arguments this script generates,
# or their override would silently lose to ours.
extra_args=()
if [ -n "${EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_ARGS})
fi

# --- start, and step down rather than strand the host -----------------------
# A table cannot measure compute buffers, so a load can still run out of device
# memory after the weights are already on the cards. Everything below re-runs on
# the SAME weights — only the KV type and the context shrink, so a retry costs
# seconds, never another download.
#
# Each rung is tried in order; the first that stays up past GRACE_SECONDS is the
# one we keep. Anything passed to `podman run <image> ...` is appended verbatim.
GRACE_SECONDS="${GRACE_SECONDS:-180}"

ladder=()
if [ "$KV_IS_OURS" = "1" ] && [ "${CTX_SIZE}" != "auto" ]; then
    # "<kv>:<ctx>". Drop the KV type first, then halve the context twice: q8_0 at
    # full context beats f16 at a quarter of it.
    [ "$AS_KV" != "q8_0" ] && ladder+=("q8_0:${CTX_SIZE}")
    ladder+=("q8_0:$(( CTX_SIZE / 2 ))")
    ladder+=("q8_0:$(( CTX_SIZE / 4 ))")
fi

# llama-server is no longer exec'd, so it is no longer PID 1 and no longer gets
# `podman stop`'s SIGTERM by itself. Without this it would sit through the stop
# timeout and be SIGKILLed, losing a clean shutdown — and the ladder below would
# read that as a failed load and "helpfully" restart it smaller.
STOPPING=0
on_signal() {
    STOPPING=1
    log "signal received, stopping llama-server"
    [ -n "${LLAMA_PID:-}" ] && kill -TERM "$LLAMA_PID" 2>/dev/null || true
}
trap on_signal TERM INT

run_llama() {
    local started rc
    log "llama-server ${*}"
    started="$(date +%s)"
    /app/bin/llama-server "$@" &
    LLAMA_PID=$!
    set +e
    # Returns 128+n the moment a trap fires, with the child still shutting down;
    # the second wait reaps it for real.
    wait "$LLAMA_PID"
    rc=$?
    [ "$STOPPING" = "1" ] && wait "$LLAMA_PID" 2>/dev/null
    set -e
    LAST_RAN=$(( $(date +%s) - started ))
    return "$rc"
}

attempt_args=("${args[@]}" "${ctx_args[@]}" "${kv_args[@]}" "${extra_args[@]}" "$@")

if run_llama "${attempt_args[@]}"; then
    exit 0
fi
[ "$STOPPING" = "1" ] && { log "stopped on signal"; exit 0; }

for rung in "${ladder[@]}"; do
    # Served for a while and then stopped: that is not a sizing problem, and
    # retrying smaller would hide a real crash.
    if [ "${LAST_RAN:-0}" -ge "$GRACE_SECONDS" ]; then
        log "llama-server ran ${LAST_RAN}s before exiting — not a fit problem, not retrying"
        break
    fi

    kv="${rung%%:*}"
    ctx="${rung#*:}"
    log ""
    log "llama-server died after ${LAST_RAN}s. Stepping down: KV ${kv}, ctx ${ctx}."
    log "  Same weights, nothing re-downloaded. Set CTX_TARGET or FIT_MARGIN_MIB"
    log "  to pick this directly next boot and skip the failed attempt."

    attempt_args=("${args[@]}" --ctx-size "$ctx" --cache-type-k "$kv" --cache-type-v "$kv" \
                  "${extra_args[@]}" "$@")
    if run_llama "${attempt_args[@]}"; then
        exit 0
    fi
    [ "$STOPPING" = "1" ] && { log "stopped on signal"; exit 0; }
done

log ""
log "FAILED llama-server will not stay up at any size this script can try."
log "  Last run lasted ${LAST_RAN:-0}s. Look above this line for the real error:"
log "    cudaMalloc failed      -> out of device memory, lower CTX_TARGET"
log "    killed with no message -> host RAM, the kernel OOM killer took it"
log "    tensor/shard errors    -> a truncated download, delete ${MODEL_DIR} and retry"

# Stop nginx too. Left running it would keep answering /manage/ and returning
# 502 on /, so the container looks alive while nothing can ever serve a token.
if [ "${GATEWAY_ENABLE:-1}" = "1" ]; then
    log "stopping nginx so this container exits instead of serving 502 forever"
    nginx -s stop 2>/dev/null || pkill -x nginx 2>/dev/null || true
fi

exit 1
