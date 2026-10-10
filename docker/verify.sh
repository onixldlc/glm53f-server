#!/usr/bin/env bash
# Sanity checks for a running container: GPU visible, build new enough for the
# agent tools, server answering. Run with: podman exec -it glm53f verify.sh
set -uo pipefail

ok()   { printf '  OK   %s\n' "$*"; }
fail() { printf '  FAIL %s\n' "$*"; rc=1; }
rc=0

printf 'GPU\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && ok "nvidia-smi"
else
    fail "nvidia-smi missing — container has no GPU passthrough"
fi

printf 'binary\n'
/app/bin/llama-server --version 2>&1 | head -3 && ok "llama-server runs"

printf 'agent tools support\n'
if /app/bin/llama-server --help 2>&1 | grep -q -- '--tools'; then
    ok "--tools present (build new enough for EXTRA_ARGS)"
else
    fail "--tools absent — llama.cpp release is too old for this EXTRA_ARGS"
fi

printf 'autoscale\n'
if [ "${AUTO_SCALE:-1}" = "1" ] && command -v autoscale.sh >/dev/null 2>&1; then
    autoscale.sh 2>&1 >/dev/null | sed 's/^/  /'
    ok "table decision printed above; compare with what the server actually loaded"
else
    ok "AUTO_SCALE off, MODEL_FILE/CTX_SIZE taken from the environment"
fi

printf 'weights\n'
# MODEL_FILE names shard 1 of N. Checking only that one would pass on a half
# finished download, because llama.cpp does not open shards 2..N until it has
# read the index in shard 1.
shards_of() {
    local first="$1" n i
    case "$first" in
        *-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf)
            n="${first##*-of-}"; n="${n%.gguf}"; n=$((10#$n))
            i=1
            while [ "$i" -le "$n" ]; do
                printf '%s%05d-of-%05d.gguf\n' "${first%%[0-9][0-9][0-9][0-9][0-9]-of-*}" "$i" "$n"
                i=$((i + 1))
            done
            ;;
        *) printf '%s\n' "$first" ;;
    esac
}

while read -r f; do
    [ -n "$f" ] || continue
    [ -s "${MODEL_DIR}/${f}" ] && ok "${f} $(du -h "${MODEL_DIR}/${f}" | cut -f1)" || fail "${MODEL_DIR}/${f} missing"
done <<EOF
$(shards_of "${MODEL_FILE}")
$([ "${ENABLE_VISION:-0}" = "1" ] && printf '%s' "${MMPROJ_FILE:-}")
EOF

if [ "${ENABLE_VISION:-0}" != "1" ]; then
    ok "ENABLE_VISION=0, mmproj not required"
fi

printf 'volume is weights-only\n'
# Quants live in a per-quant subdirectory (UD-IQ3_XXS/...), so a directory is
# expected here now; anything else that is not a .gguf is not.
stray=$(find "${MODEL_DIR}" -mindepth 1 -not -type d -not -name '*.gguf' 2>/dev/null || true)
if [ -z "$stray" ]; then
    ok "no non-gguf files in ${MODEL_DIR} ($(find "${MODEL_DIR}" -name '*.gguf' 2>/dev/null | wc -l | tr -d ' ') gguf, $(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1))"
else
    fail "stray entries in ${MODEL_DIR}: $(echo "$stray" | tr '\n' ' ')"
fi

printf 'uploads volume\n'
WS="${UPLOADS_WORKSPACE:-/workspace}"
if [ -d "$WS" ]; then
    ok "${WS} present ($(find "$WS" -type f 2>/dev/null | wc -l | tr -d ' ') file(s))"
else
    fail "${WS} missing — the shared volume is not mounted"
fi

# The whole design rests on these being one directory: mcp-files reports paths
# relative to UPLOADS_WORKSPACE, and the built-in --tools resolve relative paths
# against the working directory. If they diverge, every path handed to the model
# is wrong, and nothing else here would notice.
if [ "$(cd "$WS" 2>/dev/null && pwd -P)" = "$(pwd -P)" ]; then
    ok "working directory is the uploads volume"
else
    fail "cwd $(pwd -P) != uploads volume ${WS} — paths from mcp-files will not resolve"
fi

printf 'mcp servers\n'
if [ "${MCP_ENABLE:-0}" != "1" ]; then
    ok "MCP_ENABLE=0, no tool servers passed to llama-server"
else
    # Drive the real protocol. A tools/list that comes back populated proves jq,
    # find and the script itself all work inside this image, which a file
    # existence check would not.
    probe() {
        printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
            | "$1" 2>/dev/null \
            | jq -r '.result.tools | length' 2>/dev/null
    }

    if [ "${MCP_FILES_ENABLE:-1}" = "1" ]; then
        n="$(probe /usr/local/bin/mcp-files)"
        if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then
            ok "mcp-files answers tools/list with ${n} tools"
        else
            fail "mcp-files did not answer tools/list"
        fi
    else
        ok "MCP_FILES_ENABLE=0"
    fi

    if [ "${MCP_WEB_ENABLE:-1}" = "1" ]; then
        n="$(probe /usr/local/bin/mcp-web)"
        if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then
            ok "mcp-web answers tools/list with ${n} tools"
        else
            fail "mcp-web did not answer tools/list"
        fi
    else
        ok "MCP_WEB_ENABLE=0"
    fi
fi

printf 'front door\n'
# nginx is in THIS container, on LISTEN_PORT, with llama-server behind it on
# PORT. No compose network involved.
GW_PORT="${LISTEN_PORT:-8080}"
if [ "${GATEWAY_ENABLE:-1}" != "1" ]; then
    ok "GATEWAY_ENABLE=0, nothing serves /manage/ by design"
elif ! pgrep -x nginx >/dev/null 2>&1; then
    fail "GATEWAY_ENABLE=1 but no nginx process"
elif curl -fsS --max-time 5 "http://127.0.0.1:${GW_PORT}/manage/health" >/dev/null 2>&1; then
    ok "nginx answers on 127.0.0.1:${GW_PORT}"
    curl -fsS --max-time 5 "http://127.0.0.1:${GW_PORT}/manage/files?path=&depth=1" | head -c 300; echo
else
    fail "nginx is running but /manage/health does not answer on ${GW_PORT}"
fi

printf 'http\n'
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ok "/health"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
else
    fail "/health unreachable on ${PORT}"
fi

exit $rc
