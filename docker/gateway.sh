#!/bin/sh
# The front door, inside the model image. nginx + njs render and start here.
#
# This is not a separate container. nginx publishes :8080, llama-server binds
# 127.0.0.1:8081 behind it, and both live in one image — the gateway has no
# model to load and no CUDA in it, so a second image bought nothing but a second
# thing to version, pull and keep in step.
#
# Two modes:
#   gateway.sh start        render, check, start nginx in the background and
#                           return. entrypoint.sh uses this, then goes on to
#                           fetch weights and exec llama-server as PID 1.
#   gateway.sh foreground   render, check, exec nginx with `daemon off`. Nothing
#                           else runs: this is how CI drives the HTTP handlers
#                           without a 112 GiB download in front of them.
#
# The template ships with __PLACEHOLDER__ values and the resolved config is
# written to an ephemeral directory at start, so what is mounted and reviewed is
# the template while the rendered file never persists. Same pattern as mcp.json.
set -eu

MODE="${1:-start}"

log() { printf '[gateway] %s\n' "$*"; }

LISTEN_PORT="${LISTEN_PORT:-8080}"
# Where llama-server is. Same container, so loopback — this is PORT from the
# environment, not a service name on a compose network any more.
UPSTREAM="${UPSTREAM:-127.0.0.1:${PORT:-8081}}"
# The uploads volume, which is also llama-server's working directory. A path
# reported by /manage/files is therefore already usable by the model's built-in
# file tools unchanged; that is the whole trick.
DATA_DIR="${DATA_DIR:-${UPLOADS_WORKSPACE:-/workspace}}"
INBOX_SUBDIR="${INBOX_SUBDIR:-inbox}"
EXTRACT_SUBDIR="${EXTRACT_SUBDIR:-extracted}"
UPLOAD_TOKEN="${UPLOAD_TOKEN:-}"
MAX_UPLOAD_SIZE="${MAX_UPLOAD_SIZE:-8g}"
ERROR_LEVEL="${ERROR_LEVEL:-warn}"
# nginx shares stdout with the entrypoint and with llama-server. A browser
# sitting on the WebUI polls /props every second, which is enough to bury a
# download or a model load in access lines, so this is off unless asked for.
# "on" means /dev/stdout; any other value is passed to nginx as written, so a
# path works too.
WORKER_PROCESSES="${WORKER_PROCESSES:-4}"
ACCESS_LOG="${ACCESS_LOG:-off}"
case "$ACCESS_LOG" in
    on|1|true|yes) ACCESS_LOG=/dev/stdout ;;
esac
# Debian/Ubuntu's libnginx-mod-http-js lands here. The path is in the config
# rather than behind an include so a missing module fails at `nginx -t` instead
# of silently skipping every /manage/ route.
NJS_MODULE="${NJS_MODULE:-/usr/lib/nginx/modules/ngx_http_js_module.so}"

INBOX_DIR="${DATA_DIR}/${INBOX_SUBDIR}"

# The rename in upload.js only works if the body temp dir and the destination
# are on one filesystem, so the temp dir lives inside the volume. The leading
# dot keeps it out of /manage/files listings and out of the model's way.
TMP_DIR="${DATA_DIR}/.tmp"

mkdir -p "$INBOX_DIR" "$TMP_DIR" "${DATA_DIR}/${EXTRACT_SUBDIR}"
mkdir -p /var/lib/nginx/tmp /run

# A body buffered for a request that never completed is dead weight — nginx does
# not clean these up when client_body_in_file_only is on, and a killed container
# mid-upload leaves one behind.
stale="$(find "$TMP_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
if [ "$stale" != "0" ]; then
    log "clearing ${stale} stale body file(s) from ${TMP_DIR}"
    find "$TMP_DIR" -type f -delete 2>/dev/null || true
fi

if [ ! -f "$NJS_MODULE" ]; then
    log "FAILED njs module missing: ${NJS_MODULE}"
    log "  the image needs libnginx-mod-http-js at the same version as nginx"
    exit 1
fi

if [ -n "$UPLOAD_TOKEN" ]; then
    log "upload token REQUIRED (X-Upload-Token header or ?token=)"
else
    log "upload token disabled — anyone who can reach :${LISTEN_PORT} can write to ${INBOX_DIR}"
fi

CONF_DIR="$(mktemp -d)"
CONF="${CONF_DIR}/nginx.conf"

# Values are paths, a port, a size and a token. A separator that cannot appear
# in any of them keeps this from needing to escape them.
sed -e "s#__NJS_MODULE__#${NJS_MODULE}#g" \
    -e "s#__UPSTREAM__#${UPSTREAM}#g" \
    -e "s#__LISTEN_PORT__#${LISTEN_PORT}#g" \
    -e "s#__DATA_DIR__#${DATA_DIR}#g" \
    -e "s#__INBOX_DIR__#${INBOX_DIR}#g" \
    -e "s#__UPLOAD_TOKEN__#${UPLOAD_TOKEN}#g" \
    -e "s#__MAX_UPLOAD_SIZE__#${MAX_UPLOAD_SIZE}#g" \
    -e "s#__ERROR_LEVEL__#${ERROR_LEVEL}#g" \
    -e "s#__ACCESS_LOG__#${ACCESS_LOG}#g" \
    -e "s#__WORKER_PROCESSES__#${WORKER_PROCESSES}#g" \
    /etc/glm53f/nginx.conf.template > "$CONF"

if grep -q '__[A-Z_]*__' "$CONF"; then
    log "FAILED template still has unrendered placeholders:"
    grep -o '__[A-Z_]*__' "$CONF" | sort -u | sed 's/^/  /'
    exit 1
fi

# Fails loudly here rather than serving a half-configured front door — this is
# also what catches an njs module that loaded but does not match this build.
nginx -t -c "$CONF"

log "listening on :${LISTEN_PORT}, upstream ${UPSTREAM}, uploads -> ${INBOX_DIR}"
log "  /manage/            file manager page"
log "  /manage/files       listing JSON      /manage/upload    PUT raw body"
log "  /manage/download/   read a file back  /manage/health    gateway liveness"
log "  /                   llama.cpp WebUI and API, proxied"

# An nginx left over from a previous entrypoint run still holds LISTEN_PORT, and
# a fresh one cannot bind it: restarting the entrypoint by hand is the normal
# way to pick up new settings on a rented box, and without this it dies on
# "bind() to 0.0.0.0:PORT failed (98: Address already in use)" — killed by its
# own leftover. Only nginx is stopped here; anything else holding the port is
# not ours to kill, and nginx will report that itself.
stop_old_nginx() {
    local waited=0

    pgrep -x nginx >/dev/null 2>&1 || return 0

    log "nginx already running — stopping it before taking :${LISTEN_PORT}"
    nginx -s quit 2>/dev/null || pkill -x nginx 2>/dev/null || true

    # quit is graceful and waits for in-flight requests, so give it a moment
    # before escalating.
    while [ "$waited" -lt 10 ]; do
        pgrep -x nginx >/dev/null 2>&1 || return 0
        sleep 1
        waited=$(( waited + 1 ))
    done

    log "nginx did not stop in ${waited}s, killing it"
    pkill -x -9 nginx 2>/dev/null || true
    sleep 1
    return 0
}

case "$MODE" in
    foreground)
        stop_old_nginx
        exec nginx -c "$CONF" -g 'daemon off;'
        ;;
    start)
        # Daemonised on purpose: llama-server is the container's main process, so
        # the container's lifetime tracks the model and not the proxy. nginx
        # comes up FIRST, which is what keeps /manage/ answering through a first
        # run's 87-272 GiB download — / is 503 until the upstream is up.
        stop_old_nginx
        nginx -c "$CONF"
        log "nginx started (pid $(cat /run/nginx.pid 2>/dev/null || echo '?'))"
        ;;
    *)
        log "FAILED unknown mode '${MODE}' (start|foreground)"
        exit 1
        ;;
esac
