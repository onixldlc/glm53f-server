#!/usr/bin/env bash
# Pick MODEL_FILE, CTX_SIZE and the KV cache type from the card's VRAM.
#
# Prints KEY=VALUE lines on stdout for the entrypoint to eval; everything else
# goes to stderr. Run it by hand on a box to see what it would choose:
#
#   autoscale.sh                  # decide from nvidia-smi
#   VRAM_MIB=163840 autoscale.sh  # decide for a card you do not have
#   autoscale.sh --measure        # ask llama-fit-params instead (needs the gguf)
#
# Nothing here downloads or loads anything: the decision is made before the
# weights exist, which is the whole point — it chooses which file to fetch.
set -euo pipefail

say() { printf '[autoscale] %s\n' "$*" >&2; }

ENABLE_VISION="${ENABLE_VISION:-0}"
# PER DEVICE, not a total — same meaning as llama.cpp's --fit-target, which the
# entrypoint hands the same number. 320B of MoE weights needs far more scratch
# than a 27B dense model did, which is why this is 4x the llama.cpp stock 1024.
#
# This is per device because llama.cpp splits LAYERS across GPUs: on a 2-card
# host a total-based reserve leaves each card half of it, and whichever card
# holds the mmproj less still. Getting that wrong is a load that dies after the
# weights are already on the cards.
FIT_MARGIN_MIB="${FIT_MARGIN_MIB:-4096}"
# How many devices that margin is multiplied by. Detected from nvidia-smi;
# override when deciding for a host you are not on (with VRAM_MIB).
N_DEVICES="${N_DEVICES:-}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_PREFIX="${MODEL_PREFIX:-GLM-5.3-Flash}"
MMPROJ_MIB=1110
# GLM-5.3-Flash is a 1M-context model, but nothing here reaches for that: the
# context is pinned to CTX_TARGET and the quant is chosen around it.
MODEL_MAX_CTX=1048576

# --- KV cost ----------------------------------------------------------------
# Same per-token numbers the qwen image used: 64 MiB per 1024 tokens at f16,
# 34 MiB at q8_0. Deliberately NOT the smaller figure the glm5-next geometry
# would suggest — these are the measured ones, and overestimating KV costs a
# quant tier at worst, while underestimating it costs a failed load.
KV_MIB_PER_1K_F16=64
KV_MIB_PER_1K_Q8=34

# 131072 is a FLOOR, not a target. The quant is what flexes to make it fit:
# pick_for_vram walks the quants largest-first and takes the first one that
# holds this much context, trying f16 and then q8_0 on each. Raise it and the
# table shifts down accordingly.
CTX_TARGET="${CTX_TARGET:-131072}"

# Total across devices: llama.cpp spreads layers over every visible GPU.
detect_vram_mib() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ t += $1 + 0 } END { if (t > 0) print int(t); else exit 1 }'
}

detect_device_count() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF { n++ } END { if (n > 0) print n; else exit 1 }'
}

# Every quant of this model, largest first, with the real totals from the
# Unsloth repo summed over every shard of that quant (the 00001 shard is a
# ~9 MiB index, the rest carry the tensors). Sizes in GiB, for reading against
# the table in the README:
#   Q6_K_XL 271.79  Q5_K_XL 223.80  Q4_K_XL 185.99  IQ4_XS 146.05
#   Q3_K_XL 137.40  IQ3_XXS 112.10  Q2_K_XL 101.25  IQ2_XXS  94.85
#   IQ1_M    90.88  IQ1_S    86.69
#
# Quality, where it has been measured (perplexity vs BF16 / same-top-p):
#   IQ3_XXS 1.341x / 81.8%      Q2_K_XL  -    / 78%      IQ2_XXS 1.883x / 73.2%
# The IQ2 -> IQ3 step halves KL divergence, so IQ3_XXS is the lowest quant
# worth trusting for agentic work.
QUANTS="\
UD-Q6_K_XL:7:278314
 UD-Q5_K_XL:6:229174
 UD-Q4_K_XL:6:190456
 UD-IQ4_XS:5:149557
 UD-Q3_K_XL:4:140701
 UD-IQ3_XXS:4:114791
 UD-Q2_K_XL:4:103684
 UD-IQ2_XXS:4:97127
 UD-IQ1_M:3:93059
 UD-IQ1_S:3:88775"

# What one row costs on top of its weights, for a given KV type.
#
# The margin is multiplied by the device count: a 2-card host must keep
# FIT_MARGIN_MIB free on EACH card, so 2x that much of the total is unusable.
overhead_mib() {
    local per_1k="$1"
    local o=$(( FIT_MARGIN_MIB * N_DEVICES + (CTX_TARGET / 1024) * per_1k ))
    [ "$ENABLE_VISION" = "1" ] && o=$(( o + MMPROJ_MIB ))
    printf '%s' "$o"
}

# Sets MODEL_FILE / CACHE_TYPE / CTX_SIZE from $1 = total VRAM in MiB.
#
# Walks the quants largest-first and takes the first that holds CTX_TARGET with
# q8_0 KV, then upgrades that row to f16 if f16 ALSO fits. Deliberately this way
# round: the KV type costs far less quality than a whole quant step, so a bigger
# quant at q8_0 beats a smaller one at f16. Trying f16 first did the opposite —
# it spent 3840 MiB of headroom on the KV type and then had to drop a quant.
pick_for_vram() {
    local vram_mib="$1"
    local o_f16 o_q8 row quant shards weights_mib
    o_f16="$(overhead_mib "$KV_MIB_PER_1K_F16")"
    o_q8="$(overhead_mib "$KV_MIB_PER_1K_Q8")"

    quant=""
    for row in $QUANTS; do
        local q="${row%%:*}"
        local rest="${row#*:}"
        local sh="${rest%%:*}"
        local w="${rest#*:}"
        if [ $(( w + o_q8 )) -le "$vram_mib" ]; then
            quant="$q"; shards="$sh"; weights_mib="$w"
            if [ $(( w + o_f16 )) -le "$vram_mib" ]; then
                CACHE_TYPE=f16
                MIN_MIB=$(( w + o_f16 ))
            else
                CACHE_TYPE=q8_0
                MIN_MIB=$(( w + o_q8 ))
                say "${q} needs q8_0 KV to hold ${CTX_TARGET} in ${vram_mib} MiB"
            fi
            break
        fi
    done

    # Nothing holds CTX_TARGET, not even the smallest quant with q8_0 KV.
    if [ -z "$quant" ]; then
        quant=UD-IQ1_S; shards=3; weights_mib=88775; CACHE_TYPE=q8_0
        MIN_MIB=$(( weights_mib + o_q8 ))
        say "WARNING ${vram_mib} MiB cannot hold ${CTX_TARGET} context at any quant."
        say "  Smallest is ${quant} + q8_0 KV = ${MIN_MIB} MiB. Layers will spill to"
        say "  host RAM; drop N_GPU_LAYERS, lower CTX_TARGET, or rent more VRAM."
    fi

    MODEL_FILE="$(printf '%s/%s-%s-%05d-of-%05d.gguf' \
                  "$quant" "$MODEL_PREFIX" "$quant" 1 "$shards")"

    CTX_SIZE="$CTX_TARGET"
    [ "$CTX_SIZE" -gt "$MODEL_MAX_CTX" ] && CTX_SIZE="$MODEL_MAX_CTX"
    [ "$CTX_SIZE" -lt 4096 ] && CTX_SIZE=4096
    CTX_SIZE=$(( (CTX_SIZE / 4096) * 4096 ))

    # Reported as the minimum VRAM this row actually needs, which is more use
    # than a round label when every quant is within a few GiB of the next.
    TIER=$(( MIN_MIB / 1024 ))
    WEIGHTS_MIB="$weights_mib"
    MODEL_QUANT="$quant"
    MODEL_SHARDS="$shards"
}

# llama-fit-params loads the real model on the real card and prints the n_ctx
# and n_gpu_layers that actually fit. Needs the gguf, so it is a cross-check
# after the download, never part of the boot decision. Worth running here more
# than it was on the qwen image, because the KV constants above are derived
# rather than read out of the header.
measure() {
    local model="${MODEL_DIR}/${MODEL_FILE}"
    [ -s "$model" ] || { say "no ${model} to measure, keeping the table's answer"; return 1; }
    local mm=()
    [ "$ENABLE_VISION" = "1" ] && [ -s "${MODEL_DIR}/${MMPROJ_FILE:-}" ] && mm=(--mmproj "${MODEL_DIR}/${MMPROJ_FILE}")
    local out
    out="$(LD_LIBRARY_PATH=/app/bin:/usr/local/cuda/lib64 /app/bin/llama-fit-params \
            -m "$model" "${mm[@]}" \
            -ngl "${N_GPU_LAYERS:-99}" -fa "${FLASH_ATTN:-on}" \
            -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" \
            --fit-target "$FIT_MARGIN_MIB" 2>/dev/null | tail -1)" || return 1
    case "$out" in
        *-c\ *) CTX_SIZE="$(printf '%s' "$out" | sed -E 's/.*-c ([0-9]+).*/\1/')"; return 0 ;;
        *)      say "llama-fit-params gave nothing usable, keeping the table's answer"; return 1 ;;
    esac
}

VRAM_MIB="${VRAM_MIB:-}"
if [ -z "$VRAM_MIB" ]; then
    VRAM_MIB="$(detect_vram_mib)" || { say "no nvidia-smi and no VRAM_MIB; nothing to decide"; exit 1; }
fi

if [ -z "$N_DEVICES" ]; then
    N_DEVICES="$(detect_device_count)" || N_DEVICES=1
fi
[ "$N_DEVICES" -ge 1 ] 2>/dev/null || N_DEVICES=1

pick_for_vram "$VRAM_MIB"
SOURCE=table

if [ "${1:-}" = "--measure" ] && measure; then
    SOURCE=llama-fit-params
fi

per_1k="$KV_MIB_PER_1K_F16"
[ "$CACHE_TYPE" = "q8_0" ] && per_1k="$KV_MIB_PER_1K_Q8"
kv_mib=$(( CTX_SIZE / 1024 * per_1k ))
used_mib=$(( WEIGHTS_MIB + kv_mib ))
[ "$ENABLE_VISION" = "1" ] && used_mib=$(( used_mib + MMPROJ_MIB ))

free_mib=$(( VRAM_MIB - used_mib ))
free_per_dev=$(( free_mib / N_DEVICES ))

say "${VRAM_MIB} MiB over ${N_DEVICES} device(s) -> ${MODEL_QUANT} (${MODEL_SHARDS} shards) + ${CTX_SIZE} ctx @ ${CACHE_TYPE} KV, needs ${TIER} GiB"
say "  weights ${WEIGHTS_MIB} MiB + kv ${kv_mib} MiB + mmproj $([ "$ENABLE_VISION" = "1" ] && echo $MMPROJ_MIB || echo 0) MiB = ${used_mib} MiB"
say "  free ${free_mib} MiB total = ~${free_per_dev} MiB per device (reserve ${FIT_MARGIN_MIB}/device)"

# llama.cpp splits LAYERS, so what matters is the per-device figure. Anything
# under the reserve means the compute buffers have nowhere to go and the load
# dies after the weights are already on the cards.
if [ "$free_per_dev" -lt "$FIT_MARGIN_MIB" ]; then
    say "WARNING only ${free_per_dev} MiB free per device, under the ${FIT_MARGIN_MIB} MiB reserve."
    say "  Raise FIT_MARGIN_MIB, lower CTX_TARGET, or expect an out-of-memory abort."
fi

printf 'VRAM_MIB=%s\n'     "$VRAM_MIB"
printf 'N_DEVICES=%s\n'    "$N_DEVICES"
printf 'TIER=%s\n'         "$TIER"
printf 'MODEL_FILE=%s\n'   "$MODEL_FILE"
printf 'MODEL_QUANT=%s\n'  "$MODEL_QUANT"
printf 'MODEL_SHARDS=%s\n' "$MODEL_SHARDS"
printf 'CTX_SIZE=%s\n'     "$CTX_SIZE"
printf 'CACHE_TYPE=%s\n'   "$CACHE_TYPE"
printf 'SOURCE=%s\n'       "$SOURCE"
