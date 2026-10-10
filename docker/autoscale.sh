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
# 320B of MoE weights needs far more scratch than a 27B dense model did: the
# per-device compute buffers at long context are the reason this default is 4x
# the llama.cpp stock 1024.
FIT_MARGIN_MIB="${FIT_MARGIN_MIB:-4096}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_PREFIX="${MODEL_PREFIX:-GLM-5.3-Flash}"
MMPROJ_MIB=1110
# GLM-5.3-Flash is a 1M-context model. At the KV cost below, 1M of f16 KV is
# ~13 GiB, which is actually affordable on the larger tiers.
MODEL_MAX_CTX=1048576
# What we try to reach before trading f16 KV away for q8_0. See pick_kv().
CTX_TARGET="${CTX_TARGET:-131072}"

# --- KV cost ----------------------------------------------------------------
# ESTIMATED, not read from the GGUF header — unlike the qwen table this was
# derived from the published architecture, so treat it as a starting point and
# cross-check with --measure once the weights are on disk.
#
# 45 layers: 34 KDA linear-attention + 11 DSA sparse-attention over an MLA-only
# path. Only the 11 MLA layers hold a per-token cache; linear attention keeps a
# fixed-size state instead, which is why this is ~5x cheaper per token than a
# 27B dense model.
#
#   11 layers * (kv_lora_rank 512 + qk_rope_head_dim 64) = 6336 elem/token
#   f16:  6336 * 2      = 12672 B/token -> 12.375 MiB per 1024 tokens
#   q8_0: 6336 * 1.0625 =  6732 B/token ->  6.573 MiB per 1024 tokens
KV_MIB_PER_1K_F16=13
KV_MIB_PER_1K_Q8=7
# The 34 KDA layers' recurrent state does not grow with context, but it is not
# free either, and it is charged per parallel sequence slot.
KDA_STATE_MIB=34

# Total across devices: llama.cpp spreads layers over every visible GPU.
detect_vram_mib() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ t += $1 + 0 } END { if (t > 0) print int(t); else exit 1 }'
}

# Each threshold is the real minimum for its row, not a round number:
#
#   weights + FIT_MARGIN_MIB(4096) + KDA_STATE_MIB(34) + mmproj(1110)
#     + 128K of f16 KV(1664)  =  weights + 6904
#
# The tier LABEL is the rough VRAM class it targets; the threshold is what
# actually has to be there. Both matter because no card reports its box number:
# a 5090 reports 32607 MiB, so 4 of them are 130428 and not 131072, and
# comparing against the round number would drop every host a tier.
tier_of() {
    local m="$1"
    if   [ "$m" -ge 286000 ]; then echo 288
    elif [ "$m" -ge 237000 ]; then echo 240
    elif [ "$m" -ge 198000 ]; then echo 192
    elif [ "$m" -ge 157000 ]; then echo 160
    elif [ "$m" -ge 148000 ]; then echo 144
    elif [ "$m" -ge 122000 ]; then echo 128
    elif [ "$m" -ge 111000 ]; then echo 112
    elif [ "$m" -ge 105000 ]; then echo 104
    elif [ "$m" -ge 100000 ]; then echo 96
    elif [ "$m" -ge  96000 ]; then echo 88
    else                           echo 0
    fi
}

# Sets MODEL_FILE / CACHE_TYPE / CTX_SIZE from $1 = total VRAM in MiB.
#
# weights_mib and shards are the real totals from the Unsloth repo, summed over
# every shard of that quant (the 00001 shard is a ~9 MiB index, the rest carry
# the tensors). Sizes in GiB, for reading against the table in the README:
#   IQ1_S 86.69  IQ1_M 90.88  IQ2_XXS 94.85  Q2_K_XL 101.25  IQ3_XXS 112.10
#   Q3_K_XL 137.40  IQ4_XS 146.05  Q4_K_XL 185.99  Q5_K_XL 223.80  Q6_K_XL 271.79
#
# Quality, where it has been measured (perplexity vs BF16 / same-top-p):
#   IQ3_XXS 1.341x / 81.8%      Q2_K_XL  -    / 78%      IQ2_XXS 1.883x / 73.2%
# The IQ2 -> IQ3 step halves KL divergence, so 128 is the lowest tier worth
# trusting for agentic work.
pick_for_vram() {
    local vram_mib="$1"
    local tier weights_mib quant shards
    tier="$(tier_of "$vram_mib")"

    case "$tier" in
        288) quant=UD-Q6_K_XL; shards=7; weights_mib=278314 ;;
        240) quant=UD-Q5_K_XL; shards=6; weights_mib=229174 ;;
        192) quant=UD-Q4_K_XL; shards=6; weights_mib=190456 ;;
        160) quant=UD-IQ4_XS;  shards=5; weights_mib=149557 ;;
        144) quant=UD-Q3_K_XL; shards=4; weights_mib=140701 ;;
        128) quant=UD-IQ3_XXS; shards=4; weights_mib=114791 ;;
        112) quant=UD-Q2_K_XL; shards=4; weights_mib=103684 ;;
        104) quant=UD-IQ2_XXS; shards=4; weights_mib=97127  ;;
        96)  quant=UD-IQ1_M;   shards=3; weights_mib=93059  ;;
        88)  quant=UD-IQ1_S;   shards=3; weights_mib=88775  ;;
        # No quant of a 320B model fits under ~94 GiB of VRAM. Pick the smallest
        # one anyway and say plainly what will happen, rather than inventing a
        # tier that cannot load.
        *)   quant=UD-IQ1_S;   shards=3; weights_mib=88775
             tier=0
             say "WARNING ${vram_mib} MiB is below this model's floor."
             say "  ${quant} alone is ${weights_mib} MiB. Layers will spill to host RAM;"
             say "  drop N_GPU_LAYERS until it fits, or rent more VRAM."
             ;;
    esac

    MODEL_FILE="$(printf '%s/%s-%s-%05d-of-%05d.gguf' \
                  "$quant" "$MODEL_PREFIX" "$quant" 1 "$shards")"

    # Everything the KV pool does NOT get.
    local budget=$(( vram_mib - weights_mib - FIT_MARGIN_MIB - KDA_STATE_MIB ))
    [ "$ENABLE_VISION" = "1" ] && budget=$(( budget - MMPROJ_MIB ))
    [ "$budget" -lt 0 ] && budget=0

    # Keep f16 KV unless CTX_TARGET will not fit in f16, and only then trade it
    # for q8_0. On this architecture f16 at 128K costs ~1.6 GiB, so f16 wins on
    # every tier that can hold the weights at all — but the rule is written out
    # rather than hardcoded so it still self-corrects if the KV estimate above
    # turns out low.
    local per_1k="$KV_MIB_PER_1K_F16"
    CACHE_TYPE=f16
    if [ $(( CTX_TARGET / 1024 * KV_MIB_PER_1K_F16 )) -gt "$budget" ]; then
        CACHE_TYPE=q8_0
        per_1k="$KV_MIB_PER_1K_Q8"
        say "f16 KV cannot reach ${CTX_TARGET} in ${budget} MiB, falling back to q8_0"
    fi

    local max_ctx=$(( (budget / per_1k) * 1024 ))
    CTX_SIZE="$CTX_TARGET"
    if [ "$CTX_SIZE" -gt "$max_ctx" ]; then
        say "target ${CTX_SIZE} does not fit ${vram_mib} MiB, using ${max_ctx}"
        CTX_SIZE="$max_ctx"
    fi

    [ "$CTX_SIZE" -gt "$MODEL_MAX_CTX" ] && CTX_SIZE="$MODEL_MAX_CTX"
    [ "$CTX_SIZE" -lt 4096 ] && CTX_SIZE=4096
    CTX_SIZE=$(( (CTX_SIZE / 4096) * 4096 ))

    TIER="$tier"
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

pick_for_vram "$VRAM_MIB"
SOURCE=table

if [ "${1:-}" = "--measure" ] && measure; then
    SOURCE=llama-fit-params
fi

per_1k="$KV_MIB_PER_1K_F16"
[ "$CACHE_TYPE" = "q8_0" ] && per_1k="$KV_MIB_PER_1K_Q8"
kv_mib=$(( CTX_SIZE / 1024 * per_1k ))
used_mib=$(( WEIGHTS_MIB + kv_mib + KDA_STATE_MIB ))
[ "$ENABLE_VISION" = "1" ] && used_mib=$(( used_mib + MMPROJ_MIB ))

say "${VRAM_MIB} MiB -> ${TIER} GB tier: ${MODEL_QUANT} (${MODEL_SHARDS} shards) + ${CTX_SIZE} ctx @ ${CACHE_TYPE} KV"
say "  weights ${WEIGHTS_MIB} MiB + kv ${kv_mib} MiB + kda ${KDA_STATE_MIB} MiB + mmproj $([ "$ENABLE_VISION" = "1" ] && echo $MMPROJ_MIB || echo 0) MiB = ${used_mib} MiB, $(( VRAM_MIB - used_mib )) MiB free"

printf 'VRAM_MIB=%s\n'     "$VRAM_MIB"
printf 'TIER=%s\n'         "$TIER"
printf 'MODEL_FILE=%s\n'   "$MODEL_FILE"
printf 'MODEL_QUANT=%s\n'  "$MODEL_QUANT"
printf 'MODEL_SHARDS=%s\n' "$MODEL_SHARDS"
printf 'CTX_SIZE=%s\n'     "$CTX_SIZE"
printf 'CACHE_TYPE=%s\n'   "$CACHE_TYPE"
printf 'SOURCE=%s\n'       "$SOURCE"
