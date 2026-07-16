#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/workspace/yuhang/minwm"
CKPT_ROOT="${PROJECT_ROOT}/ckpts"

cd "${PROJECT_ROOT}"
mkdir -p "${CKPT_ROOT}"

echo "========================================"
echo "1. Download Wan2.1 base model"
echo "========================================"

hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir "${CKPT_ROOT}/Wan2.1-T2V-1.3B"

echo "========================================"
echo "2. Create Wan hardcoded model symlink"
echo "========================================"

mkdir -p "${PROJECT_ROOT}/Wan21/wan_models"

rm -f "${PROJECT_ROOT}/Wan21/wan_models/Wan2.1-T2V-1.3B"

ln -s \
    "$(realpath "${CKPT_ROOT}/Wan2.1-T2V-1.3B")" \
    "${PROJECT_ROOT}/Wan21/wan_models/Wan2.1-T2V-1.3B"

echo "========================================"
echo "3. Download HunyuanVideo 1.5 components"
echo "========================================"

hf download tencent/HunyuanVideo-1.5 \
    --local-dir "${CKPT_ROOT}/HunyuanVideo-1.5" \
    --include \
        "vae/*" \
        "scheduler/*" \
        "transformer/480p_i2v/*"

echo "========================================"
echo "4. Download Qwen text encoder"
echo "========================================"

hf download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir "${CKPT_ROOT}/HunyuanVideo-1.5/text_encoder/llm"

echo "========================================"
echo "5. Download ByT5 text encoder"
echo "========================================"

hf download google/byt5-small \
    --local-dir "${CKPT_ROOT}/HunyuanVideo-1.5/text_encoder/byt5-small"

echo "========================================"
echo "6. Download Glyph-SDXL-v2"
echo "========================================"

modelscope download \
    --model AI-ModelScope/Glyph-SDXL-v2 \
    --local_dir "${CKPT_ROOT}/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2"

echo "========================================"
echo "7. Download FLUX Redux vision encoder"
echo "========================================"

hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir "${CKPT_ROOT}/HunyuanVideo-1.5/vision_encoder/siglip"

echo "========================================"
echo "8. Download all Wan Action2V checkpoints"
echo "========================================"

hf download MIN-Lab/minWM \
    --local-dir "${CKPT_ROOT}" \
    --include \
        "Wan21/Action2V/bidirectional/*" \
        "Wan21/Action2V/ar_diffusion_tf/*" \
        "Wan21/Action2V/causal_ode/*" \
        "Wan21/Action2V/causal_cd/*" \
        "Wan21/Action2V/dmd/*"

echo "========================================"
echo "9. Download all HY Action2V checkpoints"
echo "========================================"

hf download MIN-Lab/minWM \
    --local-dir "${CKPT_ROOT}" \
    --include \
        "HY15/Action2V/bidirectional/*" \
        "HY15/Action2V/ar_diffusion_tf/*" \
        "HY15/Action2V/causal_ode/*" \
        "HY15/Action2V/causal_cd/*" \
        "HY15/Action2V/dmd/*" \
        "HY15/Action2V/dmd_ourbi/*"

echo "========================================"
echo "10. Download all HY TI2V checkpoints"
echo "========================================"

hf download MIN-Lab/minWM \
    --local-dir "${CKPT_ROOT}" \
    --include \
        "HY15/TI2V/bidirectional/*" \
        "HY15/TI2V/ar_diffusion_tf/*" \
        "HY15/TI2V/causal_ode/*" \
        "HY15/TI2V/causal_cd/*" \
        "HY15/TI2V/dmd/*"

echo "========================================"
echo "All downloads completed"
echo "Checkpoint directory: ${CKPT_ROOT}"
echo "========================================"