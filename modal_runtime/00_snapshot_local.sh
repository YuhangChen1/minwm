#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-minwm}"

MINWM_ROOT="${MINWM_ROOT:-/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minWM}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minwm_modal_artifacts}"

LOCK_ROOT="${MINWM_ROOT}/modal_runtime/locks"

mkdir -p "${ARTIFACT_ROOT}"
mkdir -p "${LOCK_ROOT}"

# 初始化 Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "============================================================"
echo "Conda environment"
echo "============================================================"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "Python=$(which python)"
python --version

if [[ "$(basename "${CONDA_PREFIX}")" != "${ENV_NAME}" ]]; then
    echo "ERROR: 当前环境不是 ${ENV_NAME}"
    exit 1
fi

echo
echo "============================================================"
echo "记录环境信息"
echo "============================================================"

# 精确 Conda 包记录，用于审计和后续重建
conda list --explicit > \
    "${LOCK_ROOT}/conda-explicit-linux-64.txt"

# 更容易阅读的环境文件
conda env export --no-builds \
    | sed '/^prefix:/d' \
    > "${LOCK_ROOT}/environment-reference.yml"

# 完整 pip 记录
python -m pip freeze > \
    "${LOCK_ROOT}/pip-freeze-reference.txt"

# 系统信息
cp /etc/os-release \
    "${LOCK_ROOT}/os-release.txt"

uname -a > \
    "${LOCK_ROOT}/uname.txt"

# 保存关键版本和 CUDA 信息
python - <<'PY' > "${LOCK_ROOT}/runtime-fingerprint.json"
import json
import platform
import sys

result = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
}

try:
    import torch

    result["torch"] = torch.__version__
    result["torch_cuda_version"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    result["torch_error"] = repr(exc)

try:
    import transformers
    result["transformers"] = transformers.__version__
except Exception as exc:
    result["transformers_error"] = repr(exc)

print(json.dumps(result, indent=2, ensure_ascii=False))
PY

# 获取当前环境所有顶层可导入模块，后面在 H100 上逐个检查
python - <<'PY'
import importlib.metadata
import json
from pathlib import Path

mapping = importlib.metadata.packages_distributions()

modules = sorted(
    module
    for module in mapping
    if module.isidentifier()
    and not module.startswith("_")
)

output = Path(
    "/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/"
    "minWM/modal_runtime/locks/top_level_modules.json"
)

output.write_text(
    json.dumps(modules, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Exported {len(modules)} importable top-level modules")
PY

echo
echo "============================================================"
echo "安装 conda-pack 到 base 环境"
echo "不会修改 minwm 环境"
echo "============================================================"

conda install \
    -n base \
    -c conda-forge \
    conda-pack \
    -y

echo
echo "============================================================"
echo "打包 minwm 环境"
echo "============================================================"

conda run -n base \
    conda-pack \
    -n "${ENV_NAME}" \
    -o "${ARTIFACT_ROOT}/minwm-conda-env.tar.gz" \
    --force

sha256sum \
    "${ARTIFACT_ROOT}/minwm-conda-env.tar.gz" \
    > "${ARTIFACT_ROOT}/minwm-conda-env.tar.gz.sha256"

echo
echo "Environment archive:"
ls -lh "${ARTIFACT_ROOT}/minwm-conda-env.tar.gz"

echo
echo "SHA256:"
cat "${ARTIFACT_ROOT}/minwm-conda-env.tar.gz.sha256"

echo
echo "Local environment snapshot completed."

