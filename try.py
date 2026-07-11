unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY
unset ALL_PROXY
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
export PYTHONNOUSERSITE=1

export HF_ENDPOINT=https://hf-mirror.com

INCLUDES=("preencode_input.json" "others/HY/Action2V/**")
for i in $(seq 0 19); do
  id=$(printf "%06d" "$i")
  INCLUDES+=("videos/${id}_*/gen.mp4")
done

hf download MIN-Lab/minWM-data \
  --repo-type dataset \
  --local-dir ./dataset_min20 \
  --include "${INCLUDES[@]}"


export PIP_CACHE_DIR=/hyperai/home/.cache/pip
export CONDA_PKGS_DIRS=/hyperai/home/.cache/conda
export HF_HOME=/hyperai/home/.cache/huggingface
export TORCH_HOME=/hyperai/home/.cache/torch
export TRITON_CACHE_DIR=/hyperai/home/.cache/triton