#!/usr/bin/env bash
# Runpod Qwen Image 2.1 model downloads; does not install or start ComfyUI.
# Default: official workflow INT8 ConvRot model/encoder + BF16 VAE.
set -euo pipefail
trap 'printf "\n[ERROR] setup failed at line %s. Fix the error above and rerun.\n" "$LINENO" >&2' ERR

BASE_DIR='/workspace/runpod-slim'
REPO_DIR="$BASE_DIR/model_download"
REPO_URL='https://github.com/cadritgit/model_download.git'

for cmd in git python3; do
  command -v "$cmd" >/dev/null || { printf '[ERROR] Missing command: %s\n' "$cmd" >&2; exit 1; }
done

if [[ ! -d "$BASE_DIR/ComfyUI" ]]; then
  printf '[ERROR] ComfyUI directory not found: %s/ComfyUI\nUse this script on the runpod-slim ComfyUI template.\n' "$BASE_DIR" >&2
  exit 1
fi

printf '[1/3] Preparing model_download repository...\n'
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
printf '[2/3] Installing Python download dependencies...\n'
python3 -m pip install -U huggingface_hub hf_xet gdown

printf '[3/3] Downloading Qwen Image 2.1 official default models...\n'
python3 qwen_image_2_1.py

printf '\n[DONE] Qwen Image 2.1 model downloads completed successfully.\n'
