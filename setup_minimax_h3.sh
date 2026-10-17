#!/usr/bin/env bash
# Runpod MiniMax H3 model downloads (does not start ComfyUI).
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

printf '[1/4] Preparing model_download repository...\n'
mkdir -p "$BASE_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
printf '[2/4] Installing Python download dependencies...\n'
python3 -m pip install -U huggingface_hub hf_xet gdown

printf '[3/4] Downloading MiniMax H3 video models and LoRAs...\n'
python3 minimax_h3.py

printf '[4/4] Downloading MiniMax H3 image VAE...\n'
python3 minimax_h3_image.py

printf '\n[DONE] Both model download scripts completed successfully.\n'
