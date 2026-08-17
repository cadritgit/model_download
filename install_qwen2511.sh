#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Qwen Image Edit 2511 - Model / LoRA Downloader
# ============================================================

COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"

DIFFUSION_DIR="${COMFYUI_DIR}/models/diffusion_models"
TEXT_ENCODER_DIR="${COMFYUI_DIR}/models/text_encoders"
VAE_DIR="${COMFYUI_DIR}/models/vae"
LORA_DIR="${COMFYUI_DIR}/models/loras"

echo "============================================================"
echo "🚀 Qwen Image Edit 2511 모델 다운로드 시작"
echo "============================================================"

# ------------------------------------------------------------
# 1. 필요한 디렉터리 생성
# ------------------------------------------------------------

mkdir -p "$DIFFUSION_DIR"
mkdir -p "$TEXT_ENCODER_DIR"
mkdir -p "$VAE_DIR"
mkdir -p "$LORA_DIR"

# ------------------------------------------------------------
# 2. 필요한 Python 패키지 설치
# ------------------------------------------------------------

echo ""
echo "📦 huggingface_hub / gdown 설치 중..."

python -m pip install -U huggingface_hub gdown

# ------------------------------------------------------------
# 3. Hugging Face 모델 다운로드
# ------------------------------------------------------------

echo ""
echo "🤗 Hugging Face 모델 다운로드 시작..."

python << 'PYTHON_SCRIPT'

from huggingface_hub import hf_hub_download

download_tasks = [
    {
        "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "file": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/diffusion_models",
    },
    {
        "repo": "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "file": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/text_encoders",
    },
    {
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "file": "split_files/vae/qwen_image_vae.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/vae",
    },
    {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras",
    },
]

for index, task in enumerate(download_tasks, start=1):
    print()
    print(f"🚀 [{index}/{len(download_tasks)}]")
    print(f"   Repo : {task['repo']}")
    print(f"   File : {task['file']}")
    print(f"   Dir  : {task['dir']}")

    downloaded_path = hf_hub_download(
        repo_id=task["repo"],
        filename=task["file"],
        local_dir=task["dir"],
    )

    print(f"✅ 다운로드 완료: {downloaded_path}")

print()
print("✅ Hugging Face 모델 다운로드 완료!")

PYTHON_SCRIPT

# ------------------------------------------------------------
# 4. Google Drive Qwen2511 LoRA 다운로드
# ------------------------------------------------------------

echo ""
echo "☁️ Google Drive Qwen2511 LoRA 다운로드 시작..."

cd "$LORA_DIR"

echo ""
echo "🚀 LoRA 1/2 다운로드..."
gdown "1HMT_xyvOFMOYcm7epSKXfFBfQW0HIk66"

echo ""
echo "🚀 LoRA 2/2 다운로드..."
gdown "133KXwzIuZ7TdkSZCZZjGbew-Ls_VabwP"

# ------------------------------------------------------------
# 5. 완료
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "✅ 모든 다운로드가 완료되었습니다!"
echo "============================================================"
echo ""
echo "📁 Diffusion Model:"
echo "   $DIFFUSION_DIR"
echo ""
echo "📁 Text Encoder:"
echo "   $TEXT_ENCODER_DIR"
echo ""
echo "📁 VAE:"
echo "   $VAE_DIR"
echo ""
echo "📁 LoRA:"
echo "   $LORA_DIR"
echo ""
