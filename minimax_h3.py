from huggingface_hub import hf_hub_download
import subprocess

# 1. Hugging Face 모델 다운로드
download_tasks = [
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "vae/minimax_h3_video_vae_fp16.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models"
    },
    {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras"
    }
]

for task in download_tasks:
    print(f"🚀 {task['repo']}에서 {task['file']} 다운로드 시작...")

    hf_hub_download(
        repo_id=task['repo'],
        filename=task['file'],
        local_dir=task['dir']
    )

print("✅ Hugging Face 모델 다운로드 완료!")


# 2. gdown 설치
print("🚀 gdown 설치 중...")

subprocess.run(
    ["pip", "install", "-q", "gdown"],
    check=True
)


# 3. Google Drive의 MiniMax H3 LoRA 다운로드
print("🚀 MiniMax H3 LoRA 다운로드 중...")

lora_dir = "/workspace/runpod-slim/ComfyUI/models/loras"

subprocess.run(
    [
        "gdown",
        "1y3bCB2K4JgdphfOwxutvP_PUE7LSxB7u",
        "-O",
        lora_dir + "/"
    ],
    check=True
)

print("✅ 모든 다운로드가 완료되었습니다!")
