from huggingface_hub import hf_hub_download
import subprocess

# 1. Hugging Face 모델 다운로드
download_tasks = [
    {
        "repo": "Mamad8/MiniMax-H3-Image-VAE", 
        "file": "minimax_h3_t1_image_vae_step1597.safetensors",
        "dir": "/workspace/runpod-slim/ComfyUI/models/vae"
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

