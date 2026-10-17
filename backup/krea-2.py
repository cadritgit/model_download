from huggingface_hub import hf_hub_download

# 1. 다운로드 정보를 리스트로 정리 (레포, 파일, 저장폴더 순서)
download_tasks = [
    {
        "repo": "Comfy-Org/Krea-2",  # 디퓨전 모델
        "file": "diffusion_models/krea2_turbo_fp8_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/diffusion_models" 
    },
    {
        "repo": "Comfy-Org/Krea-2", # Turbo Lora
        "file": "loras/krea2_turbo_lora_rank_64_bf16.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras" # 대문자 아님에 주의
    },
    {
        "repo": "Comfy-Org/Krea-2", # 텍스트 인코더
        "file": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/text_encoders"
    },
    {
        "repo": "Comfy-Org/Krea-2", # VAE
        "file": "vae/qwen_image_vae.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/vae" 
    },
    {
        "repo": "conradlocke/krea2-identity-edit", # krea2-identity-edit Lora
        "file": "krea2_identity_edit_v1_2.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras" # 대문자 아님에 주의
    },


]

# 2. 반복문으로 각기 다른 설정을 적용해 다운로드
for task in download_tasks:
    print(f"🚀 {task['repo']}에서 {task['file']} 다운로드 시작...")
    
    hf_hub_download(
        repo_id=task['repo'],
        filename=task['file'],
        local_dir=task['dir'],
        local_dir_use_symlinks=False
    )

print("✅ 서로 다른 작업이 모두 완료되었습니다!")
