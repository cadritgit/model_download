from huggingface_hub import hf_hub_download

# 1. 다운로드 정보를 리스트로 정리 (레포, 파일, 저장폴더 순서)
download_tasks = [
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",  # 디퓨전 모델
        "dir": "runpod-slim/ComfyUI/models/diffusion_models"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",  # 텍스트 인코더
        "dir": "runpod-slim/ComfyUI/models/text_encoders"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "resolve/main/vae/minimax—h3_video_vae_fp16.safetensors",  # 비디오 VAE
        "dir": "runpod-slim/ComfyUI/models/vae"
    },
    {
        "repo": "Comfy-Org/MiniMax-H3", 
        "file": "resolve/main/vae/minimax—h3_audeo_vae_fp32.safetensors",   # 오디오 VAE
        "dir": "runpod-slim/ComfyUI/models/vae"
    }
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
