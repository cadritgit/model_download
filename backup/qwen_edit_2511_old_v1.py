from huggingface_hub import hf_hub_download

# 1. 다운로드 정보를 리스트로 정리 (레포, 파일, 저장폴더 순서)
download_tasks = [
    {
        "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI", 
        "file": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/diffusion_models"
    },
    {
        "repo": "Comfy-Org/HunyuanVideo_1.5_repackaged", 
        "file": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/text_encoders"
    },
    {
        "repo": "Comfy-Org/Qwen-Image_ComfyUI", 
        "file": "split_files/vae/qwen_image_vae.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/vae"
    },
    {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning", 
        "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras"
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
