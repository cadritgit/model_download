from huggingface_hub import hf_hub_download

# 1. 다운로드 정보를 리스트로 정리 (레포, 파일, 저장폴더 순서)
download_tasks = [
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/diffusion_models"
    },
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/diffusion_models"
    },
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras"
    },
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras"
    },
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/text_encoders"
    },
    {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", 
        "file": "split_files/vae/wan_2.1_vae.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/vae"
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
