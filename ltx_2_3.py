from huggingface_hub import hf_hub_download

# 1. 다운로드 정보를 리스트로 정리 (레포, 파일, 저장폴더 순서)
download_tasks = [
    {
        "repo": "Lightricks/LTX-2.3-fp8",  # 체크포인트
        "file": "ltx-2.3-22b-dev-fp8.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/checkpoints" # diffusion_models 폴더가 아님에 주의
    },
    {
        "repo": "Lightricks/LTX-2.3", # Distilled Lora
        "file": "ltx-2.3-22b-distilled-lora-384.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras" # 대문자 아님에 주의
    },
    {
        "repo": "Lightricks/LTX-2.3", # 업스케일러
        "file": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/latent_upscale_models" # upsclae_models 폴더가 아님에 주의
    },
    {
        "repo": "Comfy-Org/ltx-2", # 텍스트 인코더
        "file": "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/text_encoders"
    },
    {
        "repo": "Comfy-Org/ltx-2", # abliterated Lora
        "file": "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors", 
        "dir": "/workspace/runpod-slim/ComfyUI/models/loras" # 대문자 아님에 주의
    },
    {
        "repo": "SulphurAI/Sulphur-2-base", # Sulphur 2 distilled Lora 
        "file": "sulphur_lora_rank_768.safetensors", 
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
