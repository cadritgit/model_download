"""Download official Qwen-Image 2.1 weights for ComfyUI (generation and editing).

Install in the target environment: python -m pip install -U huggingface_hub
Run: python qwen_image_2_1.py
Preview: python qwen_image_2_1.py --dry-run
Source: https://huggingface.co/Comfy-Org/Qwen-Image-2.1
Update ComfyUI before use. No workflow or sampler settings are changed.
License: Qwen Research License; check upstream terms before commercial use.
"""

import argparse
from pathlib import Path

REPO_ID = 'Comfy-Org/Qwen-Image-2.1'
# Pin the verified repository snapshot for reproducible downloads.
REVISION = 'a8342359c61979b06a8dafb1a47991fe1f30e994'
DEFAULT_COMFYUI_DIR = '/workspace/runpod-slim/ComfyUI'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfyui-dir', type=Path, default=Path(DEFAULT_COMFYUI_DIR))
    parser.add_argument('--dry-run', action='store_true', help='List files without downloading or creating folders')
    parser.add_argument('--precision', choices=('bf16', 'int8'), default='int8',
                        help='INT8 ConvRot official workflow default; BF16 is optional')
    args = parser.parse_args(argv)
    suffix = 'bf16' if args.precision == 'bf16' else 'int8_convrot'
    files = [
        f'diffusion_models/qwen_image_2.1_{suffix}.safetensors',
        f'text_encoders/qwen3vl_8b_{suffix}.safetensors',
        'vae/qwen_image_2.1_vae_bf16.safetensors',
        'loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors',
    ]
    models_dir = args.comfyui_dir.expanduser() / 'models'
    tasks = [(REPO_ID, REVISION, filename, models_dir) for filename in files]
    # Creator-recommended step 1500; download only, never auto-apply LoRAs.
    tasks.append(('ausboss/Qwen-Image-2.1-Consistency-LoRA',
                  '1b58c4406925d6ec5fa9abd2d080531030f7cc0d',
                  'qwen-image-2.1-consistency.safetensors', models_dir / 'loras'))
    for repo, revision, filename, destination in tasks:
        print(f'{repo}/{filename} -> {destination / filename}')
    if args.dry_run:
        return 0
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        parser.exit(1, 'huggingface_hub 필요: python -m pip install -U huggingface_hub\n')
    try:
        for repo, revision, filename, destination in tasks:
            hf_hub_download(repo_id=repo, filename=filename,
                            revision=revision, local_dir=str(destination))
    except Exception as exc:
        parser.exit(1, f'다운로드 실패: {type(exc).__name__}. 연결/디스크/접근 권한을 확인하고 재실행하세요.\n')
    print('Qwen-Image 2.1 모델 다운로드 완료!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
