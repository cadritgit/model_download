import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'qwen_image_2_1.py'


class QwenDownloadTests(unittest.TestCase):
    def test_default_dry_run_lists_official_int8_files_without_writes(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'ComfyUI'
            result = subprocess.run(
                ['python', str(SCRIPT), '--comfyui-dir', str(target), '--dry-run'],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('diffusion_models/qwen_image_2.1_int8_convrot.safetensors', result.stdout)
            self.assertIn('text_encoders/qwen3vl_8b_int8_convrot.safetensors', result.stdout)
            self.assertIn('vae/qwen_image_2.1_vae_bf16.safetensors', result.stdout)
            self.assertFalse(target.exists())
            self.assertIn('loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors', result.stdout)
            self.assertIn('loras/qwen-image-2.1-consistency.safetensors', result.stdout)

    def test_int8_download_uses_pinned_revision_and_correct_layout(self):
        import sys
        import tempfile
        import types
        from unittest.mock import Mock, patch
        spec = importlib.util.spec_from_file_location('qwen_download', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        download = Mock()
        fake_hub = types.ModuleType('huggingface_hub')
        fake_hub.hf_hub_download = download
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'huggingface_hub': fake_hub}), patch.object(module, 'download_drive_lora') as drive:
            self.assertEqual(module.main(['--precision', 'int8', '--comfyui-dir', tmp]), 0)
            drive.assert_called_once_with(Path(tmp) / 'models')
            self.assertEqual(download.call_count, 5)
            calls = [call.kwargs for call in download.call_args_list]
            self.assertEqual([c['filename'] for c in calls], [
                'diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
                'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
                'vae/qwen_image_2.1_vae_bf16.safetensors',
                'loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors',
                'qwen-image-2.1-consistency.safetensors',
            ])
            for call in calls[:4]:
                self.assertEqual(call['repo_id'], 'Comfy-Org/Qwen-Image-2.1')
                self.assertEqual(call['revision'], module.REVISION)
                self.assertEqual(call['local_dir'], str(Path(tmp) / 'models'))
            self.assertEqual(calls[4]['repo_id'], 'ausboss/Qwen-Image-2.1-Consistency-LoRA')
            self.assertEqual(calls[4]['revision'], '1b58c4406925d6ec5fa9abd2d080531030f7cc0d')
            self.assertEqual(calls[4]['local_dir'], str(Path(tmp) / 'models' / 'loras'))
            download.side_effect = OSError('network unavailable')
            with self.assertRaises(SystemExit) as failure:
                module.main(['--comfyui-dir', tmp])
            self.assertEqual(failure.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
