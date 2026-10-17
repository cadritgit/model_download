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
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'huggingface_hub': fake_hub}):
            self.assertEqual(module.main(['--precision', 'int8', '--comfyui-dir', tmp]), 0)
            self.assertEqual(download.call_count, 3)
            calls = [call.kwargs for call in download.call_args_list]
            self.assertEqual([c['filename'] for c in calls], [
                'diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
                'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
                'vae/qwen_image_2.1_vae_bf16.safetensors',
            ])
            for call in calls:
                self.assertEqual(call['repo_id'], 'Comfy-Org/Qwen-Image-2.1')
                self.assertEqual(call['revision'], module.REVISION)
                self.assertEqual(call['local_dir'], str(Path(tmp) / 'models'))
            download.side_effect = OSError('network unavailable')
            with self.assertRaises(SystemExit) as failure:
                module.main(['--comfyui-dir', tmp])
            self.assertEqual(failure.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
