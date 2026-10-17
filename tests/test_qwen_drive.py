import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'qwen_image_2_1.py'
spec = importlib.util.spec_from_file_location('qwen_drive', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DriveTests(unittest.TestCase):
    def test_drive_download_uses_gdown_and_validates_before_replacing(self):
        payload = b'test fixture bytes, not model weights'
        fake = types.ModuleType('gdown')
        def transfer(**kwargs):
            Path(kwargs['output']).write_bytes(payload)
            return kwargs['output']
        fake.download = Mock(side_effect=transfer)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'gdown': fake}), patch.object(module, 'DRIVE_SHA256', hashlib.sha256(payload).hexdigest()), patch.object(module, 'DRIVE_SIZE', len(payload)):
            target = Path(tmp) / 'loras' / module.DRIVE_FILENAME
            module.download_drive_lora(Path(tmp))
            self.assertEqual(target.read_bytes(), payload)
            fake.download.assert_called_once_with(url=module.DRIVE_URL, output=str(target) + '.part', quiet=False, resume=True, use_cookies=False)
            module.download_drive_lora(Path(tmp))
            self.assertEqual(fake.download.call_count, 1)
            target.write_bytes(b'original')
            def invalid(**kwargs):
                Path(kwargs['output']).write_bytes(b'<html>quota</html>')
                return kwargs['output']
            fake.download.side_effect = invalid
            with self.assertRaises(ValueError):
                module.download_drive_lora(Path(tmp))
            self.assertEqual(target.read_bytes(), b'original')
            self.assertFalse(Path(str(target) + '.part').exists())
            fake.download.side_effect = None
            fake.download.return_value = None
            with self.assertRaises(RuntimeError):
                module.download_drive_lora(Path(tmp))
            self.assertEqual(target.read_bytes(), b'original')

    def test_dry_run_includes_drive_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            module.main(['--comfyui-dir', tmp, '--dry-run'])
            self.assertIn('1Wu-smpFKbJ7iVIXQDF3mP9lC75QDy8aD', out.getvalue())
            self.assertIn('models/loras/NSFW Qwen by TheseAlpacas V2.safetensors', out.getvalue())
            self.assertFalse((Path(tmp) / 'models').exists())


if __name__ == '__main__':
    unittest.main()
