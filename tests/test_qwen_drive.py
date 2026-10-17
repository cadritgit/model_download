import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'qwen_image_2_1.py'
spec = importlib.util.spec_from_file_location('qwen_drive', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DriveTests(unittest.TestCase):
    def test_drive_download_is_atomic_validated_and_reusable(self):
        import hashlib
        payload = b'test fixture bytes, not model weights'
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'loras' / module.DRIVE_FILENAME
            with patch.object(module, 'DRIVE_SHA256', hashlib.sha256(payload).hexdigest()), patch.object(module, 'DRIVE_SIZE', len(payload)), patch('urllib.request.urlopen', return_value=io.BytesIO(payload)) as request:
                module.download_drive_lora(Path(tmp))
                self.assertEqual(target.read_bytes(), payload)
                module.download_drive_lora(Path(tmp))
                self.assertEqual(request.call_count, 1)
                target.write_bytes(b'broken')
                request.return_value = io.BytesIO(b'<html>quota exceeded</html>')
                with self.assertRaises(ValueError):
                    module.download_drive_lora(Path(tmp))
                self.assertEqual(target.read_bytes(), b'broken')
                self.assertFalse(list(target.parent.glob('*.part')))

    def test_dry_run_includes_drive_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            module.main(['--comfyui-dir', tmp, '--dry-run'])
            self.assertIn('1Wu-smpFKbJ7iVIXQDF3mP9lC75QDy8aD', out.getvalue())
            self.assertIn('models/loras/NSFW Qwen by TheseAlpacas V2.safetensors', out.getvalue())
            self.assertFalse((Path(tmp) / 'models').exists())


if __name__ == '__main__':
    unittest.main()
