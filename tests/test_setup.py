"""Exercise setup with real local Git; stub Python to avoid huge downloads.
Only the fixed workspace and repository URL are redirected into a sandbox.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'runpod-slim'
        (self.base / 'ComfyUI').mkdir(parents=True)
        self.origin = self.root / 'origin'
        self.origin.mkdir()
        self.git('init', '-b', 'main', str(self.origin))
        self.git('-C', str(self.origin), 'config', 'user.name', 'Test')
        self.git('-C', str(self.origin), 'config', 'user.email', 'test@example.invalid')
        for name in ('minimax_h3.py', 'minimax_h3_image.py'):
            (self.origin / name).write_text('# fixture\n')
        self.git('-C', str(self.origin), 'add', '.')
        self.git('-C', str(self.origin), 'commit', '-m', 'fixture')
        self.log = self.root / 'calls'
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        python = bin_dir / 'python3'
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                          'if [[ "$*" == "$FAIL_CALL" ]]; then exit 17; fi\n')
        python.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
                        CALL_LOG=str(self.log), FAIL_CALL='')

    def git(self, *args):
        return subprocess.run(['git', *args], check=True, capture_output=True, text=True)

    def run_setup(self):
        self.assertTrue((REPO / 'setup.sh').exists(), 'setup.sh has not been implemented')
        text = (REPO / 'setup.sh').read_text()
        text = text.replace('/workspace/runpod-slim', str(self.base))
        text = text.replace('https://github.com/cadritgit/model_download.git', str(self.origin))
        script = self.root / 'setup.sh'
        script.write_text(text)
        return subprocess.run(['bash', str(script)], env=self.env,
                              capture_output=True, text=True)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_fresh_clone_and_ordered_downloads(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.base / 'model_download/.git').exists())
        self.assertEqual(self.calls(), ['-m pip install huggingface_hub gdown',
                                       'minimax_h3.py', 'minimax_h3_image.py'])

    def test_missing_comfyui_stops_before_install_or_download(self):
        (self.base / 'ComfyUI').rmdir()
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ComfyUI', result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
