"""Run the shell setup with real sandbox Git and stubbed Python downloads."""
import subprocess
import test_setup
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class QwenSetupTests(test_setup.SetupTests):
    def run_setup(self):
        source = REPO / 'setup_qwen_image_2_1.sh'
        self.assertTrue(source.exists(), 'Qwen setup script is missing')
        text = source.read_text().replace('/workspace/runpod-slim', str(self.base))
        text = text.replace('https://github.com/cadritgit/model_download.git', str(self.origin))
        script = self.root / 'setup_qwen_image_2_1.sh'
        script.write_text(text)
        return subprocess.run(['bash', str(script)], env=self.env,
                              capture_output=True, text=True)

    def test_fresh_clone_and_ordered_downloads(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.base / 'model_download/.git').exists())
        self.assertEqual(self.calls(), ['-m pip install -U huggingface_hub hf_xet gdown',
                                       'qwen_image_2_1.py'])
        # Re-running uses fast-forward pull and still succeeds.
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_install_failure_stops_before_download(self):
        self.env['FAIL_CALL'] = '-m pip install -U huggingface_hub hf_xet gdown'
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [self.env['FAIL_CALL']])
        self.assertNotIn('[DONE]', result.stdout)

    def test_download_failure_is_not_reported_as_success(self):
        self.env['FAIL_CALL'] = 'qwen_image_2_1.py'
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('[DONE]', result.stdout)
