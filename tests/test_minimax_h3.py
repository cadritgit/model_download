"""Exercise the downloader without network, package installs or model files."""
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


class MiniMaxDownloadsTest(unittest.TestCase):
    def test_adds_singularity_without_changing_existing_downloads(self):
        hub = types.ModuleType("huggingface_hub")
        download = Mock()
        setattr(hub, "hf_hub_download", download)
        script = Path(__file__).resolve().parents[1] / "minimax_h3.py"
        with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch.object(subprocess, "run") as commands, \
                contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script), run_name="__main__")
        models = "/workspace/runpod-slim/ComfyUI/models"
        expected = [
            ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", models),
            ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", models),
            ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", models),
            ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", models),
            ("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", models + "/loras"),
        ]
        actual = [(c.kwargs["repo_id"], c.kwargs["filename"], c.kwargs["local_dir"])
                  for c in download.call_args_list]
        singularity = ("WarmBloodAban/Minimax-h3_Singularity",
                       "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
                       models + "/diffusion_models")
        self.assertIn(singularity, actual)
        self.assertEqual([task for task in actual if task != singularity], expected)
        self.assertEqual(actual.count(singularity), 1)
        self.assertEqual(commands.call_count, 2)
        self.assertEqual(commands.call_args_list[0].args[0], ["pip", "install", "-q", "gdown"])
        self.assertEqual(commands.call_args_list[1].args[0],
                         ["gdown", "1y3bCB2K4JgdphfOwxutvP_PUE7LSxB7u", "-O", models + "/loras/"])
        self.assertTrue(all(c.kwargs["check"] for c in commands.call_args_list))


if __name__ == "__main__":
    unittest.main()
