"""Real H3 audio VAE encode/save/load/decode smoke test; no DiT required."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--seconds", type=float, default=0.5)
parser.add_argument("--output", default="tests/audio_smoke_result.json")
opts = parser.parse_args()

import torch
import comfy.sd
import comfy.utils
from comfy_extras.nodes_audio import vae_decode_audio
from comfy.ldm.minimax.model import PackedLayout

# Match ComfyUI's inference execution environment for the diagnostic process.
torch.set_grad_enabled(False)
audio_module = importlib.import_module(f"custom_nodes.{ROOT.name}.audio")
sd, metadata = comfy.utils.load_torch_file(opts.checkpoint, return_metadata=True)
vae = comfy.sd.VAE(sd=sd, metadata=metadata)
vae.throw_exception_if_invalid()
del sd
t = torch.arange(round(opts.seconds * 32000)) / 32000
waveform = (0.1 * torch.sin(2 * torch.pi * 220 * t)).reshape(1, 1, -1)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
mod = audio_module.make_audio_mod(vae, {"waveform": waveform, "sample_rate":32000}, "smoke", max_seconds=opts.seconds)
assert torch.isfinite(mod.latent).all()
with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "smoke")
    mod.save(path)
    loaded = type(mod).load(path)
    torch.testing.assert_close(loaded.latent, mod.latent, rtol=0, atol=0)
block = loaded.ref_block()
layout = PackedLayout(4, 1, 4, 4, 20, refs=[block])
decoded = vae_decode_audio(vae, {"samples": loaded.latent})
assert torch.isfinite(decoded["waveform"]).all()
result = {"checkpoint":opts.checkpoint, "latent_shape":list(mod.latent.shape),
          "tokens":mod.token_count, "decoded_shape":list(decoded["waveform"].shape),
          "sample_rate":decoded["sample_rate"], "seconds_elapsed":time.perf_counter()-start,
          "peak_cuda_mib":torch.cuda.max_memory_allocated()/1024**2,
          "native_layout":True, "roundtrip_exact":True, "finite":True}
Path(opts.output).write_text(json.dumps(result,indent=2), encoding="utf-8")
print(json.dumps(result,indent=2))
