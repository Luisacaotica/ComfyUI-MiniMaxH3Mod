"""Isolated gradient feasibility probe using a small, randomly initialized H3.

This does not train a useful RefMod or estimate full-checkpoint memory.
Run with ComfyUI's Python. No production modules or checkpoints are modified.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import struct
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))


def eager_autograd_scope():
    """Temporary substitutions only inside this standalone diagnostic process."""
    import torch
    import comfy.quant_ops
    import comfy.ldm.minimax.model as h3

    def modulate(h, shift, scale, segments):
        return torch.cat([h[a:b] * (1 + scale[row].to(h.dtype)) + shift[row].to(h.dtype)
                          for a,b,row in segments])

    def gate(x, gates, other, segments):
        return torch.cat([x[a:b] + other[a:b] * gates[row].to(x.dtype)
                          for a,b,row in segments])

    def rms_rope(q, k, freqs, qw, kw, epsilon, rot_dim):
        def apply(x, weight):
            x = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, epsilon)
            half = rot_dim // 2
            a, b = x[..., :half], x[..., half:rot_dim]
            c, s = freqs[...,0,0], freqs[...,1,0]
            return torch.cat([a*c-b*s, a*s+b*c, x[...,rot_dim:]], dim=-1)
        return apply(q,qw), apply(k,kw)

    scope = contextlib.ExitStack()
    scope.enter_context(patch.object(h3, "_mod_scale_shift", modulate))
    scope.enter_context(patch.object(h3, "_mod_gate", gate))
    scope.enter_context(patch.object(comfy.quant_ops.ck, "rms_rope_split_half", rms_rope))
    return scope


def checkpoint_inventory(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 128 * 1024**2:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(stream.read(length))
    sizes = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        dtype = value["dtype"]
        sizes[dtype] = sizes.get(dtype, 0) + value["data_offsets"][1] - value["data_offsets"][0]
    return {"path": str(path), "file_gib": Path(path).stat().st_size / 1024**3,
            "storage_gib_by_dtype": {k:v/1024**3 for k,v in sizes.items()},
            "tensor_count": len(header) - int("__metadata__" in header),
            "sample_tensor_names": [k for k in header if k != "__metadata__"][:12]}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="tests/h3_gradient_probe_result.json")
    parser.add_argument("--eager-autograd", action="store_true")
    args = parser.parse_args()
    import psutil
    import torch
    from torch.utils.checkpoint import checkpoint
    from comfy.cli_args import args as comfy_args
    comfy_args.use_pytorch_cross_attention = True
    import comfy.model_management as mm
    from comfy.ldm.minimax.model import MiniMaxH3Model

    device = "cuda"
    torch.manual_seed(401)
    report = {"scope": "Random small H3, float32, reference-input gradients only; no quality claim",
              "gpu": torch.cuda.get_device_name(), "ram_available_gib": psutil.virtual_memory().available/1024**3,
              "cuda_free_gib": torch.cuda.mem_get_info()[0]/1024**3, "results": []}
    if args.checkpoint:
        report["checkpoint"] = checkpoint_inventory(args.checkpoint)
    model = MiniMaxH3Model(hidden_size=128, num_layers=2, token_refiner_num_layers=1,
                          num_attention_heads=2, attention_head_dim=64, ffn_hidden_size=256,
                          text_dim=64, timestep_input_dim=32, time_embed_hidden_size=128,
                          time_embed_dim=64, rope_inv_freq_len=8,
                          dtype=torch.float32, device=device, operations=torch.nn)
    model.rope.inv_freq.copy_(torch.logspace(0, -3, 8))
    model.requires_grad_(False)
    video = torch.randn(1,24,2,8,8, device=device)
    audio = torch.randn(1,32,2,4, device=device)
    context = torch.randn(1,4,64, device=device)
    initial = torch.randn(1,24,1,4,4, device=device)
    timestep = torch.tensor([500.], device=device)
    target = torch.randn_like(video)

    def forward(ref):
        payload = {"refs": [{"kind":"image", "latent_h":4,"latent_w":4}],
                   "cond_video_latents": [ref], "seed":401}
        return model([video,audio], timestep, context, minimax_payload=payload)[0]

    previous = mm.in_training
    mm.in_training = True
    baseline = None
    native_out = forward(initial).detach()
    adapter_scope = eager_autograd_scope() if args.eager_autograd else contextlib.nullcontext()
    adapter_scope.__enter__()
    try:
        if args.eager_autograd:
            adapted_out = forward(initial).detach()
            torch.testing.assert_close(adapted_out, native_out, rtol=1e-4, atol=1e-5)
            report["eager_forward_max_error"] = (adapted_out-native_out).abs().max().item()
        for strategy in ("direct", "checkpoint", "saved_activations_cpu"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ref = initial.clone().requires_grad_(True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = {"strategy":strategy}
            try:
                scope = torch.autograd.graph.save_on_cpu(pin_memory=True) if strategy == "saved_activations_cpu" else contextlib.nullcontext()
                with scope:
                    out = checkpoint(forward, ref, use_reentrant=False) if strategy == "checkpoint" else forward(ref)
                    loss = (out-target).square().mean()
                gradient, = torch.autograd.grad(loss, ref)
                if not torch.isfinite(gradient).all() or gradient.norm() == 0:
                    raise AssertionError("Reference gradient is nonfinite or zero")
                if baseline is None:
                    baseline = gradient.detach().clone()
                torch.testing.assert_close(gradient, baseline, rtol=1e-4, atol=1e-6)
                result.update(passed=True, loss=loss.item(), gradient_norm=gradient.norm().item())
                del out, loss, gradient
            except (RuntimeError, AssertionError, NotImplementedError) as error:
                result.update(passed=False, error=str(error))
            torch.cuda.synchronize()
            result.update(seconds=time.perf_counter()-started,
                          peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
            report["results"].append(result)
            del ref
        if args.eager_autograd and all(r["passed"] for r in report["results"]):
            ref = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.Adam([ref], lr=0.03)
            losses = []
            for _ in range(6):
                optimizer.zero_grad()
                loss = (forward(ref)-target).square().mean()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if losses[-1] >= losses[0]:
                raise AssertionError("Synthetic optimization did not reduce loss")
            report["synthetic_optimization_losses"] = losses
            report["model_weight_gradients_absent"] = all(p.grad is None for p in model.parameters())
    finally:
        adapter_scope.__exit__(None,None,None)
        mm.in_training = previous
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
