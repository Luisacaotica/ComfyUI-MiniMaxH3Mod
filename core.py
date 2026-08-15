"""
core.py — "RefMod" adapter for MiniMax H3 (no training required)

The reference-video path of MiniMax H3 works by injecting *reference tokens*
into the packed sequence: the ref is VAE-encoded, patchified, projected and
placed on the 3D RoPE grid, then every DiT block attends to it.  A video ref
is expensive because it contributes thousands of tokens.

A RefMod is the same reference, compressed to a handful of tokens:

  * the ref is VAE-encoded to its full latent [1, 24, T, H, W],
  * the latent is average-pooled to a tiny grid (default 4x4) and a few
    latent frames, so the patchified token count drops to ~4-16,
  * optionally the small latent is refined with a few gradient steps that
    reconstruct the full latent (still no model weights involved).

At generation time the pooled latent is handed back to the model through the
native ``refs`` payload, so it flows through the exact same per-block
attention machinery as a full reference — the only thing that changes is the
token budget.

This is the MiniMax H3 analog of the LTX "Mod" concept tokens + per-block
injection: instead of a trained hypernetwork predicting AdaLN deltas, the
model's own cross-attention over the compressed ref tokens does the work.
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Metadata key embedded in the safetensors header (keeps a mod in one file,
# so it can be shared/uploaded as a single artifact).
META_KEY = "refmod_meta"


def read_refmod_meta(path_no_ext: str) -> Optional[Dict]:
    """
    Read the metadata block from a mod/preset file without loading tensors.

    Prefers the metadata embedded in the safetensors header; falls back to
    the legacy sidecar ``{path}.json`` written by older versions.
    """
    try:
        with safe_open(path_no_ext + ".safetensors", framework="pt") as f:
            meta = f.metadata()
        if meta and META_KEY in meta:
            return json.loads(meta[META_KEY])
    except Exception:
        pass
    jpath = path_no_ext + ".json"
    if os.path.isfile(jpath):
        try:
            with open(jpath) as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Latent compression
# ═══════════════════════════════════════════════════════════════════════════

def pool_latent(
    z: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """
    Average-pool a VAE latent ``[1, 24, T, H, W]`` down to a tiny grid.

    ``latent_h/latent_w`` are the *latent* grid dims (the DiT patches each 2x2
    latent cell into one token), so a 4x4 latent = 2x2 = 4 tokens per frame.
    Dims must be even (the DiT's 2x2 patch).  Pooling runs in fp32; the result
    keeps the input dtype.
    """
    if z.shape[2] == latent_t and z.shape[3] == latent_h and z.shape[4] == latent_w:
        return z
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise ValueError(f"pool_h/pool_w must be even (got {latent_h}x{latent_w})")
    pooled = F.adaptive_avg_pool3d(
        z.float(), (latent_t, latent_h, latent_w)
    )
    return pooled.to(z.dtype)


def optimize_latent(
    z_small: torch.Tensor,
    z_full: torch.Tensor,
    steps: int = 150,
    lr: float = 0.02,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Model-free refinement of the compressed latent.

    Optimizes the small latent so its trilinearly upsampled reconstruction
    matches the full reference latent.  This pulls the pooled representation
    closer to what the DiT would see from the full ref, with nothing but the
    tiny latent trainable (~1-2K params) and no diffusion model loaded.

    Returns the refined latent detached, same shape/dtype as ``z_small``.
    """
    if steps <= 0:
        return z_small
    device = device or z_full.device
    # Node execution runs under ComfyUI's global inference mode, which would
    # disable the autograd this refine loop needs.  Re-enable it for this
    # scope and clone the inputs to drop the inference flag.
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        target = z_full.clone().float().to(device)
        param = nn.Parameter(z_small.clone().float().to(device))
        opt = torch.optim.Adam([param], lr=lr)
        size = tuple(target.shape[2:])
        for _ in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=size, mode="trilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()
        # materialize inside the scope so the result is a normal tensor, not
        # an inference-mode tensor (it gets stored in the mod and reused)
        refined = param.detach().to(z_small.dtype)
    return refined


# ═══════════════════════════════════════════════════════════════════════════
# H3RefMod — the saved artifact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class H3RefMod:
    """
    A compressed reference for MiniMax H3.

    ``latent`` is the VAE latent ``[1, 24, latent_t, latent_h, latent_w]`` — a
    full-resolution encode (``mode="full"``) or a pooled thumbnail
    (``mode="pooled"``).  ``kind`` is ``"image"`` (single latent frame) or
    ``"video"`` (a few frames), matching the native ref block kinds the
    model's ``PackedLayout`` understands.

    ``mode="full"`` stores the encode at the resolution the official ref2video
    path uses, so the injected ref carries real identity detail; ``mode="pooled"``
    stores a tiny average-pooled grid (concept/motion only).
    """

    name: str
    kind: str
    latent: torch.Tensor
    latent_h: int = 4
    latent_w: int = 4
    latent_t: int = 1
    mode: str = "pooled"
    source: str = ""          # "image" | "video" | "manual"
    source_shape: str = ""    # original latent dims as "TxHxW"
    pool: str = "4x4x1"       # pool_t x pool_h x pool_w
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in ("image", "video"):
            raise ValueError(f"kind must be 'image' or 'video' (got {self.kind!r})")
        if self.kind == "image":
            self.latent_t = 1

    # ── token budget ──────────────────────────────────────────────────

    @property
    def token_count(self) -> int:
        """Number of patchified tokens the mod injects into the packed sequence."""
        per_frame = (self.latent_h // 2) * (self.latent_w // 2)
        return self.latent_t * per_frame

    # ── native ref block ──────────────────────────────────────────────

    def ref_block(self, strength: float = 1.0) -> Optional[Dict]:
        """
        Build the ref block dict the model's ``PackedLayout`` / payload consumes.

        The shape mirrors what the native ref2va nodes emit, so the pooled
        latent rides the exact same path: ``cond_video_latents`` -> patchify ->
        ``video_patch_proj`` -> packed sequence with a 3D RoPE grid.

        ``strength`` uses the model's own conditioning-strength mechanism:
        refs are weakened by mixing the latent toward noise, exactly like the
        model's ``visual_cond_noise_aug`` (``aug * z + (1 - aug) * noise``).
        Scaling values toward zero instead pushes them out of the normalized
        latent distribution, which reads as grey output, not as a weaker ref.
        ``strength <= 0`` drops the block entirely (no tokens injected).
        """
        if strength <= 0.0:
            return None
        latent = self.latent
        if strength < 1.0:
            seed = zlib.crc32(self.name.encode("utf-8")) ^ int(strength * 1e4)
            gen = torch.Generator().manual_seed(seed)
            noise = torch.randn(latent.shape, generator=gen, dtype=latent.dtype)
            latent = strength * latent + (1.0 - strength) * noise
        block: Dict = {
            "kind": self.kind,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "latent": latent,
        }
        if self.kind == "video":
            block["latent_t"] = self.latent_t
            block["ref_audio_t"] = 0
            block["audio_latent"] = None
        return block

    # ── serialization ─────────────────────────────────────────────────

    def save(self, path_no_ext: str) -> str:
        """Save as a single ``{path}.safetensors`` with metadata in the header."""
        os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
        meta = {
            "name": self.name,
            "kind": self.kind,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "latent_t": self.latent_t,
            "mode": self.mode,
            "source": self.source,
            "source_shape": self.source_shape,
            "pool": self.pool,
            "optimize_steps": self.optimize_steps,
            "tags": self.tags,
            "_format_version": 2,
        }
        save_file({"latent": self.latent.contiguous()}, path_no_ext + ".safetensors",
                  metadata={META_KEY: json.dumps(meta)})
        return path_no_ext + ".safetensors"

    @classmethod
    def load(cls, path_no_ext: str, device: str = "cpu") -> "H3RefMod":
        """Load from ``{path}.safetensors`` (metadata in header, or legacy .json)."""
        meta = read_refmod_meta(path_no_ext)
        if meta is None:
            raise ValueError(
                f"{path_no_ext}.safetensors has no RefMod metadata "
                f"(header key '{META_KEY}' or sidecar .json missing).")
        # clone drops the file mmap, so the file isn't locked on Windows and
        # can be re-saved over the same name
        latent = load_file(path_no_ext + ".safetensors", device=device)["latent"].clone()
        return cls(
            name=meta.get("name", os.path.basename(path_no_ext)),
            kind=meta.get("kind", "image"),
            latent=latent,
            latent_h=int(meta.get("latent_h", latent.shape[3])),
            latent_w=int(meta.get("latent_w", latent.shape[4])),
            latent_t=int(meta.get("latent_t", latent.shape[2])),
            mode=meta.get("mode", "pooled"),
            source=meta.get("source", ""),
            source_shape=meta.get("source_shape", ""),
            pool=meta.get("pool", ""),
            optimize_steps=int(meta.get("optimize_steps", 0)),
            tags=list(meta.get("tags", [])),
        )

