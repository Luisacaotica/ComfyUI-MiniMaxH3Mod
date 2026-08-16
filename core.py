"""
core.py — "RefMod" adapter for MiniMax H3 (no training required)

The reference-video path of MiniMax H3 works by injecting *reference tokens*
into the packed sequence: the ref is VAE-encoded, patchified, projected and
placed on the 3D RoPE grid, then every DiT block attends to it.  A video ref
is expensive because it contributes thousands of tokens.

Mode naming: ``encode`` (was ``full``) = straight full-res VAE encode;
``training`` (was ``pooled``) = compressed grid refined by gradient steps
(the only mode that "trains" — the refinement loop).  The old names are
accepted everywhere as legacy aliases so saved workflows and mods keep
working.

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
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Metadata key embedded in the safetensors header (keeps a mod in one file,
# so it can be shared/uploaded as a single artifact).
META_KEY = "refmod_meta"

# What a mod is standing in for. This is documentation + a routing hint, not
# a hard constraint on the tensors — but it drives two things: (1) the
# "identity in training mode" warning in nodes.py, and (2) the loader's
# prompt_hint output, which merges each loaded mod's concept_type +
# description into a string you can concat onto your CLIP Text Encode prompt.
# There's no CLIP-Vision / image-embedding injection point on H3's ref2va
# path to hang a "visual clue" off of — text is the only channel the model's
# text encoder reads, so that's the channel this uses.
CONCEPT_TYPES = (
    "generic",       # unspecified / mixed
    "identity",      # a specific person/character — use mode="encode" for this
    "pose_motion",   # a pose, dance, gesture, camera move
    "clothing",      # an outfit / garment, decoupled from who's wearing it
    "background",    # environment / set / location plate
    "style",         # look/grade/animation style, not a concrete subject
)


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass (downsample then upsample) used as the
    weakening target for ``ref_block``'s ``strength`` below ``1.0``.

    Originally this mixed toward ``torch.randn()`` noise, mirroring what the
    docstring below describes as the model's own ``visual_cond_noise_aug``
    mechanism. In practice, at the retention range people actually use this
    dial for (0.15-0.7, i.e. large noise fractions), it produces a visible
    woven/static texture instead of a gracefully weaker reference: a VAE
    latent's channels are correlated, so raw iid noise isn't read as "less
    reference," it's read as real-but-garbled content and gets rendered as
    an actual (wrong) texture. ``visual_cond_noise_aug`` in the model's own
    training is likely a small-magnitude robustness augmentation, not
    something swept this far — so copying its math doesn't reproduce its
    behavior at these strengths. A blurred copy stays on the latent
    manifold (smooth, plausible) while still discarding detail as strength
    drops, which is what "weaker reference" should actually look like.
    """
    if z.dim() != 5:
        return z
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


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

MODE_ALIASES = {"full": "encode", "pooled": "training"}


def normalize_mode(mode: str) -> str:
    """Map legacy mode names (``full``/``pooled``) to the current ones."""
    return MODE_ALIASES.get(mode, mode)


def aspect_grid(pool_h: int, pool_w: int, aspect: float):
    """Even pool grid dims that match ``aspect`` (h/w) within the dial caps.

    The DiT patches each 2x2 latent cell into one token and the rope grid is
    area-normalized per axis, so a pooled grid that ignores the source aspect
    squishes the subject: a portrait latent (tall person) forced into a square
    16x16 grid is compressed 5x in height but only 3x in width, and comes out
    "fat".  The official ref2video node never pools — it feeds the model
    aspect-correct latent dims, which is what the DiT was trained on.

    The long edge follows the user's dial (``max(pool_h, pool_w)``) and the
    short edge is derived from the source aspect, both rounded to even for the
    2x2 patch.  Square sources keep the exact dial value (16x16 stays 16x16).
    """
    long_edge = max(pool_h, pool_w)
    if aspect >= 1.0:
        h, w = long_edge, long_edge / aspect
    else:
        w, h = long_edge, long_edge * aspect
    h = max(2, round(h / 2) * 2)
    w = max(2, round(w / 2) * 2)
    return h, w


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
    progress_every: int = 0,
) -> torch.Tensor:
    """
    Model-free refinement of the compressed latent.

    Optimizes the small latent so its trilinearly upsampled reconstruction
    matches the full reference latent.  This pulls the pooled representation
    closer to what the DiT would see from the full ref, with nothing but the
    tiny latent trainable (~1-2K params) and no diffusion model loaded.

    ``progress_every`` > 0 prints a ``[RefMod] identity <step>/<steps>`` line
    every N steps so long pooled-mode refinement isn't silent (the default
    ``identity=500`` takes a while per ref on CPU/VRAM-bound setups).

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
        for i in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=size, mode="trilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[RefMod] identity {i + 1}/{steps}")
        # materialize inside the scope so the result is a normal tensor, not
        # an inference-mode tensor (it gets stored in the mod and reused)
        refined = param.detach().to(z_small.dtype)
    return refined


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame strength curve
# ═══════════════════════════════════════════════════════════════════════════

# Direction is where the envelope points (its endpoints); shape is how it
# travels between them; value is the non-zero endpoint ("user input").
CURVE_DIRECTIONS = ("constant", "increase", "decrease")
CURVE_SHAPES = ("linear", "ease", "quadratic", "cubic", "exponential",
                "stair", "elastic", "bump", "dip")

# old single-combo preset names -> (direction, shape, value) so workflows
# saved before the curve was split still resolve
_LEGACY_CURVES = {
    "flat": ("constant", "linear", 1.0),
    "fade_in": ("increase", "linear", 1.0),
    "fade_out": ("decrease", "linear", 1.0),
    "bump": ("increase", "bump", 1.0),
    "dip": ("constant", "dip", 1.0),
}


def _ease(shape: str, x: float) -> float:
    """Shape a progress ``x`` in [0, 1] per the easing name (may overshoot)."""
    if shape == "linear":
        return x
    if shape == "ease":  # smoothstep
        return x * x * (3.0 - 2.0 * x)
    if shape == "quadratic":
        return x * x
    if shape == "cubic":
        return x * x * x
    if shape == "exponential":
        return 2.0 ** x - 1.0
    if shape == "stair":
        return min(1.0, math.floor(x * 4) / 3.0)
    if shape == "elastic":
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "dip":
        return abs(2.0 * x - 1.0)
    return x


def curve_strengths(spec, t: int) -> Optional[List[float]]:
    """Resolve a curve spec to ``t`` per-frame strength multipliers in [0, 1].

    Kept as plain Python so the Apply node exposes the curve as normal
    combo/float widgets instead of depending on ComfyUI's new Curve widget.
    Accepts:

      * a ``(direction, shape, value)`` tuple — direction is ``constant``
        (one strength everywhere), ``increase`` (0 -> value, a crescent) or
        ``decrease`` (value -> 0, a decrescent); shape is how the envelope
        travels between its endpoints (see ``_ease``); value is the non-zero
        endpoint (1.0 = full strength there);
      * a legacy preset name (``flat``/``fade_in``/``fade_out``/``bump``/
        ``dip``) from before the split;
      * a list of per-frame floats (len == t), used as-is;
      * a list of ``(x, y)`` control points (x in [0, 1]), linearly
        interpolated across the frames;
      * any object with ``interp(x) -> float`` (legacy CurveInput saved in an
        old workflow).

    Returns None for flat/no curve so the caller keeps its single-strength
    path unchanged (a ``constant`` curve at value 1.0 included).
    """
    if t <= 1 or spec is None or spec == "":
        return None
    if isinstance(spec, str):
        legacy = _LEGACY_CURVES.get(spec)
        return curve_strengths(legacy, t) if legacy is not None else None
    if (isinstance(spec, tuple) and len(spec) == 3
            and isinstance(spec[0], str) and isinstance(spec[1], str)):
        direction, shape, value = spec
        value = float(value)
        if direction == "constant":
            # linear = flat at ``value`` (1.0 == no curve); the other shapes
            # modulate around the constant, so constant + dip is a trough,
            # constant + bump is a peak (both endpoints equal)
            if shape == "linear":
                return None if value >= 1.0 else [value] * t
            p = [_ease(shape, i / (t - 1)) for i in range(t)]
            return [max(0.0, min(1.0, value * y)) for y in p]
        p = [_ease(shape, i / (t - 1)) for i in range(t)]
        if direction == "increase":
            return [max(0.0, min(1.0, value * y)) for y in p]
        if direction == "decrease":
            return [max(0.0, min(1.0, value * (1.0 - y))) for y in p]
        return None  # unknown direction: treat as flat
    if isinstance(spec, (list, tuple)):
        if len(spec) == t and all(isinstance(v, (int, float)) for v in spec):
            return [float(v) for v in spec]
        pts = [(float(x), float(y)) for x, y in spec
               if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        if pts:
            pts = sorted(pts, key=lambda p: p[0])
            out = []
            for i in range(t):
                x = i * (1.0 / (t - 1))
                if x <= pts[0][0]:
                    out.append(pts[0][1])
                elif x >= pts[-1][0]:
                    out.append(pts[-1][1])
                else:
                    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                        if x0 <= x <= x1:
                            out.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
                            break
            return out
    interp = getattr(spec, "interp", None)
    if callable(interp):
        return [float(interp(i / (t - 1))) for i in range(t)]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Token budget cap (shared by the Extract node and extract_mod.py)
# ═══════════════════════════════════════════════════════════════════════════


def dedup_frame_indices(z: torch.Tensor, threshold: float = 0.02) -> List[int]:
    """Indices of latent frames kept by greedy temporal dedup.

    Each frame is compared to the last *kept* frame: if the mean-abs
    difference between them, normalized by the frames' own magnitude, is
    below ``threshold`` the frame is dropped as a (near-)duplicate.  Video
    refs (a dance loop, a static shot, a talking head) contain long runs of
    frames that differ only by codec noise — each one still costs a token
    per spatial patch in every DiT block, so cutting them is the cheapest
    way to honor a token cap.  Stacked image refs survive: different
    angles/expressions land well above the threshold.
    """
    t = z.shape[2]
    if t <= 1:
        return list(range(t))
    flat = z[0].float()  # [24, T, H, W]
    kept = [0]
    prev = flat[:, 0]
    for i in range(1, t):
        cur = flat[:, i]
        denom = (cur.abs().mean() + prev.abs().mean()) / 2 + 1e-6
        diff = (cur - prev).abs().mean() / denom
        if diff >= threshold:
            kept.append(i)
            prev = cur
    return kept


def fit_token_budget(latent: torch.Tensor, budget: int, label: str) -> torch.Tensor:
    """Bring a stacked latent's injected token count under ``budget``.

    Tokens = (H/2)*(W/2) per frame (the DiT's 2x2 patch).  When the stack
    exceeds the budget it is cut in two cheap, loss-ordered passes:

      1. temporal dedup — drop latent frames that are (near-)identical to
         the last kept frame (free: they carry no new reference info);
      2. budget-fit resample — uniformly subsample the remaining frames down
         to the largest count that fits the budget.

    The spatial dims are never touched: resizing a latent's H/W changes the
    rope grid and degrades the reference, so time is the only honest lever.
    ``label`` is the mod name for the console notes.
    """
    h, w = latent.shape[3], latent.shape[4]
    per_frame = (h // 2) * (w // 2)
    t = latent.shape[2]
    if per_frame * t <= budget:
        return latent
    kept = dedup_frame_indices(latent)
    if len(kept) < t:
        print(f"[RefMod] {label}: over {budget}-token cap, "
              f"dropped {t - len(kept)} near-duplicate frame(s) "
              f"({t} -> {len(kept)})")
        latent = latent[:, :, kept]
        t = latent.shape[2]
    if per_frame * t > budget:
        fit_t = max(1, budget // per_frame)
        if fit_t < t:
            idx = torch.linspace(0, t - 1, fit_t).round().long()
            latent = latent[:, :, idx]
            print(f"[RefMod] {label}: still over {budget}-token cap, "
                  f"resampled {t} -> {fit_t} frame(s) to fit")
        if per_frame * latent.shape[2] > budget:
            print(f"[RefMod] {label}: one frame alone is {per_frame} tokens > "
                  f"{budget} — lower ref_resolution or pool_h/pool_w to meet "
                  f"the cap.")
    return latent


# ═══════════════════════════════════════════════════════════════════════════
# H3RefMod — the saved artifact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class H3RefMod:
    """
    A compressed reference for MiniMax H3.

    ``latent`` is the VAE latent ``[1, 24, latent_t, latent_h, latent_w]`` — a
    full-resolution encode (``mode="encode"``) or a pooled thumbnail
    (``mode="training"``).  ``kind`` is ``"image"`` (single latent frame) or
    ``"video"`` (a few frames), matching the native ref block kinds the
    model's ``PackedLayout`` understands.

    ``mode="encode"`` stores the encode at the resolution the official ref2video
    path uses, so the injected ref carries real identity detail;
    ``mode="training"`` stores a tiny average-pooled grid refined by gradient
    steps (concept/motion, or identity at high pool sizes).
    """

    name: str
    kind: str
    latent: torch.Tensor
    latent_h: int = 4
    latent_w: int = 4
    latent_t: int = 1
    mode: str = "training"
    source: str = ""          # "image" | "video" | "manual"
    source_shape: str = ""    # original latent dims as "TxHxW"
    pool: str = "4x4x1"       # pool_t x pool_h x pool_w
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""     # optional text describing the concept (emitted by the loaders)
    concept_type: str = "generic"  # what this mod represents; see CONCEPT_TYPES above

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

    def ref_block(self, strength: float = 1.0,
                  curve=None) -> Optional[Dict]:
        """
        Build the ref block dict the model's ``PackedLayout`` / payload consumes.

        The shape mirrors what the native ref2va nodes emit, so the pooled
        latent rides the exact same path: ``cond_video_latents`` -> patchify ->
        ``video_patch_proj`` -> packed sequence with a 3D RoPE grid.

        ``strength`` weakens the ref by mixing the latent toward a heavily
        blurred copy of itself (see ``_blur_latent``) rather than toward
        zero or random noise: scaling toward zero pushes values out of the
        normalized latent distribution (reads as grey output), and mixing
        toward random noise reads as real-but-garbled content and decodes
        as a wrong texture rather than "weaker" — a blurred copy stays
        on-manifold while still discarding the specific detail that makes a
        reference strong. ``strength <= 0`` drops the block entirely (no
        tokens injected).

        ``curve`` (optional) is a per-frame strength spec for the ref's
        latent timeline — a ``(direction, shape, value)`` tuple (e.g.
        ``("decrease", "ease", 1.0)``), a legacy preset name, a list of
        per-frame floats, ``(x, y)`` control points, or any object with
        ``interp(x)`` (see ``curve_strengths``).  Each frame is mixed with
        its own strength ``retention * curve(x)`` instead of one flat value,
        so the ref can fade in/out across the video.  A flat curve is
        identical to passing ``strength`` alone.
        """
        if strength <= 0.0:
            return None
        latent = self.latent
        if curve is not None and self.latent_t > 1:
            strengths = curve_strengths(curve, self.latent_t)
            if strengths is not None:
                t = self.latent_t
                st = torch.tensor(
                    [max(0.0, min(1.0, strength * s)) for s in strengths],
                    dtype=latent.dtype, device=latent.device)
                st = st.view(1, 1, t, 1, 1)
                latent = st * latent + (1.0 - st) * _blur_latent(latent)
        elif strength < 1.0:
            latent = strength * latent + (1.0 - strength) * _blur_latent(latent)
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
            "description": self.description,
            "concept_type": self.concept_type,
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
            mode=normalize_mode(meta.get("mode", "training")),
            source=meta.get("source", ""),
            source_shape=meta.get("source_shape", ""),
            pool=meta.get("pool", ""),
            optimize_steps=int(meta.get("optimize_steps", 0)),
            tags=list(meta.get("tags", [])),
            description=str(meta.get("description", "") or ""),
            concept_type=str(meta.get("concept_type", "generic") or "generic"),
        )

