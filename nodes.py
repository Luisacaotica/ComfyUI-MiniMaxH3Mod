"""
nodes.py — ComfyUI nodes for MiniMax H3 "RefMod" (no-training reference mods)

  MiniMaxH3RefModExtract       — Autogrow reference inputs (image or video frames)
                                 -> a saved mod, output as an H3_REF_MODS bundle
  MiniMaxH3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  MiniMaxH3RefModsLoader       — load 1-8 mods with a typed strength each (LoRA-style)
  MiniMaxH3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)
  MiniMaxH3RefModApply         — inject the bundle into a MINIMAX_H3_COND conditioning or the
                                 built-in ComfyUI CONDITIONING (one node, old ApplyCond
                                 workflows auto-migrate via node replacement)

Mods are stored in ``models/refmods/`` (created on first run, next to loras/
and unet/); mods saved by older versions in the pack's ``mods/`` folder still
load.

The mod rides the model's native ref2va path: the Apply nodes append reference
blocks (the mod latents) to the conditioning's ``refs``, and the DiT attends
to those tokens through all of its blocks, exactly like a full image/video
reference but at a fraction of the token budget.

Reference strength uses the model's own conditioning-strength dial, but
weakening a ref mixes its latent toward a heavily blurred copy of itself
(not toward noise or toward zero — see core.py's ``_blur_latent``/
``ref_block`` for why). ``retention`` on the Apply nodes is a preset master
strength (fully_preserved / partially_preserved / attribute_transfer /
weak_reference) multiplied with each loader row's strength.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .common import (
    list_media_files,
    load_image_file,
    load_video_file,
    refmods_dir,
)
from .core import (
    CONCEPT_TYPES,
    CURVE_DIRECTIONS,
    CURVE_SHAPES,
    H3RefMod,
    _blur_latent,
    aspect_grid,
    curve_value_at,
    fit_token_budget,
    normalize_mode,
    optimize_latent,
    pool_latent,
    read_refmod_meta,
)
from .debug_grid import (
    graph_pnginfo,
    pil_to_tensor,
    read_graph_meta,
    render_debug_grid,
)

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODS_DIR = os.path.join(_PACK_DIR, "mods")  # pre-models/refmods storage, still read
_MOD_CACHE: Dict[str, H3RefMod] = {}
_MOD_CACHE_MAX = 24          # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None   # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
_MOD_LIST_CACHE_VAL = None

# Mod storage lives in ComfyUI's models/ tree (created on first run) and is
# registered as a first-class folder type so it shows up next to loras/unet.
try:
    folder_paths.add_model_folder_path("refmods", refmods_dir())
except Exception:
    pass

# reference retention presets (master strength multiplier on Apply)
RETENTION = {
    "fully_preserved": 1.0,
    "partially_preserved": 0.7,
    "attribute_transfer": 0.4,
    "weak_reference": 0.15,
}


# ═══════════════════════════════════════════════════════════════════════════
# ComfyUI-MiniMaxH3 pack integration
# ═══════════════════════════════════════════════════════════════════════════

def _pack_dir() -> str:
    custom_nodes = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(custom_nodes, "ComfyUI-MiniMaxH3")


def _h3_pack_submodule(subpath: str):
    """
    Import a submodule of the ComfyUI-MiniMaxH3 pack.

    ComfyUI registers custom node folders in sys.modules under their absolute
    path with dots replaced by ``_x_``, so the normal import statement can't
    reference it.  Prefer the already-loaded instance (shared VAE caches);
    fall back to loading the pack under a clean name if it hasn't loaded yet.
    """
    pack_dir = os.path.abspath(_pack_dir())
    if not os.path.isdir(pack_dir):
        raise RuntimeError(
            "ComfyUI-MiniMaxH3 pack not found at " + pack_dir + ". "
            "Install it first (ComfyUI Manager: search 'MiniMax H3', or git "
            "clone https://github.com/xiaolibai-sys/ComfyUI-MiniMaxH3 into "
            "custom_nodes/) — it is required for the av_encoder input on "
            "Extract H3 RefMod and the pack-conditioning Apply H3 RefMod node."
        )
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None) or getattr(mod, "__path__", None)
        if path is None:
            continue
        try:
            root = os.path.abspath(path if isinstance(path, str) else path[0])
        except Exception:
            continue
        if root.startswith(pack_dir + os.sep) or root == pack_dir:
            try:
                return importlib.import_module(name + "." + subpath)
            except ImportError:
                pass
    module_name = "ComfyUI_MiniMaxH3"
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(pack_dir, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(module_name + "." + subpath)


# ═══════════════════════════════════════════════════════════════════════════
# Mods folder helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mod_search_dirs() -> List[str]:
    dirs = [refmods_dir()]
    root_models_mods = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "mods")
    for d in (root_models_mods, LEGACY_MODS_DIR):
        if d not in dirs and os.path.isdir(d):
            dirs.append(d)
    return dirs


def _list_mod_names() -> List[str]:
    """Available RefMod names across the search dirs (for the loader dropdown).

    Only entries with valid RefMod metadata (embedded in the safetensors header
    or a legacy sidecar .json) are listed, so other mod formats in
    models/mods/ (e.g. LTXMod files) don't show up.

    Called by INPUT_TYPES/VALIDATE_INPUTS on every prompt validation, so the
    result is cached until any mod file appears/disappears/changes (checked
    via cheap os.stat, not by re-reading every safetensors header).
    """
    global _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL
    sig = []
    for d in _mod_search_dirs():
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".safetensors"):
                continue
            try:
                st = os.stat(os.path.join(d, fn))
                sig.append(f"{fn}:{st.st_size}:{int(st.st_mtime)}")
            except OSError:
                pass
    key = "\n".join(sig)
    if key == _MOD_LIST_CACHE_KEY:
        return _MOD_LIST_CACHE_VAL
    names = set()
    for d in _mod_search_dirs():
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".safetensors"):
                continue
            stem = fn[:-len(".safetensors")]
            meta = read_refmod_meta(os.path.join(d, stem))
            if meta is not None and meta.get("kind") in ("image", "video"):
                names.add(stem)
    _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL = key, sorted(names)
    return _MOD_LIST_CACHE_VAL


def _find_mod_path(name: str) -> str:
    for d in _mod_search_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p + ".safetensors"):
            return p
    raise FileNotFoundError(
        f"RefMod '{name}' not found. Searched:\n" +
        "\n".join(f"  - {d}/{name}.safetensors" for d in _mod_search_dirs()))


def _load_mod(name: str) -> H3RefMod:
    if name in _MOD_CACHE:
        return _MOD_CACHE[name]
    mod = H3RefMod.load(_find_mod_path(name), device="cpu")
    _MOD_CACHE[name] = mod
    if len(_MOD_CACHE) > _MOD_CACHE_MAX:
        # FIFO eviction: pop the oldest-loaded mod so a long session loading
        # many different mods doesn't accumulate every one of them in RAM
        _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# Graph presets (shared curve files, next to the mods)
# ═══════════════════════════════════════════════════════════════════════════

def _graph_presets_dir() -> str:
    """models/refmods/graph_presets — shared curve presets, created on first use."""
    d = os.path.join(refmods_dir(), "graph_presets")
    os.makedirs(d, exist_ok=True)
    return d


def _list_graph_presets() -> List[str]:
    """Graph preset names in the presets folder, for the dropdown.

    Presets are PNGs with the graph embedded in their tEXt metadata (a saved
    debug grid); legacy .json files from before the switch still list.  PNGs
    without a valid graph chunk are skipped so random images dropped in the
    folder don't show up.
    """
    d = _graph_presets_dir()
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return []
    names = []
    for fn in entries:
        if fn.endswith(".json"):
            names.append(fn[:-5])
        elif fn.endswith(".png") and read_graph_meta(os.path.join(d, fn)) is not None:
            names.append(fn[:-4])
    return names


def _load_graph_preset(name: str) -> Optional[tuple]:
    """Read a graph preset -> (direction, shape, value) or None if invalid.

    Presets are PNG files with the graph embedded in their tEXt metadata (the
    saved debug grid — share the image itself); legacy .json presets still
    load.
    """
    d = _graph_presets_dir()
    meta = read_graph_meta(os.path.join(d, name + ".png"))
    if meta is not None:
        return meta
    try:
        with open(os.path.join(d, name + ".json"), "r",
                  encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    direction, shape = data.get("direction"), data.get("shape")
    if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
        return None
    try:
        value = float(data.get("value", 1.0))
    except (TypeError, ValueError):
        return None
    return (direction, shape, value)


def _save_graph_preset(name: str, spec, img=None) -> str:
    """Write a (direction, shape, value) tuple as a PNG preset with tEXt meta.

    The saved file is the debug grid itself (a mini preview of the curve)
    with the graph embedded in its metadata, so sharing the image shares the
    curve.  ``img`` is the rendered grid from Apply; when absent a minimal
    grid is rendered just for the file.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                    for c in str(name).strip())
    if not safe:
        return ""
    if img is None:
        img = render_debug_grid(spec)
    img.save(os.path.join(_graph_presets_dir(), safe + ".png"),
             pnginfo=graph_pnginfo(spec))
    return safe


def _resize_ref(image, short_edge: int, canvas=None):
    """Aspect-preserving downscale (never upscale) to ``short_edge`` px; dims to /32.

    When several refs are stacked into one mod they must share a single spatial
    canvas, so ``canvas`` (tw, th) cover-crops each ref to it (like the official
    node's follower keyframes).  Mirrors the official ref2video node: refs are
    resized before VAE encode, so the stored latent rides the same
    full-resolution path the model was trained with (the pooled path below is
    the cheap "thumbnail" alternative).
    """
    h, w = image.shape[1], image.shape[2]
    if h <= 0 or w <= 0:
        raise ValueError(
            f"_resize_ref: source has an empty frame ({h}x{w}) before any "
            f"resize — the reference itself is invalid.")
    scale = min(1.0, short_edge / min(h, w))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    crop = "disabled"
    if canvas is not None:
        tw, th = canvas
        crop = "center"
    if tw <= 0 or th <= 0:
        raise ValueError(
            f"_resize_ref: computed a zero-size resize target ({tw}x{th}) "
            f"for a {h}x{w} source (short_edge={short_edge}, canvas={canvas}). "
            f"This should be impossible — please report this shape.")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", crop)
    if samples.shape[2] <= 0 or samples.shape[3] <= 0:
        raise ValueError(
            f"_resize_ref: common_upscale produced an empty result "
            f"{tuple(samples.shape)} from a {h}x{w} source targeting "
            f"{tw}x{th} (crop={crop}, canvas={canvas}). This points to a bug "
            f"in comfy.utils.common_upscale for this input, not in RefMod's "
            f"own math.")
    return samples.movedim(1, -1)


def _snap_to_causal_grid(n_frames: int) -> int:
    """Round a video frame count down to the nearest valid ``4k + 1``.

    MiniMax H3's video VAE is causal: it compresses time in groups of 4 with
    one leading keyframe, so it only accepts pixel-frame counts of the form
    4k+1 (1, 5, 9, 13, 17, ...). Anything else makes its internal temporal
    chunker produce a zero-length chunk list and crash on
    ``torch.cat(): expected a non-empty list of Tensors``. The official
    ref2video path already trims to this grid before encoding; RefMod
    extraction previously didn't, so an arbitrary frame_load_cap/
    select_every_nth combo from a video loader would break it.
    """
    if n_frames <= 1:
        return 1
    return ((n_frames - 1) // 4) * 4 + 1


def _ensure_min_size(image, floor: int = 320):
    """Upscale (never downscale) so both spatial dims are >= ``floor`` px.

    The MiniMax H3 VAE encodes with internal tiled_encode (~256px tiles). A
    reference smaller than the tile size in one dimension can make the tiler
    compute a zero-size edge tile, which crashes deep inside conv_in with a
    cryptic 'Expected 4D or 5D... but got [1,3,1,0,W]' error. This applies
    regardless of extraction mode ('encode' already resizes down to
    ref_resolution but never guarantees a floor; 'training' now resizes to
    the same cap, also without a floor), so it's a separate, unconditional
    safety net right before encode.
    """
    import comfy.utils
    h, w = image.shape[1], image.shape[2]
    if h >= floor and w >= floor:
        return image
    scale = floor / min(h, w)
    tw = max(floor, round(w * scale / 32) * 32)
    th = max(floor, round(h * scale / 32) * 32)
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _normalize_mask_batch(mask, label: str = "mask") -> torch.Tensor:
    """Canonicalize a MASK input to ``[N, H, W]`` float32 in [0, 1]."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"MiniMaxH3RefModExtract: {label} must be a MASK tensor, "
                          f"got {type(mask)}")
    if mask.dim() == 2:  # [H, W]
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"MiniMaxH3RefModExtract: {label} has unexpected shape "
                          f"{tuple(mask.shape)} (expected [H,W] or [N,H,W])")
    return mask.float().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a ``[T, H, W]`` mask to ``target_h x target_w`` (bilinear)."""
    samples = mask.unsqueeze(1)  # [T, 1, H, W]
    samples = comfy.utils.common_upscale(samples, target_w, target_h, "bilinear", "disabled")
    return samples.squeeze(1).clamp(0.0, 1.0)


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass: downsample then upsample back.

    Used as the suppression target instead of random noise. A VAE latent's
    channels are correlated (it's not iid per-pixel noise in this space), so
    feeding the model raw torch.randn() as a "suppressed" reference isn't
    read as absence — it's read as real, garbled content, and gets rendered
    as an actual (wrong) texture: the woven/static pattern is what
    out-of-distribution noise looks like once a diffusion model tries to
    make sense of it as a reference. A blurred copy of the real latent stays
    on the manifold (smooth, plausible) while discarding the specific
    structure (a skyline, a treeline) that was dictating unwanted content.
    """
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up


def _mask_latent(z: torch.Tensor, mask_px: torch.Tensor, background_retention: float,
                  seed_key: str) -> torch.Tensor:
    """Suppress the latent outside ``mask_px`` toward a blurred copy of itself, per cell.

    ``mask_px`` is pixel-space (already resized/cropped to match the encoded
    source), 1 = keep, 0 = suppress; ``background_retention`` sets the floor
    weight for suppressed regions (0 = fully blurred there, 1 = no
    suppression at all). ``seed_key`` is unused now (kept for call-site
    compatibility) — the suppression target is deterministic, not random.

    ``z``: ``[1, 24, T, H, W]`` VAE latent. Downsamples ``mask_px`` to the
    latent's ``H x W`` via average pooling (soft edges instead of a hard cut,
    since the DiT patchifies in 2x2 cells anyway).
    """
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    mp = mask_px.unsqueeze(1)  # [T_src, 1, H, W]
    if mp.shape[0] == 1 and t > 1:
        mp = mp.expand(t, -1, -1, -1)
    elif mp.shape[0] != t:
        idx = torch.linspace(0, mp.shape[0] - 1, t).round().long()
        mp = mp[idx]
    mp = F.adaptive_avg_pool2d(mp.float(), (h, w))          # [T, 1, h, w]
    mp = mp.permute(1, 0, 2, 3).unsqueeze(0).clamp(0.0, 1.0)  # [1, 1, T, h, w]
    weight = background_retention + (1.0 - background_retention) * mp
    blurred = _blur_latent(z)
    return (weight * z.float() + (1.0 - weight) * blurred).to(z.dtype)


def _normalize_ref(src, label: str = "reference") -> torch.Tensor:
    """Canonicalize any ref source to ``[T, H, W, C]`` (T=1 for stills).

    Accepts ``[H, W, C]``, ``[B, H, W, C]``, and batch-video ``[B, T, H, W, C]``
    (some video loaders emit the batch form).  Rejects empty frames with a
    clear error instead of letting the VAE crash on a zero spatial dim.
    """
    if not isinstance(src, torch.Tensor) or src.dim() not in (3, 4, 5):
        raise ValueError(
            f"MiniMaxH3RefModExtract: {label} must be a 3-5D tensor, "
            f"got {getattr(src, 'shape', src)}")
    if src.dim() == 5:  # [B, T, H, W, C] batch video
        if src.shape[0] == 0:
            raise ValueError(
                f"MiniMaxH3RefModExtract: {label} has no frames "
                f"(T=0) — check the source image/video.")
        src = src[0] if src.shape[0] == 1 else src.reshape(-1, *src.shape[2:])
    if src.dim() == 3:  # [H, W, C]
        src = src.unsqueeze(0)
    if src.shape[-1] != 3 and src.shape[1] == 3:  # channel-first [B, C, H, W]
        src = src.movedim(1, -1)
    if src.dim() != 4 or src.shape[-1] != 3:
        raise ValueError(
            f"MiniMaxH3RefModExtract: {label} has an unexpected layout "
            f"{tuple(src.shape)} (expected [T, H, W, 3])")
    if src.shape[0] <= 0:
        raise ValueError(
            f"MiniMaxH3RefModExtract: {label} has no frames (T={src.shape[0]}) "
            f"— check the source image/video.")
    if src.shape[1] <= 0 or src.shape[2] <= 0:
        raise ValueError(
            f"MiniMaxH3RefModExtract: {label} has an empty frame "
            f"({src.shape[1]}x{src.shape[2]}) — check the source image/video.")
    return src


def _sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    if not name:
        raise ValueError("mod name must not be empty")
    return name


def _resolve_folder(folder: str) -> str:
    """Resolve a folder input: absolute path, a name inside input/, or input/ itself."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return folder_paths.get_input_directory()
    if os.path.isabs(folder):
        resolved = os.path.normpath(folder)
    else:
        resolved = os.path.join(folder_paths.get_input_directory(), folder)
    if not os.path.isdir(resolved):
        raise ValueError(
            f"folder not found: {folder!r} (looked at '{resolved}'; use an "
            "absolute path or a folder name inside input/).")
    return resolved


def _summarize(mod: H3RefMod) -> str:
    mb = mod.latent.numel() * mod.latent.element_size() / 1024 / 1024
    return (f"'{mod.name}' {mod.mode} {mod.kind} {tuple(mod.latent.shape)} "
            f"({mod.token_count} tokens, {mb:.2f} MB)")


def _info_lines(mod: H3RefMod) -> List[str]:
    opt = mod.optimize_steps
    if mod.mode == "encode":
        opt = f"n/a ({mod.optimize_steps} — encode mode stores the actual encode)"
    return [
        "=" * 52,
        f"  MiniMax H3 RefMod: {mod.name}",
        f"  {'concept_type':<18} {mod.concept_type}",
        f"  {'mode':<18} {mod.mode}",
        f"  {'kind':<18} {mod.kind}",
        f"  {'latent':<18} {tuple(mod.latent.shape)}",
        f"  {'tokens injected':<18} {mod.token_count}",
        f"  {'source':<18} {mod.source} ({mod.source_shape})",
        f"  {'pool':<18} {mod.pool}",
        f"  {'identity':<18} {opt}",
        f"  {'tags':<18} {', '.join(mod.tags) if mod.tags else '-'}",
        f"  {'description':<18} {mod.description or '-'}",
        "=" * 52,
    ]


def _ref_blocks(mods, retention, curve=None, seed=-1) -> List[Dict]:
    """Ref blocks for a loader bundle, scaled by row strength x retention.

    ``retention`` is a master strength multiplier: a float 0-1 (1.0 =
    fully_preserved, 0.7 = partially_preserved, 0.4 = attribute_transfer,
    0.15 = weak_reference), or one of those preset names for legacy
    workflows saved with the old combo widget.

    ``curve`` (optional) is a per-frame strength spec — a ``(direction,
    shape, value)`` tuple, a legacy preset name, per-frame values, control
    points (see ``core.curve_strengths``) — applied on top of the row
    strength.  A flat/no curve keeps today's behavior.

    ``seed`` (default -1 = off) enables ref scrambling: with 2+ refs in the
    bundle, the order is shuffled and a random subset kept, so a different
    ref leads each run instead of the same one always "popping".  Same seed
    -> same scramble; connect/randomize the seed for per-run variation.
    """
    if isinstance(retention, str):
        factor = RETENTION.get(retention, 1.0)
    else:
        factor = float(retention)
    items = list(mods)
    if int(seed) >= 0 and len(items) > 1:
        rng = random.Random(int(seed))
        rng.shuffle(items)
        keep = rng.randint(max(1, len(items) // 2), len(items))
        items = items[:keep]
        print(f"[MiniMaxH3RefModApply] scramble seed={int(seed)}: "
              f"{len(mods)} refs -> kept {len(items)} (order shuffled)")
    blocks = []
    for mod, strength in items:
        eff = min(1.0, max(0.0, strength * factor))
        block = mod.ref_block(eff, curve=curve)
        if block is not None:
            blocks.append(block)
    return blocks


def _make_step_wrapper(spec) -> Callable:
    """DIFFUSION_MODEL wrapper re-mixing every ref latent per denoising step.

    The Apply node's frame curve is baked into the ref latents once, before
    sampling.  This wrapper re-scales them per step instead: it caches the
    pristine latents (and blurred copies) on the first forward, then each
    step mixes toward the blur with ``curve_value_at(spec, 1 - sigma)`` — the
    same direction/shape/value envelope as the frame curve, running over the
    denoise timeline (0 = first step, high sigma) instead of the video's.
    ``cond_video_latents`` is re-read from the payload every forward, so
    replacing the list per step is all it takes; the packed layout stays
    valid because only values change, never shapes.
    """
    state = {"pristine": None, "blurred": None}

    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        cond = payload.get("cond_video_latents")
        if cond:
            if state["pristine"] is None:
                state["pristine"] = [z.clone() for z in cond]
                state["blurred"] = [_blur_latent(z) for z in state["pristine"]]
            sigma = float((timestep.flatten()[0] / 1000.0).clamp(0.0, 1.0))
            s = curve_value_at(spec, 1.0 - sigma)
            if s >= 1.0:
                payload["cond_video_latents"] = state["pristine"]
            else:
                s = max(0.0, min(1.0, s))
                payload["cond_video_latents"] = [
                    s * p + (1.0 - s) * b
                    for p, b in zip(state["pristine"], state["blurred"])
                ]
        return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper


def _prompt_hint(loads) -> str:
    """Merge loaded mods' concept_type + description into one prompt-ready string.

    e.g. "identity: ginger woman, tattooed neck, black lipstick; pose_motion:
    slow twirl into camera, hair whipping". Concat this onto your positive
    prompt (a string-concat node ahead of CLIP Text Encode) instead of
    retyping each mod's description by hand. Mods with no description are
    skipped — a bare concept_type with nothing to say isn't a useful clue.
    """
    parts = []
    for mod, _strength in loads:
        if mod.description:
            parts.append(f"{mod.concept_type}: {mod.description}")
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModsLoader
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModsLoader:
    """Load 1-8 RefMods in one node, each with its own typed strength."""

    MAX_SLOTS = 8
    NONE = "(none)"

    @classmethod
    def INPUT_TYPES(cls):
        names = [cls.NONE] + _list_mod_names()
        required = {
            "show_info": ("BOOLEAN", {"default": False,
                "tooltip": "Print full details (tokens, layout, source, pool) of every loaded mod to the console."}),
        }
        for i in range(1, cls.MAX_SLOTS + 1):
            required[f"mod_{i}"] = (names, {"tooltip": f"RefMod {i} to load, or {cls.NONE}."})
            required[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                "step": 0.01, "display": "number",
                "tooltip": "How strongly this mod's reference is preserved. 1.0 = full ref (official "
                           "behavior). Lower values blur the ref toward a softened copy of itself — "
                           "identity fades smoothly and stays plausible instead of turning into "
                           "static/noise texture. 0 skips the mod entirely."})
            required[f"copies_{i}"] = ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                "display": "number",
                "tooltip": "How many copies of this mod to inject (1 = normal, 2+ = the same ref "
                           "repeated — the manual row-duplication trick as a knob, up to 10x). More "
                           "copies = noticeably stronger reference, but each copy costs its full "
                           "token count in every DiT block, so it slows down inference and eats "
                           "VRAM — 2-3 copies is the sweet spot, 10x will be very slow."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/mod"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        available = set(_list_mod_names())
        for i in range(1, cls.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", cls.NONE))
            if name and name != cls.NONE and name not in available:
                return (f"RefMod slot {i}: '{name}' not found in mods/. "
                        "Run Extract H3 RefMod first.")
        return True

    def load(self, show_info=False, **kwargs):
        rows = []  # (mod, strength, copies)
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or name == self.NONE or strength <= 0.0:
                continue
            rows.append((_load_mod(name), min(1.0, max(0.0, strength)),
                         int(kwargs.get(f"copies_{i}", 1))))
        loads = []
        for mod, strength, copies in rows:
            loads.extend([(mod, strength)] * copies)
        if loads:
            print("[MiniMaxH3RefModsLoader] " + ", ".join(
                f"{m.name}@{s:.2f}" + (f" x{c}" if c > 1 else "")
                for m, s, c in rows)
                + f" ({sum(m.token_count * c for m, _, c in rows)} tokens total)")
        else:
            print("[MiniMaxH3RefModsLoader] no mods selected "
                  "(all slots (none) or strength 0)")
        if show_info:
            for mod, strength, copies in rows:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}"
                      + (f"  (x{copies} copies)" if copies > 1 else ""))
        hint = _prompt_hint([(m, s) for m, s, _ in rows])
        if hint:
            print(f"[MiniMaxH3RefModsLoader] prompt_hint: {hint}")
        return (loads, hint)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModsAxis (signed A/B sliders)
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModsAxis:
    """A/B mod pairs on one signed slider each.

    Each row has an A-side mod, a B-side mod and one ``value`` slider in
    [-1, 1]: negative values use the A mod, positive values use the B mod, and
    the magnitude is the reference strength (same 0-1 math as the loader).  A
    value of 0 skips the row entirely.  This makes concept axes like "young
    <-> old" or "clean <-> weathered" a single dial: extract the two extremes
    once, then slide between them.
    """

    MAX_SLOTS = 8
    NONE = "(none)"

    @classmethod
    def INPUT_TYPES(cls):
        names = [cls.NONE] + _list_mod_names()
        required = {
            "show_info": ("BOOLEAN", {"default": False,
                "tooltip": "Print the selected A/B pairs and strengths to the console."}),
        }
        for i in range(1, cls.MAX_SLOTS + 1):
            required[f"mod_a_{i}"] = (names, {"tooltip": f"A-side RefMod {i} (used when value_{i} is negative), or {cls.NONE}."})
            required[f"mod_b_{i}"] = (names, {"tooltip": f"B-side RefMod {i} (used when value_{i} is positive), or {cls.NONE}."})
            required[f"value_{i}"] = ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0,
                "step": 0.01, "display": "number",
                "tooltip": "Signed strength: negative uses mod_a, positive uses mod_b, 0 skips the "
                           "row. The magnitude is the reference strength (same 0-1 math as "
                           "Load H3 RefMods), so -0.5 injects mod_a at half strength."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/mod"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        available = set(_list_mod_names())
        for i in range(1, cls.MAX_SLOTS + 1):
            for side in ("a", "b"):
                name = str(kwargs.get(f"mod_{side}_{i}", cls.NONE))
                if name and name != cls.NONE and name not in available:
                    return (f"RefMod slot {i} ({side}): '{name}' not found in mods/. "
                            "Run Extract H3 RefMod first.")
        return True

    def load(self, show_info=False, **kwargs):
        loads = []
        for i in range(1, self.MAX_SLOTS + 1):
            value = float(kwargs.get(f"value_{i}", 0.0))
            if abs(value) < 1e-6:
                continue
            side = "b" if value > 0 else "a"
            name = str(kwargs.get(f"mod_{side}_{i}", self.NONE))
            if not name or name == self.NONE:
                continue
            loads.append((_load_mod(name), min(1.0, abs(value))))
        if loads:
            print("[MiniMaxH3RefModsAxis] " + ", ".join(
                f"{m.name}@{s:+.2f}" for m, s in loads)
                + f" ({sum(m.token_count for m, _ in loads)} tokens total)")
        else:
            print("[MiniMaxH3RefModsAxis] no rows selected (values 0 or both sides (none))")
        if show_info:
            for mod, strength in loads:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}")
        hint = _prompt_hint(loads)
        if hint:
            print(f"[MiniMaxH3RefModsAxis] prompt_hint: {hint}")
        return (loads, hint)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModApply / ApplyCond
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModApply(io.ComfyNode):
    """
    Inject a loader bundle of RefMods into a MiniMax H3 conditioning.

    Accepts either the ComfyUI-MiniMaxH3 pack's MINIMAX_H3_COND or the built-in
    ComfyUI CONDITIONING (from the core MiniMaxH3ReferenceToVideo node) and
    returns the same type.  Appends each mod's reference latent to the
    conditioning's ``refs`` / ``minimax_refs``, so the DiT attends to it
    through all blocks exactly like a reference image/video.  ``retention`` is
    a master strength over the loader's per-row strengths; the curve is split
    into ``curve_direction`` (constant / concept_at_start / concept_at_end),
    ``curve_shape`` (how the envelope travels between its endpoints) and
    ``curve_value`` (the non-zero endpoint) — all plain widgets, no ComfyUI
    Curve widget required.  ``scramble_seed`` (default -1 = off) shuffles the
    ref order and keeps a random subset per run so a multi-ref mod can "pop"
    a different ref each time instead of always the same one.
    """

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template(
            "cond",
            allowed_types=[io.Custom("MINIMAX_H3_COND"), io.Conditioning])
        return io.Schema(
            node_id="MiniMaxH3RefModApply",
            display_name="Apply H3 RefMod",
            description=(
                "Inject a loader bundle of RefMods into a MiniMax H3 conditioning. "
                "Accepts both the pack's MINIMAX_H3_COND and the built-in "
                "CONDITIONING and returns the same type."
            ),
            category="MiniMax-H3/mod",
            inputs=[
                io.MatchType.Input("conditioning", template=template,
                    tooltip="MINIMAX_H3_COND (ComfyUI-MiniMaxH3 pack) or CONDITIONING "
                            "(core MiniMaxH3ReferenceToVideo)."),
                io.Custom("H3_REF_MODS").Input("mods",
                    tooltip="Bundle from Load H3 RefMods / Load H3 RefMod Axis / Extract H3 RefMod."),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Master reference strength, multiplied with each loader row's "
                             "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                             "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                             "style/attributes, not identity), 0.15 = weak_reference. "
                             "0 = no reference."),
                io.Combo.Input("curve_direction", options=list(CURVE_DIRECTIONS),
                    default="concept_at_end",
                    tooltip="Where the concept shows up in the output (the mirror of the ref's "
                            "strength envelope): 'concept_at_end' (default, was 'decrease') locks "
                            "the ref's literal footage in at the START and releases it toward the "
                            "end — the identity/character emerges in the second half, without "
                            "dragging the ref's background in; 'concept_at_start' (was 'increase') "
                            "opens free from the ref and locks onto it near the END — the concept "
                            "shows early; 'concept_at_middle' peaks mid-video ([0..1..0] — the "
                            "concept appears only in the middle); 'concept_at_ends' holds both "
                            "ends with a mid dip ([1..0..1]); 'constant' keeps one strength for "
                            "the whole video (flat at curve_value = 1.0, today's behavior). Old "
                            "'decrease'/'increase' values saved in workflows still resolve."),
                io.Int.Input("scramble_seed", default=-1, min=-1, max=2147483647, step=1,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Ref scrambling seed. -1 (default) = off: all refs in saved order. "
                            "With 2+ refs in the bundle, a seed >= 0 shuffles the ref order and "
                            "keeps a random subset, so a different ref leads each run (a multi-ref "
                            "mod 'pops' a different video/image per seed). Same seed = same "
                            "scramble; set this widget's control-after-generate to 'randomize' "
                            "for per-run variation."),
                io.Combo.Input("curve_shape", options=list(CURVE_SHAPES),
                    default="ease",
                    tooltip="How the envelope travels between its endpoints: 'ease' (smoothstep, "
                            "default), 'linear', 'sigmoid'/'tanh' (smooth S-curves, tanh with a "
                            "steeper knee), 'quadratic', 'cubic', 'exponential', 'stair' "
                            "(stepped), 'elastic' (overshoots), 'bump' (peak mid-video, for "
                            "one specific action), 'dip' (trough mid-video)."),
                io.Float.Input("curve_value", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Endpoint value ('user input'): both endpoints for 'constant' and "
                            "'concept_at_ends', the end for 'concept_at_start', the start for "
                            "'concept_at_end', the mid peak for 'concept_at_middle'. "
                            "1.0 = full strength there."),
                io.Combo.Input("graph_preset",
                    options=["(none)"] + _list_graph_presets(), default="(none)",
                    optional=True,
                    tooltip="Optional shared graph preset — leave on '(none)' to use the curve "
                            "widgets above. Selecting one loads direction/shape/value from a "
                            "saved debug-grid PNG (graph embedded in its metadata) or a legacy "
                            ".json, in models/refmods/graph_presets/. Share the preset PNG "
                            "itself to share a curve. New presets appear after a restart."),
                io.String.Input("save_preset_as", default="", optional=True,
                    tooltip="Optional: type a name and run to save the current (resolved) curve "
                            "as a PNG preset — the curve graph itself with the graph embedded in "
                            "its metadata — in models/refmods/graph_presets/. Share that image "
                            "to share the curve. Leave empty to skip."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="conditioning",
                    tooltip="The conditioning with the ref blocks injected, same type as the input."),
                io.Image.Output("debug", display_name="curve graph",
                    tooltip="Optional 1024x1024 curve graph: the strength envelope "
                            "(direction/shape/value) with the concept zone shaded. Leave "
                            "unconnected to skip the preview."),
            ],
        )

    @classmethod
    def execute(cls, conditioning, mods, retention=1.0,
                curve_direction="concept_at_end", curve_shape="ease", curve_value=1.0,
                strength_curve=None, scramble_seed=-1, graph_preset="", save_preset_as=""):
        # workflows saved before the curve split pass the old single preset name
        curve = strength_curve if strength_curve is not None \
            else (curve_direction, curve_shape, curve_value)
        # a selected graph preset overrides the curve widgets
        preset_name = ""
        if graph_preset and graph_preset != "(none)":
            loaded = _load_graph_preset(graph_preset)
            if loaded is None:
                print(f"[MiniMaxH3RefModApply] WARNING: graph preset '{graph_preset}' "
                      f"not found or invalid — using widget curve")
            else:
                curve = loaded
                preset_name = graph_preset
        img = render_debug_grid(curve, preset_name)
        if save_preset_as:
            saved = _save_graph_preset(save_preset_as, curve, img)
            if saved:
                print(f"[MiniMaxH3RefModApply] graph preset saved: {saved}.png "
                      f"({curve[0]} + {curve[1]} @ {float(curve[2]):.2f})")
        blocks = _ref_blocks(mods, retention, curve, seed=scramble_seed)
        if isinstance(conditioning, list):
            # built-in ComfyUI CONDITIONING (core MiniMaxH3ReferenceToVideo)
            out = []
            for t in conditioning:
                d = dict(t[1])
                d["minimax_refs"] = list(d.get("minimax_refs", [])) + blocks
                out.append([t[0], d])
            print(f"[MiniMaxH3RefModApply] retention={retention} "
                  f"({len(blocks)} ref block(s) injected)")
        else:
            # ComfyUI-MiniMaxH3 pack MINIMAX_H3_COND
            out = replace(conditioning, refs=list(conditioning.refs) + blocks)
            print(f"[MiniMaxH3RefModApply] retention={retention} "
                  f"({len(blocks)} ref block(s) injected, {len(out.refs)} total)")
        return io.NodeOutput(out, pil_to_tensor(img))


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModStepCurve
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModStepCurve:
    """Per-step (per-sigma) reference strength curve, applied at generation time.

    The Apply node's frame curve is baked into the ref latent once, before
    sampling.  This node instead re-mixes every ref latent once per denoising
    step: early steps (high sigma) set global structure and identity, late
    steps (low sigma) paint fine texture — so the same direction/shape/value
    envelope runs over the denoise timeline instead of the video's.  Same
    curve widgets as Apply; attach between the model loader and the sampler
    (MODEL -> MODEL, same type).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The H3 model to patch. Returned unchanged apart from the per-step "
                               "ref-mixing wrapper."}),
                "curve_direction": (list(CURVE_DIRECTIONS), {"default": "concept_at_end",
                    "tooltip": "Same envelope as the Apply frame curve, but over the DENOISE "
                               "timeline: 'concept_at_end' (default) keeps the refs at full "
                               "strength in the early steps (high sigma — composition and identity "
                               "set first) and releases them toward the final steps (clean texture, "
                               "no ref grain); 'concept_at_start' opens weak and locks full strength "
                               "in the late steps (identity detail refined at the end); 'constant' "
                               "keeps one strength for every step; 'concept_at_middle' peaks "
                               "mid-denoise; 'concept_at_ends' holds the extremes and dips "
                               "mid-denoise. Old 'decrease'/'increase' values still resolve."}),
                "curve_shape": (list(CURVE_SHAPES), {"default": "ease",
                    "tooltip": "How the per-step strength travels between its endpoints (same "
                               "shapes as the Apply frame curve: linear / ease / sigmoid / tanh / "
                               "quadratic / cubic / exponential / stair / elastic / bump / dip)."}),
                "curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Endpoint strength ('user input'): both endpoints for 'constant' and "
                               "'concept_at_ends', the end for 'concept_at_start', the start for "
                               "'concept_at_end', the mid peak for 'concept_at_middle'. "
                               "1.0 = full ref there."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/mod"

    def apply(self, model, curve_direction="concept_at_end",
              curve_shape="ease", curve_value=1.0):
        model = model.clone()
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "minimax_h3_refmod_step_curve",
            _make_step_wrapper((curve_direction, curve_shape, curve_value)))
        print(f"[MiniMaxH3RefModStepCurve] {curve_direction} + {curve_shape} "
              f"@ {curve_value:.2f} attached — refs re-mixed per denoising step")
        return (model,)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModFolderLoader
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModFolderLoader:
    """Load every image/video in a folder as an ordered ref list.

    Feed the ``refs_bundle`` input of Extract H3 RefMod to bulk-extract a
    whole folder (e.g. all photos of a character).  Images load first (by
    filename), then videos; unreadable files are skipped with a note.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "",
                    "tooltip": "Folder with reference images/videos. An absolute path, or a folder "
                               "name inside ComfyUI's input/ directory (empty = input/ itself)."}),
                "max_items": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                    "display": "number",
                    "tooltip": "Max media files loaded (images first, then videos, by filename)."}),
                "max_frames": ("INT", {"default": 240, "min": 2, "max": 4800, "step": 1,
                    "display": "number",
                    "tooltip": "Video frames kept (uniformly sampled during decode, so a long video "
                               "is never fully decoded into RAM — memory stays bounded by this cap "
                               "x max_edge resolution). 240 = ~10s at 24fps."}),
                "max_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                    "display": "number",
                    "tooltip": "Longest edge in px for loaded images/videos (downscale only, never "
                               "upscale). Loading many 4K files at native resolution is what OOMs "
                               "ComfyUI — the Extract node resizes to ref_resolution anyway, so "
                               "1024-1280 is plenty for folder extraction."}),
            },
        }

    RETURN_TYPES = ("H3_REF_LIST", "INT")
    RETURN_NAMES = ("refs", "count")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/mod"

    @classmethod
    def VALIDATE_INPUTS(cls, folder):
        try:
            _resolve_folder(folder)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, folder, max_items=32, max_frames=240, max_edge=1024):
        try:
            images, videos = list_media_files(_resolve_folder(folder))
            parts = []
            for p in (images + videos)[:max_items]:
                try:
                    st = os.stat(p)
                    parts.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
                except OSError:
                    parts.append(f"{os.path.basename(p)}:missing")
            return "|".join(parts)
        except Exception:
            return ""

    def load(self, folder, max_items=32, max_frames=240, max_edge=1024):
        folder = _resolve_folder(folder)
        images, videos = list_media_files(folder)
        items = (images + videos)[:max_items]
        total = len(items)
        refs, failed = [], []
        pbar = comfy.utils.ProgressBar(total)
        for i, p in enumerate(items, start=1):
            kind = "video" if p in videos else "image"
            print(f"[MiniMaxH3RefModFolderLoader] [{i}/{total}] loading {kind} "
                  f"{os.path.basename(p)}")
            try:
                if p in images:
                    refs.append(load_image_file(p, max_edge=max_edge))
                else:
                    refs.append(load_video_file(p, max_frames=max_frames, max_edge=max_edge))
            except Exception as exc:
                failed.append(f"{os.path.basename(p)} ({type(exc).__name__})")
                pbar.update_absolute(i)
                continue
            print(f"[MiniMaxH3RefModFolderLoader] [{i}/{total}] {os.path.basename(p)} "
                  f"-> {tuple(refs[-1].shape)}")
            pbar.update_absolute(i)
        if failed:
            print(f"[MiniMaxH3RefModFolderLoader] skipped unreadable files: {', '.join(failed)}")
        if not refs:
            raise ValueError(
                f"MiniMaxH3RefModFolderLoader: no images/videos found in {folder} "
                "(images: png/jpg/jpeg/webp/bmp/gif, videos: mp4/webm/mov/mkv/avi/m4v).")
        n_vid = sum(r.shape[0] > 1 for r in refs)
        print(f"[MiniMaxH3RefModFolderLoader] loaded {len(refs)} media from {folder} "
              f"({n_vid} video, {len(refs) - n_vid} image)")
        return (refs, len(refs))


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModExtract (V3 — Autogrow reference inputs)
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModExtract(io.ComfyNode):
    """
    Turn one or more references of the same concept into a RefMod.

    Refs are added with the "+" button: stills plug into ``ref_image_1``,
    video frames into ``ref_video_1``, and the next slot of that type appears.
    Each ref is one row of the same concept (different angle / expression /
    setting / a dance move); they are stacked into a video-kind mod.

    Two modes:

      * ``encode`` (default) — each ref is resized to ``ref_resolution`` short
        edge (down only) and VAE-encoded at that resolution, exactly like the
        official ref2video node.  The mod stores the real encode, so identity
        (a face, an outfit) comes through; files are ~0.2-1 MB per frame.
        (Old name: ``full``.)
      * ``training`` — each ref is first resized to ``ref_resolution`` short
        edge too (the latent is pooled to a tiny grid anyway, so encoding at
        native resolution is wasted compute — this is the main speed dial for
        training mode), then average-pooled to a tiny grid (4x4 = 4 tokens
        per frame) and refined with gradient steps against the encode — still
        no diffusion model.  Nearly free to inject but only carries concept /
        motion, not fine identity.  (Old name: ``pooled``.)

    ``identity`` (training mode only) is the refinement loop — the only
    "training" in the pack.

    ``max_tokens`` (0 = off) hard-caps the total injected tokens: when the
    stacked refs exceed it, near-duplicate latent frames are dropped first,
    then frames are resampled to fit (see ``core.fit_token_budget``).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefModExtract",
            display_name="Extract H3 RefMod",
            description=(
                "Turn one or more references of the same concept into a RefMod. "
                "Stills plug into ref_image_1, video frames into ref_video_1, "
                "and the next slot of that type appears. refs are stacked into "
                "one video-kind mod, so a multi-image moodboard keeps each ref's "
                "own content instead of averaging away. 'training' mode (default) "
                "compresses the refs to a grid and refines it — good identity at "
                "a fraction of the tokens; 'encode' stores the full-res encode "
                "(max identity, MB-size mod)."
            ),
            category="MiniMax-H3/mod",
            inputs=[
                io.String.Input("name", default="my_concept",
                    tooltip="Saved mod name (appears in the Load H3 RefMods dropdown after a reload)."),
                io.Combo.Input("mode", options=["training", "encode"],
                    default="training",
                    tooltip="'training' (default) = compressed grid refined by the 'identity' "
                            "dial — a good balance of identity vs tokens. 'encode' = straight "
                            "full-res VAE encode (max identity, MB-size mod, ~1K tokens/img). "
                            "Old mods saved as 'full'/'pooled' still load and normalize to "
                            "these two."),
                io.Combo.Input("concept_type", options=list(CONCEPT_TYPES), default="generic",
                    tooltip="What this mod represents — 'identity' (a specific person/character), "
                            "'pose_motion' (a pose/dance/gesture/camera move), 'clothing', "
                            "'background', 'style', or 'generic'. Stored in the mod and used by "
                            "the loaders' prompt_hint output (merges concept_type + description "
                            "into a string you can concat onto your CLIP prompt). 'identity' in "
                            "training mode with a small grid also triggers a warning nudging you "
                            "toward 'encode' mode or a bigger grid — pooling is lossy in exactly "
                            "the way that destroys facial identity."),
                io.Autogrow.Input("refs_image", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference still: one image of the "
                            "concept (angle / expression / outfit). Optional — leave empty when using "
                            "video refs and/or a folder bundle."),
                        prefix="ref_image_", min=0, max=16)),
                io.Autogrow.Input("refs_video", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames "
                            "[T,H,W,C] (a multi-frame batch = a video ref with motion). "
                            "Optional — leave empty when using image refs and/or a folder bundle."),
                        prefix="ref_video_", min=0, max=8)),
                io.Custom("H3_REF_LIST").Input("refs_bundle", optional=True,
                    tooltip="All images/videos from a Load H3 RefMod Folder node, appended after "
                            "the autogrow refs (bulk extraction)."),
                io.Mask.Input("mask", optional=True,
                    tooltip="Subject mask (or a batch, one per reference in order: images then "
                            "videos) marking what to keep at full weight. Everything outside the "
                            "mask collapses toward a heavily blurred copy of itself per spatial "
                            "cell (stays in-distribution — a flat noise-mix here decodes as a "
                            "woven/static texture instead of 'nothing'), controlled by "
                            "background_retention. Fixes 'encode' mode pulling in a background/style "
                            "that doesn't belong to the subject. A single mask broadcasts to every "
                            "reference; a batch must match the reference count."),
                io.Float.Input("background_retention", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Only used when 'mask' is connected. Floor weight for the region "
                            "outside the mask: 0 = that region collapses to a heavily blurred "
                            "copy of itself (kills specific structure like a skyline/treeline "
                            "while staying smooth and in-distribution), 1 = mask has no effect. "
                            "Middle values (0.3-0.6) partially blur instead of fully."),
                io.Custom("MINIMAX_H3_AV_ENCODER").Input("av_encoder", optional=True,
                    tooltip="MiniMax-H3 VAE pack output (preferred; share the pack's VAE cache)."),
                io.Vae.Input("vae", optional=True,
                    tooltip="Standard VAE, used when av_encoder is not connected."),
                io.Int.Input("ref_resolution", default=1024, min=256, max=2048, step=64,
                    tooltip="Target short edge in px (downscale only, never upscale), applied to "
                            "BOTH modes: 'encode' stores at that res, 'training' encodes smaller "
                            "too (it pools to a grid anyway, so native-res encoding is wasted "
                            "compute — this is the main speed dial for training mode). 1024 is a "
                            "good default; 512 halves encode cost; 2048 = official max fidelity, "
                            "4x the tokens of 1024."),
                io.Int.Input("pool_h", default=16, min=2, max=64, step=2,
                    tooltip="Pooled mode: spatial latent grid after pooling. The grid is auto-fit to "
                            "the source's aspect ratio (long edge = max of the two dials, other edge "
                            "derived), so a portrait person isn't squished into a square grid "
                            "(the 'fat/chubby' distortion). Square sources keep the exact dial value. "
                            "16x16 = 64 tokens/frame (concept sweet spot); 32x32 = 256; 64x64 = 1024, "
                            "full-mode parity for identity."),
                io.Int.Input("pool_w", default=16, min=2, max=64, step=2,
                    tooltip="Pooled mode: grid width (long edge if the source is wider than tall)."),
                io.Int.Input("latent_frames", default=16, min=1, max=16,
                    tooltip="Frames kept per video ref: training mode pools them, encode mode uniformly "
                            "samples them (16x16x16 = 4096 tokens per video ref). Images always use 1."),
                io.Int.Input("identity", default=500, min=0, max=2000, step=50,
                    tooltip="Pooled mode only: how tightly the mod clings to the reference "
                            "(gradient refinement steps). Higher = more identity detail but sticks "
                            "to the refs' framing/background; lower = deviates from the refs but "
                            "loses detail. 500 is a good default; 0 = pure pooling."),
                io.Int.Input("multiplier", default=1, min=1, max=10, step=1,
                    tooltip="Data multiplier: repeat the extracted ref N times along time so a short "
                            "video/GIF (few tokens) isn't drowned out by the main video's tokens. "
                            "Each repeat duplicates the same latent frames, so attention weight on "
                            "the ref scales roughly with N. 1 = no repeat; file size grows with N."),
                io.Int.Input("max_tokens", default=5120, min=0, max=65536, step=512,
                    tooltip="Hard cap on the total tokens the mod injects (0 = no cap; 5120 is a good "
                            "performance default). If the stacked refs exceed it, near-duplicate "
                            "latent frames are dropped first (video refs are full of frames that "
                            "differ only by noise — each one still costs a token per spatial patch "
                            "in every block), then frames are resampled to fit. The cap is honored "
                            "after the multiplier. Lower latent_frames/ref_resolution instead to "
                            "avoid wasting encode work: ~23K tokens = one 1024px encode-mode video "
                            "ref at 16 frames."),
                io.String.Input("description", default="", multiline=True,
                    tooltip="Optional text describing the concept (e.g. 'a ginger woman with messy "
                            "hair', 'an animation style', 'handheld camera movement'). Stored in "
                            "the mod and printed in the info block — documentation only, no wiring."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to mods/ so Load H3 RefMods can pick it up later."),
            ],
            outputs=[
                io.Custom("H3_REF_MODS").Output("mods",
                    tooltip="Bundle with this one mod at strength 1.0. Feed it to Apply H3 RefMod "
                            "(or Load H3 RefMods after saving)."),
            ],
        )

    @classmethod
    def execute(cls, name, mode, refs_image=None, refs_video=None, refs_bundle=None,
                av_encoder=None, vae=None,
                ref_resolution=1024, pool_h=16, pool_w=16, latent_frames=16,
                identity=500, multiplier=1, max_tokens=0, description="", save=True,
                concept_type="generic", mask=None, background_retention=0.0,
                **legacy) -> io.NodeOutput:
        name = _sanitize_name(name)
        mode = normalize_mode(mode)  # accept legacy 'full'/'pooled'
        if concept_type == "identity" and mode == "training" and max(pool_h, pool_w) < 16:
            print(
                f"[MiniMaxH3RefModExtract] warning: concept_type='identity' with "
                f"mode='training' at a {pool_h}x{pool_w} grid — pooling averages away "
                f"exactly the detail that carries a face (this is almost certainly "
                f"your 'chubby/older' drift). For a person, either switch mode='encode' "
                f"(real identity, higher token cost) or raise pool_h/pool_w toward "
                f"32x32+ and expect it to still be a soft approximation, not a lock."
            )
        # old pre-Autogrow workflows pass their widget values through as kwargs:
        # map them onto the new inputs so those saved workflows keep running.
        # ``pool`` is the old height; ``pool_w`` arrives as the named param.
        if legacy.get("optimize") is not None:
            identity = legacy["optimize"]
        if legacy.get("pool") is not None:
            pool_h = int(legacy["pool"])
            if pool_w == 16:  # old single-pool default: square grid
                pool_w = pool_h
        if av_encoder is None and vae is None:
            raise ValueError(
                "MiniMaxH3RefModExtract: connect an av_encoder (MiniMax-H3 "
                "VAE loader) or a standard VAE.")
        pack = None
        if av_encoder is not None:
            vae_pack_mod = _h3_pack_submodule("models.vae")
            pack = vae_pack_mod.load_vae_pack(av_encoder.video_path, av_encoder.audio_path)

        # each Autogrow arrives as a dict keyed by its slot names
        # (ref_image_1..N / ref_video_1..N); videos stay multi-frame, images are
        # pinned to a single still.  Legacy workflows used flat image /
        # ref_image_N / ref_video_N inputs; a folder bundle is appended last.
        ordered = []
        for key in sorted((refs_image or {}).keys(),
                          key=lambda k: int(k.rsplit("_", 1)[1])):
            src = (refs_image or {})[key]
            if src is not None:
                ordered.append((src, False))
        for key in sorted((refs_video or {}).keys(),
                          key=lambda k: int(k.rsplit("_", 1)[1])):
            src = (refs_video or {})[key]
            if src is not None:
                ordered.append((src, True))
        legacy_keys = sorted(
            (k for k in legacy if k == "image" or k.startswith("ref_image_")
             or k.startswith("ref_video_")),
            key=lambda k: (0 if k == "image" else 1 if k.startswith("ref_image_") else 2,
                           int(k.rsplit("_", 1)[1]) if "_" in k else 0))
        for key in legacy_keys:
            if legacy[key] is not None:
                ordered.append((legacy[key], key.startswith("ref_video_")))
        if refs_bundle is not None:
            for src in refs_bundle:
                if src is not None:
                    norm = _normalize_ref(src, label="folder reference")
                    ordered.append((norm, norm.shape[0] > 1))
        if not ordered:
            raise ValueError(
                "MiniMaxH3RefModExtract: connect at least one image to "
                "ref_image_1, or video frames to ref_video_1, or a folder bundle.")
        sources = []
        for i, (src, is_video) in enumerate(ordered):
            norm = _normalize_ref(src, label=f"reference {i + 1}")
            if is_video:
                sources.append((norm, norm.shape[0] > 1))
            else:
                # image slot: pin to a single still even if a batch arrived
                sources.append((norm[:1], False))
        # encode each source independently (full-res or pooled), then stack.
        # Full-res refs must share one spatial canvas so the stacked latent has
        # a single H/W: anchor on the first source, cover-crop the rest to it.
        canvas = None
        if mode == "encode" and len(sources) > 1:
            h, w = sources[0][0].shape[1], sources[0][0].shape[2]
            scale = min(1.0, ref_resolution / min(h, w))
            canvas = (max(32, round(w * scale / 32) * 32),
                      max(32, round(h * scale / 32) * 32))
        # training mode: anchor the pool grid to the first source's aspect so
        # a portrait person isn't squished into a square 16x16 grid (the
        # "fat" distortion).  The VAE scales space uniformly, so pixel
        # aspect == latent aspect.
        pool_grid = None
        if mode == "training":
            h0, w0 = sources[0][0].shape[1], sources[0][0].shape[2]
            pool_grid = aspect_grid(pool_h, pool_w, h0 / w0)
            if pool_grid != (pool_h, pool_w):
                print(f"[MiniMaxH3RefModExtract] pooled grid {pool_h}x{pool_w} -> "
                      f"{pool_grid[0]}x{pool_grid[1]} to match source aspect "
                      f"{w0}x{h0} (avoids squishing the subject wide)")
        gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)

        mask_batch = _normalize_mask_batch(mask, label="mask")
        if mask_batch is not None:
            if mask_batch.shape[0] == 1 and len(sources) > 1:
                mask_batch = mask_batch.expand(len(sources), -1, -1)
            elif mask_batch.shape[0] != len(sources):
                raise ValueError(
                    f"MiniMaxH3RefModExtract: mask has {mask_batch.shape[0]} entries but "
                    f"there are {len(sources)} references (images then videos, in order). "
                    f"Connect one mask (broadcasts to every ref) or exactly one per ref.")

        frames = []
        n_img = n_vid = 0
        source_shapes = []
        n_refs = len(sources)
        pbar = comfy.utils.ProgressBar(n_refs)
        for src_idx in range(len(sources)):
            src, is_video = sources[src_idx]
            label = f"ref {src_idx + 1}/{n_refs} ({'video' if is_video else 'image'})"
            print(f"[MiniMaxH3RefModExtract] {label}: "
                  f"source {tuple(src.shape)}, mode={mode}"
                  + (f", identity={identity} steps" if mode == "training" and identity > 0 else ""))
            if mode == "encode":
                # downscale (never upscale) to the target short edge, sample
                # videos to latent_frames frames, then encode at full res
                if is_video and latent_frames < src.shape[0]:
                    idx = torch.linspace(0, src.shape[0] - 1, latent_frames).round().long()
                    src = src[idx]
                src = _resize_ref(src, ref_resolution, canvas)
            else:
                # training mode: encode smaller too — the latent is pooled
                # to a tiny grid anyway, so encoding at native resolution is
                # wasted compute. Resize preserves aspect, so the pool grid
                # anchored on the first source's aspect still applies.
                orig = (src.shape[1], src.shape[2])
                src = _resize_ref(src, ref_resolution, None)
                if (src.shape[1], src.shape[2]) != orig:
                    print(f"[MiniMaxH3RefModExtract] {label}: resized "
                          f"{orig[0]}x{orig[1]} -> {src.shape[1]}x{src.shape[2]} "
                          f"(ref_resolution={ref_resolution}) before encode")
            src = _ensure_min_size(src)
            if is_video and src.shape[0] > 1:
                valid_t = _snap_to_causal_grid(src.shape[0])
                if valid_t != src.shape[0]:
                    print(f"[MiniMaxH3RefModExtract] reference {src_idx + 1} "
                          f"(video): trimming {src.shape[0]} -> {valid_t} frames "
                          f"to match the VAE's causal 4k+1 grid.")
                    src = src[:valid_t]
            mask_px = None
            if mask_batch is not None:
                mask_px = _resize_mask(mask_batch[src_idx:src_idx + 1], src.shape[1], src.shape[2])
            if src.shape[1] <= 0 or src.shape[2] <= 0:
                raise ValueError(
                    f"MiniMaxH3RefModExtract: reference {src_idx + 1} "
                    f"({'video' if is_video else 'image'}) has an empty frame "
                    f"{tuple(src.shape)} right before VAE encode (mode={mode}, "
                    f"ref_resolution={ref_resolution}, canvas={canvas}). "
                    f"Check that this specific reference's source image/video "
                    f"is valid.")
            # encode-path conventions differ:
            #  - av_encoder -> pack's raw H3 VAE: channel-first [1, 3, T, H, W]
            #    in [-1, 1] (same as the pack's own conditioning node)
            #  - vae -> comfy sd.VAE wrapper: channel-last [T, H, W, C] in [0, 1];
            #    the wrapper does its own layout conversion and /16 cropping, and
            #    would misread channel-first input (narrowing the channel dim to 0)
            if pack is not None:
                moved = src.movedim(-1, 1)
                if moved.shape[0] == 1:
                    pixels = moved
                else:
                    pixels = moved.permute(1, 0, 2, 3).unsqueeze(0)
                pixels = (pixels * 2.0 - 1.0).to(torch.float16)
                z = pack.encode_video(pixels)
            else:
                z = vae.encode(src)
            if z.dim() != 5 or z.shape[1] != 24:
                raise ValueError(
                    f"Expected a MiniMax H3 video VAE latent [1,24,T,H,W], "
                    f"got {tuple(z.shape)}. The connected VAE is not the H3 VAE.")
            source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")

            if mask_px is not None:
                z = _mask_latent(z, mask_px, background_retention, seed_key=f"{name}:{src_idx}")
                print(f"[MiniMaxH3RefModExtract] {label}: applied subject mask "
                      f"(background_retention={background_retention})")

            if mode == "encode":
                pooled = z.to(torch.float16)
            else:
                pool_t = min(latent_frames, z.shape[2]) if is_video else 1
                gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)
                pooled = pool_latent(z, pool_t, gh, gw).to(torch.float16)
                if identity > 0:
                    print(f"[MiniMaxH3RefModExtract] {label}: refining identity "
                          f"({int(identity)} gradient steps)...")
                    pooled = optimize_latent(pooled, z.float(), steps=int(identity),
                                              progress_every=100)
                    print(f"[MiniMaxH3RefModExtract] {label}: identity refinement done")
            frames.append(pooled)
            if is_video:
                n_vid += 1
            else:
                n_img += 1
            pbar.update_absolute(src_idx + 1)
            print(f"[MiniMaxH3RefModExtract] {label}: encoded "
                  f"{tuple(pooled.shape)} ({pooled.numel() * pooled.element_size() / 1024 / 1024:.2f} MB)")
            # drop the decoded source and the full-res latent as soon as we're
            # done with them, so a large folder doesn't keep every source +
            # every full encode resident while the remaining refs are encoded
            sources[src_idx] = None
            src = None
            z = None

        if mode == "encode" and identity > 0:
            print(f"[MiniMaxH3RefModExtract] warning: 'identity' only applies to "
                  f"training mode — encode mode stores the actual encode, so "
                  f"identity={identity} was ignored.")

        latent = torch.cat(frames, dim=2)  # [1, 24, total_t, h, w]
        if multiplier > 1:
            latent = latent.repeat(1, 1, multiplier, 1, 1)  # data multiplier
        if max_tokens > 0:
            latent = fit_token_budget(latent, max_tokens, name)
        total_t = latent.shape[2]
        kind = "video" if total_t > 1 else "image"
        # the VAE encodes at 16x spatial scale, so a latent of 40x20 = 640x320 px
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
        mod = H3RefMod(
            name=name,
            kind=kind,
            latent=latent,
            latent_h=latent.shape[3],
            latent_w=latent.shape[4],
            latent_t=total_t,
            mode=mode,
            source="stack" if len(frames) > 1 else ("video" if n_vid else "image"),
            source_shape=" +".join(source_shapes),
            pool=f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode" else f"{total_t}x{gh}x{gw}",
            optimize_steps=int(identity) if mode == "training" else 0,
            tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
                + ([f"masked (bg_retention={background_retention})"] if mask_batch is not None else []),
            description=(description or "").strip(),
            concept_type=concept_type,
        )

        if save:
            path = mod.save(os.path.join(refmods_dir(), name))
            _MOD_CACHE[name] = mod
            if len(_MOD_CACHE) > _MOD_CACHE_MAX:
                _MOD_CACHE.pop(next(iter(_MOD_CACHE)))
            _MOD_LIST_CACHE_KEY = None  # new mod -> refresh the dropdown listing
            print(f"[MiniMaxH3RefModExtract] saved {_summarize(mod)} -> {path}")
        else:
            print(f"[MiniMaxH3RefModExtract] {_summarize(mod)} (not saved)")
        if mod.description:
            print(f"[MiniMaxH3RefModExtract] description: {mod.description}")
        return io.NodeOutput([(mod, 1.0)])


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RefModExtract": MiniMaxH3RefModExtract,
    "MiniMaxH3RefModFolderLoader": MiniMaxH3RefModFolderLoader,
    "MiniMaxH3RefModsLoader": MiniMaxH3RefModsLoader,
    "MiniMaxH3RefModsAxis": MiniMaxH3RefModsAxis,
    "MiniMaxH3RefModApply": MiniMaxH3RefModApply,
    "MiniMaxH3RefModStepCurve": MiniMaxH3RefModStepCurve,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefModExtract": "Extract H3 RefMod",
    "MiniMaxH3RefModFolderLoader": "Load H3 RefMod Folder",
    "MiniMaxH3RefModsLoader": "Load H3 RefMods",
    "MiniMaxH3RefModsAxis": "Load H3 RefMod Axis",
    "MiniMaxH3RefModApply": "Apply H3 RefMod",
    "MiniMaxH3RefModStepCurve": "H3 RefMod Step Curve",
}

# The old Apply node was split into two (pack MINIMAX_H3_COND vs built-in
# CONDITIONING); the merged node above accepts both.  Old workflows saved with
# MiniMaxH3RefModApplyCond are migrated to the merged node at load time by the
# replacement below (the old id is deliberately not registered so the manager
# rewrites it).  Registered once at import; PromptServer exists by the time
# custom nodes load (main.py creates it before init_extra_nodes).
try:
    from comfy_api.latest import ComfyAPI
    from server import PromptServer
    if PromptServer.instance is not None:
        manager = PromptServer.instance.node_replace_manager
        manager.register(io.NodeReplace(
            new_node_id="MiniMaxH3RefModApply",
            old_node_id="MiniMaxH3RefModApplyCond",
            old_widget_ids=["retention"],
            input_mapping=[
                {"new_id": "conditioning", "old_id": "conditioning"},
                {"new_id": "mods", "old_id": "mods"},
                {"new_id": "retention", "old_id": "retention"},
            ],
            output_mapping=[{"new_idx": 0, "old_idx": 0}],
        ))
except Exception:
    # standalone/CLI contexts without a running server: migration just won't
    # be registered until ComfyUI actually loads the pack
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
