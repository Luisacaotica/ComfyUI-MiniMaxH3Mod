"""
nodes.py — ComfyUI nodes for MiniMax H3 "RefMod" (no-training reference mods)

  MiniMaxH3RefModExtract       — Autogrow reference inputs (image or video frames)
                                 -> a saved mod, output as an H3_REF_MODS bundle
  MiniMaxH3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  MiniMaxH3RefModsLoader       — load 1-8 mods with a typed strength each (LoRA-style)
  MiniMaxH3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)
  MiniMaxH3RefModApply         — inject the bundle into a MINIMAX_H3_COND conditioning
  MiniMaxH3RefModApplyCond     — same, for the built-in ComfyUI CONDITIONING type

Mods are stored in ``models/refmods/`` (created on first run, next to loras/
and unet/); mods saved by older versions in the pack's ``mods/`` folder still
load.

The mod rides the model's native ref2va path: the Apply nodes append reference
blocks (the mod latents) to the conditioning's ``refs``, and the DiT attends
to those tokens through all of its blocks, exactly like a full image/video
reference but at a fraction of the token budget.

Reference strength uses the model's own conditioning-strength mechanism:
weakening a ref mixes its latent toward noise (``visual_cond_noise_aug``
semantics) instead of scaling values toward zero, which pushed latents out of
distribution and read as grey output.  ``retention`` on the Apply nodes is a
preset master strength (fully_preserved / partially_preserved /
attribute_transfer / weak_reference) multiplied with each loader row's
strength.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import replace
from typing import Dict, List

import torch

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
    H3RefMod,
    optimize_latent,
    pool_latent,
    read_refmod_meta,
)

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_MODS_DIR = os.path.join(_PACK_DIR, "mods")  # pre-models/refmods storage, still read
_MOD_CACHE: Dict[str, H3RefMod] = {}

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
    """
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
    return sorted(names)


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
    return mod


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
    regardless of extraction mode ('full' already resizes down to
    ref_resolution but never guarantees a floor; 'pooled' never resizes at
    all), so it's a separate, unconditional safety net right before encode.
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
    if mod.mode == "full":
        opt = f"n/a ({mod.optimize_steps} — full mode stores the actual encode)"
    return [
        "=" * 52,
        f"  MiniMax H3 RefMod: {mod.name}",
        f"  {'mode':<18} {mod.mode}",
        f"  {'kind':<18} {mod.kind}",
        f"  {'latent':<18} {tuple(mod.latent.shape)}",
        f"  {'tokens injected':<18} {mod.token_count}",
        f"  {'source':<18} {mod.source} ({mod.source_shape})",
        f"  {'pool':<18} {mod.pool}",
        f"  {'identity':<18} {opt}",
        f"  {'tags':<18} {', '.join(mod.tags) if mod.tags else '-'}",
        "=" * 52,
    ]


def _ref_blocks(mods, retention) -> List[Dict]:
    """Ref blocks for a loader bundle, scaled by row strength x retention.

    ``retention`` is a master strength multiplier: a float 0-1 (1.0 =
    fully_preserved, 0.7 = partially_preserved, 0.4 = attribute_transfer,
    0.15 = weak_reference), or one of those preset names for legacy
    workflows saved with the old combo widget.
    """
    if isinstance(retention, str):
        factor = RETENTION.get(retention, 1.0)
    else:
        factor = float(retention)
    blocks = []
    for mod, strength in mods:
        eff = min(1.0, max(0.0, strength * factor))
        block = mod.ref_block(eff)
        if block is not None:
            blocks.append(block)
    return blocks


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
                           "behavior). Lower values mix the ref toward noise — the model's own "
                           "conditioning-strength mechanism — so identity fades smoothly instead of "
                           "greying. 0 skips the mod entirely."})
        return {"required": required}

    RETURN_TYPES = ("H3_REF_MODS",)
    RETURN_NAMES = ("mods",)
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
        loads = []
        for i in range(1, self.MAX_SLOTS + 1):
            name = str(kwargs.get(f"mod_{i}", self.NONE))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or name == self.NONE or strength <= 0.0:
                continue
            loads.append((_load_mod(name), min(1.0, max(0.0, strength))))
        if loads:
            print("[MiniMaxH3RefModsLoader] " + ", ".join(
                f"{m.name}@{s:.2f}" for m, s in loads)
                + f" ({sum(m.token_count for m, _ in loads)} tokens total)")
        else:
            print("[MiniMaxH3RefModsLoader] no mods selected "
                  "(all slots (none) or strength 0)")
        if show_info:
            for mod, strength in loads:
                print("\n".join(_info_lines(mod)))
                print(f"  {'strength':<18} {strength:.2f}")
        return (loads,)


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

    RETURN_TYPES = ("H3_REF_MODS",)
    RETURN_NAMES = ("mods",)
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
        return (loads,)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModApply / ApplyCond
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModApply:
    """
    Inject a loader bundle of RefMods into a MiniMax-H3 conditioning.

    Appends each mod's reference latent to the conditioning's ``refs``, so the
    DiT attends to it through all blocks exactly like a reference image/video.
    ``retention`` is a master strength over the loader's per-row strengths.
    Works with the ComfyUI-MiniMaxH3 pack's MINIMAX_H3_COND output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("MINIMAX_H3_COND",),
                "mods": ("H3_REF_MODS",),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Master reference strength, multiplied with each loader row's "
                               "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                               "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                               "style/attributes, not identity), 0.15 = weak_reference. "
                               "0 = no reference."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/mod"

    def apply(self, conditioning, mods, retention=1.0):
        types_mod = _h3_pack_submodule("utils.types")
        if not isinstance(conditioning, types_mod.H3Conditioning):
            raise ValueError(
                "MiniMaxH3RefModApply expects the ComfyUI-MiniMaxH3 pack's "
                "MINIMAX_H3_COND. For the built-in ComfyUI CONDITIONING path "
                "use MiniMaxH3RefModApplyCond.")
        blocks = _ref_blocks(mods, retention)
        out = replace(conditioning, refs=list(conditioning.refs) + blocks)
        print(f"[MiniMaxH3RefModApply] retention={retention} "
              f"({len(blocks)} ref block(s) injected, {len(out.refs)} total)")
        return (out,)


class MiniMaxH3RefModApplyCond:
    """
    Inject a loader bundle of RefMods into a built-in ComfyUI CONDITIONING.

    Use this when conditioning comes from the core MiniMaxH3ReferenceToVideo
    node.  Appends to any existing ``minimax_refs``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mods": ("H3_REF_MODS",),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Master reference strength, multiplied with each loader row's "
                               "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                               "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                               "style/attributes, not identity), 0.15 = weak_reference. "
                               "0 = no reference."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/mod"

    def apply(self, conditioning, mods, retention=1.0):
        blocks = _ref_blocks(mods, retention)
        out = []
        for t in conditioning:
            d = dict(t[1])
            d["minimax_refs"] = list(d.get("minimax_refs", [])) + blocks
            out.append([t[0], d])
        print(f"[MiniMaxH3RefModApplyCond] retention={retention} "
              f"({len(blocks)} ref block(s) injected)")
        return (out,)


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
                    "tooltip": "Video frames kept (uniformly sampled; 240 = ~10s at 24fps)."}),
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
    def IS_CHANGED(cls, folder, max_items=32, max_frames=240):
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

    def load(self, folder, max_items=32, max_frames=240):
        folder = _resolve_folder(folder)
        images, videos = list_media_files(folder)
        items = (images + videos)[:max_items]
        refs, failed = [], []
        for p in items:
            try:
                if p in images:
                    refs.append(load_image_file(p))
                else:
                    refs.append(load_video_file(p, max_frames=max_frames))
            except Exception as exc:
                failed.append(f"{os.path.basename(p)} ({type(exc).__name__})")
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

      * ``full`` (default) — each ref is resized to ``ref_resolution`` short
        edge (down only) and VAE-encoded at that resolution, exactly like the
        official ref2video node.  The mod stores the real encode, so identity
        (a face, an outfit) comes through; files are ~0.2-1 MB per frame.
      * ``pooled`` — the latent is average-pooled to a tiny grid (4x4 = 4
        tokens per frame).  Nearly free to inject but only carries concept /
        motion, not fine identity.

    ``identity`` (pooled mode only) refines the small latent with a few
    gradient steps against the full encode — still no diffusion model.
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
                "own content instead of averaging away. 'pooled' mode (default) "
                "compresses the refs to a grid and refines it — good identity at "
                "a fraction of the tokens; 'full' stores the full-res encode "
                "(max identity, MB-size mod)."
            ),
            category="MiniMax-H3/mod",
            inputs=[
                io.String.Input("name", default="my_concept",
                    tooltip="Saved mod name (appears in the Load H3 RefMods dropdown after a reload)."),
                io.Combo.Input("mode", options=["full", "pooled"], default="pooled",
                    tooltip="'pooled' (default) = compressed grid refined by 'identity' — a good "
                            "balance of identity vs tokens. 'full' = full-res encode (max "
                            "identity, MB-size mod, ~1K tokens/img)."),
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
                io.Custom("MINIMAX_H3_AV_ENCODER").Input("av_encoder", optional=True,
                    tooltip="MiniMax-H3 VAE pack output (preferred; share the pack's VAE cache)."),
                io.Vae.Input("vae", optional=True,
                    tooltip="Standard VAE, used when av_encoder is not connected."),
                io.Int.Input("ref_resolution", default=1024, min=256, max=2048, step=64,
                    tooltip="Full mode: target short edge in px (downscale only, never upscale). "
                            "2048 = official max fidelity, 4x the tokens of 1024."),
                io.Int.Input("pool_h", default=16, min=2, max=16, step=2,
                    tooltip="Pooled mode: spatial latent grid after pooling (even). 16x16 = 256 tokens per frame."),
                io.Int.Input("pool_w", default=16, min=2, max=16, step=2),
                io.Int.Input("latent_frames", default=16, min=1, max=16,
                    tooltip="Frames kept per video ref: pooled mode pools them, full mode uniformly "
                            "samples them (16x16x16 = 4096 tokens per video ref). Images always use 1."),
                io.Int.Input("identity", default=500, min=0, max=2000, step=50,
                    tooltip="Pooled mode only: how tightly the mod clings to the reference "
                            "(gradient refinement steps). Higher = more identity detail but sticks "
                            "to the refs' framing/background; lower = deviates from the refs but "
                            "loses detail. 500 is a good default; 0 = pure pooling."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to mods/ so Load H3 RefMods can pick it up later."),
            ],
            outputs=[io.Custom("H3_REF_MODS").Output("mods",
                tooltip="Bundle with this one mod at strength 1.0. Feed it to Apply H3 RefMod "
                        "(or Load H3 RefMods after saving).")],
        )

    @classmethod
    def execute(cls, name, mode, refs_image=None, refs_video=None, refs_bundle=None,
                av_encoder=None, vae=None,
                ref_resolution=1024, pool_h=16, pool_w=16, latent_frames=16,
                identity=500, save=True, **legacy) -> io.NodeOutput:
        name = _sanitize_name(name)
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
        if mode == "full" and len(sources) > 1:
            h, w = sources[0][0].shape[1], sources[0][0].shape[2]
            scale = min(1.0, ref_resolution / min(h, w))
            canvas = (max(32, round(w * scale / 32) * 32),
                      max(32, round(h * scale / 32) * 32))
        frames = []
        n_img = n_vid = 0
        source_shapes = []
        for src_idx, (src, is_video) in enumerate(sources):
            if mode == "full":
                # downscale (never upscale) to the target short edge, sample
                # videos to latent_frames frames, then encode at full res
                if is_video and latent_frames < src.shape[0]:
                    idx = torch.linspace(0, src.shape[0] - 1, latent_frames).round().long()
                    src = src[idx]
                src = _resize_ref(src, ref_resolution, canvas)
            src = _ensure_min_size(src)
            if is_video and src.shape[0] > 1:
                valid_t = _snap_to_causal_grid(src.shape[0])
                if valid_t != src.shape[0]:
                    print(f"[MiniMaxH3RefModExtract] reference {src_idx + 1} "
                          f"(video): trimming {src.shape[0]} -> {valid_t} frames "
                          f"to match the VAE's causal 4k+1 grid.")
                    src = src[:valid_t]
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

            if mode == "full":
                pooled = z.to(torch.float16)
            else:
                pool_t = min(latent_frames, z.shape[2]) if is_video else 1
                pooled = pool_latent(z, pool_t, pool_h, pool_w).to(torch.float16)
                if identity > 0:
                    pooled = optimize_latent(pooled, z.float(), steps=int(identity))
            frames.append(pooled)
            if is_video:
                n_vid += 1
            else:
                n_img += 1

        if mode == "full" and identity > 0:
            print(f"[MiniMaxH3RefModExtract] warning: 'identity' only applies to "
                  f"pooled mode — full mode stores the actual encode, so "
                  f"identity={identity} was ignored.")

        latent = torch.cat(frames, dim=2)  # [1, 24, total_t, h, w]
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
            pool=f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "full" else f"{total_t}x{pool_h}x{pool_w}",
            optimize_steps=int(identity) if mode == "pooled" else 0,
            tags=[f"{n_img} img, {n_vid} vid"],
        )

        if save:
            path = mod.save(os.path.join(refmods_dir(), name))
            _MOD_CACHE[name] = mod
            print(f"[MiniMaxH3RefModExtract] saved {_summarize(mod)} -> {path}")
        else:
            print(f"[MiniMaxH3RefModExtract] {_summarize(mod)} (not saved)")
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
    "MiniMaxH3RefModApplyCond": MiniMaxH3RefModApplyCond,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefModExtract": "Extract H3 RefMod",
    "MiniMaxH3RefModFolderLoader": "Load H3 RefMod Folder",
    "MiniMaxH3RefModsLoader": "Load H3 RefMods",
    "MiniMaxH3RefModsAxis": "Load H3 RefMod Axis",
    "MiniMaxH3RefModApply": "Apply H3 RefMod",
    "MiniMaxH3RefModApplyCond": "Apply H3 RefMod (Cond)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
