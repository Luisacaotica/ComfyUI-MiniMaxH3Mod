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
  MiniMaxH3RefModStepCurve     — per-denoising-step ref strength envelope (MODEL -> MODEL)
  MiniMaxH3RefModConfig        — fix tuned Apply/Step-Curve settings into a mod's metadata so
                                 the Apply/Step Curve ``override`` toggle can reuse them

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

import itertools
from contextvars import ContextVar
import json
import math
import ntpath
import os
import random
from dataclasses import replace
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy_api.latest import io
from comfy_execution.validation import validate_node_input
from comfy_extras.nodes_audio import vae_decode_audio
from .library import register_routes

from .common import (
    list_media_files,
    load_image_file,
    load_video_file,
    refmods_dir,
    mod_output_path,
    resize_ref as _resize_ref,
    snap_to_causal_grid as _snap_to_causal_grid,
    ensure_min_size as _ensure_min_size,
)
from . import continuum_bridge
from .audio import make_audio_mod
from .refmod_profile import ProfileRefMod
from .refmod_native import prepare_clip, reference_map, validate_apply, APPLIED_KEY
from .core import (
    CONCEPT_TYPES,
    CURVE_DIRECTIONS,
    CURVE_SHAPES,
    H3RefMod,
    _blur_latent,
    aspect_grid,
    curve_strengths,
    curve_value_at,
    fit_token_budget,
    normalize_mode,
    optimize_latent,
    optimize_latent_multi,
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
_MOD_CACHE_STAMPS = {}
_MOD_CACHE_BYTES = 256 * 1024 * 1024
_MOD_CACHE_MAX = 24          # cap: never pin more mods in RAM than this (FIFO eviction)
_MOD_LIST_CACHE_KEY = None   # (dirs, mtimes, sizes) signature of the last _list_mod_names() scan
_MOD_LIST_CACHE_VAL = None
_MOD_SKIP_DIRS = {"graph_presets", ".git", "__pycache__"}

# Mod storage lives in ComfyUI's models/ tree (created on first run) and is
# registered as a first-class folder type so it shows up next to loras/unet.
try:
    folder_paths.add_model_folder_path("refmods", os.path.join(folder_paths.models_dir, "refmods"))
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

def _resolve_visual_vae(vae=None, av_encoder=None):
    """Use native ComfyUI VAE loading for legacy VAERef connections too."""
    if vae is not None:
        return vae
    if av_encoder is None:
        raise ValueError("Connect the MiniMax H3 video VAE to vae.")
    import comfy.sd
    path = os.fspath(av_encoder.video_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"H3 video VAE not found: {path}")
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    loaded = comfy.sd.VAE(sd=state, metadata=metadata)
    loaded.throw_exception_if_invalid()
    return loaded


# ═══════════════════════════════════════════════════════════════════════════
# Mods folder helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mod_search_dirs() -> List[str]:
    dirs = list(folder_paths.get_folder_paths("refmods"))
    for d in (os.path.join(folder_paths.models_dir, "refmods"), os.path.join(folder_paths.models_dir, "mods"),
              os.path.join(os.path.dirname(_PACK_DIR), "models", "mods"), LEGACY_MODS_DIR,
              os.path.join(folder_paths.models_dir, "audio_refmods")):
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
    dirs = _mod_search_dirs()
    sig, candidates = [], []
    for d in dirs:
        for root, subdirs, files in os.walk(d):
            subdirs[:] = sorted(s for s in subdirs if s not in _MOD_SKIP_DIRS)
            for fn in sorted(files):
                if not fn.endswith(".safetensors"):
                    continue
                path = os.path.join(root, fn)
                stem = path[:-len(".safetensors")]
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                sidecar = None
                try:
                    js = os.stat(stem + ".json")
                    sidecar = (js.st_size, js.st_mtime_ns)
                except OSError:
                    pass
                sig.append((path, st.st_size, st.st_mtime_ns, sidecar))
                name = os.path.relpath(stem, d).replace("\\", "/")
                candidates.append((name, stem))
    key = (tuple(dirs), tuple(sig))
    if key == _MOD_LIST_CACHE_KEY:
        return _MOD_LIST_CACHE_VAL
    names = set()
    seen = set()
    for name, stem in candidates:
        if name in seen:
            continue
        seen.add(name)  # match _find_mod_path's first-search-directory priority
        meta = read_refmod_meta(stem)
        if isinstance(meta, dict) and meta.get("kind") in ("image", "video", "audio", "character"):
            names.add(name)
    _MOD_LIST_CACHE_KEY, _MOD_LIST_CACHE_VAL = key, sorted(names)
    return _MOD_LIST_CACHE_VAL


def _normalize_mod_name(name) -> str:
    if name is None:
        return ""
    name = str(name).replace("\\", "/")
    return "" if name in ("", "None", "(none)") else name


def _find_mod_path(name: str) -> str:
    name = _normalize_mod_name(name)
    if (not name or ntpath.splitdrive(name)[0] or name.startswith("/")
            or ".." in name.split("/")
            or any(part in _MOD_SKIP_DIRS for part in name.split("/"))):
        raise ValueError(f"RefMod '{name}': expected a relative mod name under models/refmods/.")
    dirs = _mod_search_dirs()
    for d in dirs:
        p = os.path.join(d, name)
        root = os.path.realpath(d)
        target = os.path.realpath(p + ".safetensors")
        try:
            contained = os.path.commonpath((root, target)) == root
        except ValueError:  # a symlink can point to another Windows drive
            contained = False
        if not contained:
            continue
        if os.path.isfile(p + ".safetensors"):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"RefMod '{name}' not found. Searched:\n" +
        "\n".join(f"  - {d}/{name}.safetensors" for d in dirs))


def _file_stamp(path):
    stamps = []
    for ext in (".safetensors", ".json"):
        try:
            st = os.stat(path + ext)
            stamps.append((st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        except FileNotFoundError:
            stamps.append(None)
    return (os.path.normcase(os.path.abspath(path)), *stamps)


def _cache_mod(path, mod):
    key = os.path.normcase(os.path.abspath(path))
    _MOD_CACHE[key] = mod
    _MOD_CACHE_STAMPS[key] = _file_stamp(path)
    while _MOD_CACHE and (len(_MOD_CACHE) > _MOD_CACHE_MAX or
            sum(m.storage_bytes for m in _MOD_CACHE.values()) > _MOD_CACHE_BYTES):
        evicted = next(iter(_MOD_CACHE))
        _MOD_CACHE.pop(evicted)
        _MOD_CACHE_STAMPS.pop(evicted, None)


def _load_mod(name: str) -> H3RefMod:
    path = _find_mod_path(name)
    key = os.path.normcase(path)
    if key in _MOD_CACHE and _MOD_CACHE_STAMPS.get(key) == _file_stamp(path):
        return _MOD_CACHE[key]
    _MOD_CACHE.pop(key, None)
    _MOD_CACHE_STAMPS.pop(key, None)
    mod = H3RefMod.load(path, device="cpu")
    _cache_mod(path, mod)
    return mod


def _mods_changed(kwargs):
    stamps = []
    for field, value in sorted(kwargs.items()):
        if not field.startswith("mod_"):
            continue
        if value is None:
            return float("nan")
        name = _normalize_mod_name(value)
        if name:
            try:
                stamps.append(_file_stamp(_find_mod_path(name)))
            except (FileNotFoundError, ValueError):
                stamps.append((name, "missing"))
    return tuple(stamps)


def _check_token_budget(mods, budget):
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or int(budget) != budget:
        raise ValueError("Token budget must be a finite integer.")
    total = sum(mod.token_count for mod, strength in mods if strength > 0)
    if budget < 0:
        raise ValueError("Token budget cannot be negative.")
    if budget and total > budget:
        raise ValueError(f"RefMods require {total} tokens after copies; budget is {budget}. Reduce copies or selected mods.")
    return total


def _number_error(field, value, kind, options):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"RefMod input '{field}': expected a finite {kind}."
    if kind == "INT" and int(value) != value:
        return f"RefMod input '{field}': expected an integer."
    if value < options["min"] or value > options["max"]:
        return f"RefMod input '{field}': expected {options['min']}..{options['max']}, got {value}."
    return None


def _check_loader_numbers(required, kwargs):
    for field, (kind, options) in required.items():
        if kind in ("INT", "FLOAT") and field in kwargs:
            error = _number_error(field, kwargs[field], kind, options)
            if error:
                raise ValueError(error)



def _validate_mod_inputs(required, input_types, kwargs):
    linked = input_types or {}
    for field, (expected, _options) in required.items():
        is_mod = isinstance(expected, list)
        if field in linked:
            received = linked[field]
            if isinstance(received, list):
                received = "COMBO"
            allowed = "STRING,COMBO" if is_mod else expected
            if not validate_node_input(received, allowed):
                return f"RefMod input '{field}': expected {allowed}, got {received}."
            continue  # upstream values are resolved at execution, not queue time
        if expected in ("INT", "FLOAT") and field in kwargs and kwargs[field] is not None:
            error = _number_error(field, kwargs[field], expected, _options)
            if error:
                return error
        if is_mod:
            name = _normalize_mod_name(kwargs.get(field))
            if name and name not in expected:
                return (f"RefMod input '{field}': '{name}' not found in models/refmods/ "
                        "or legacy mods/ folders. Run Extract H3 RefMod first.")
    return True


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


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int, crop="disabled") -> torch.Tensor:
    """Resize a ``[T, H, W]`` mask to ``target_h x target_w`` (bilinear)."""
    samples = mask.unsqueeze(1)  # [T, 1, H, W]
    samples = comfy.utils.common_upscale(samples, target_w, target_h, "bilinear", crop)
    return samples.squeeze(1).clamp(0.0, 1.0)


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
    if isinstance(mod, ProfileRefMod):
        return mod.report()
    if mod.kind == "audio":
        return f"{mod.name}: audio, {mod.latent_t / 40:.2f}s, {mod.token_count} tokens"
    mb = mod.latent.numel() * mod.latent.element_size() / 1024 / 1024
    return (f"'{mod.name}' {mod.mode} {mod.kind} {tuple(mod.latent.shape)} "
            f"({mod.token_count} tokens, {mb:.2f} MB)")


def _info_lines(mod: H3RefMod) -> List[str]:
    if isinstance(mod, ProfileRefMod):
        return [mod.report(), f"  Saved: {mod.path}.safetensors"]
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


def _saved_curve(cfg: Optional[Dict], key: str) -> Optional[tuple]:
    """(direction, shape, value) from a mod's saved config entry, or None.

    Validates against the known curve names so hand-edited/foreign metadata
    can't inject junk into the Apply or Step Curve nodes — anything invalid
    just falls back to the manual widgets.
    """
    entry = (cfg or {}).get(key)
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        return None
    direction, shape, value = entry
    if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
        return None
    try:
        return (direction, shape, float(value))
    except (TypeError, ValueError):
        return None


def _ref_blocks(mods, retention, curve=None, seed=-1, scramble_mode="legacy_subset", scramble_keep=1, max_total_tokens=0) -> List[Dict]:
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
        if scramble_mode == "legacy_subset":
            keep = rng.randint(max(1, len(items) // 2), len(items))
            items = items[:keep]
        elif scramble_mode == "subset":
            items = items[:max(1, int(scramble_keep))]
        elif scramble_mode != "shuffle":
            raise ValueError("Unknown scramble mode.")
        print(f"[MiniMaxH3RefModApply] scramble seed={int(seed)}: "
              f"{len(mods)} refs -> kept {len(items)} (order shuffled)")
    _check_token_budget(items, max_total_tokens)
    blocks = []
    notes = []
    summaries = []  # (name, effective average multiplier)
    for mod, strength in items:
        eff = min(1.0, max(0.0, strength * factor))
        if isinstance(mod, ProfileRefMod):
            blocks.extend(mod.ref_blocks(eff, curve))
            summaries.append((mod.name, eff))
            continue
        used_curve = curve
        if (mod.latent_t <= 1 and isinstance(curve, tuple) and len(curve) == 3
                and isinstance(curve[0], str) and str(curve[0]) != "constant"):
            # curve directions run across a mod's own ref frames, which is
            # meaningless on single-frame mods — previously this silently
            # no-op'ed and identity mods always injected at full strength.
            # Now: treat curve_value as a plain strength cap instead.
            cap = max(0.0, min(1.0, float(curve[2])))
            if cap < eff:
                notes.append(
                    f"'{mod.name}' is image-kind (1 frame): direction "
                    f"'{curve[0]}' has no effect; using curve_value "
                    f"{cap:.2f} as its strength cap instead")
                eff = min(eff, cap)
            used_curve = None
        block = mod.ref_block(eff, curve=used_curve)
        if block is not None:
            # marker the step-curve wrapper uses to tell this ref apart from
            # native ref2va refs (original input video, keyframes, ...) so it
            # only ever re-mixes what the Apply node injected
            block["refmod"] = True
            blocks.append(block)
        avg = eff
        if used_curve is not None and mod.latent_t > 1:
            strengths = curve_strengths(used_curve, mod.latent_t)
            if strengths:
                avg = eff * (sum(strengths) / len(strengths))
        summaries.append((mod.name, avg))
    for line in notes:
        print(f"[MiniMaxH3RefModApply] note: {line}")
    if summaries and any(s < 0.999 for _n, s in summaries):
        detail = ", ".join(f"{n}@{s:.2f}" for n, s in summaries)
        weak = min(s for _n, s in summaries) < 0.3
        print(f"[MiniMaxH3RefModApply] effective ref strength (row x retention"
              f"{' x curve-mean' if curve is not None else ''}): {detail}"
              + ("  <- below 0.30, expect a weak insertion" if weak else ""))
    return blocks


_STEP_WRAPPER_SEQ = itertools.count()


def _make_step_wrapper(spec, progress=None) -> Callable:
    """Mix only marked refs in this forward; never retain or mutate a payload."""
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs.get("minimax_payload")
        if not payload or not any(r.get("refmod") for r in payload.get("refs", [])):
            return executor(x, timestep, context, transformer_options, **kwargs)
        start = progress.get() if progress is not None else 1000.0
        sigma = min(1.0, max(0.0, float(timestep.flatten()[0]) / max(start, 1e-8)))
        strength = min(1.0, max(0.0, curve_value_at(spec, 1.0 - sigma)))
        if strength >= 1.0:
            return executor(x, timestep, context, transformer_options, **kwargs)
        mixed = dict(payload)
        for field, latent_key in (("cond_video_latents", "latent"),
                                  ("cond_audio_latents", "audio_latent")):
            cond = payload.get(field)
            if not cond:
                continue
            out = list(cond)
            index = sum(k.get(latent_key) is not None for k in payload.get("keyframes", []))
            for ref in payload.get("refs", []):
                if ref.get(latent_key) is None:
                    continue
                if index >= len(out):
                    raise ValueError("RefMod Step Curve: reference layout does not match conditioning latents.")
                if ref.get("refmod"):
                    z = out[index]
                    out[index] = strength * z + (1.0 - strength) * _blur_latent(z)
                index += 1
            mixed[field] = out
        return executor(x, timestep, context, transformer_options,
                        **dict(kwargs, minimax_payload=mixed))
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
        optional = {"max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576}),
                    "clip": ("CLIP", {"tooltip": "For combined character files: connect CLIP Loader here, then this node's clip output to the official MiniMax H3 Reference to Video."})}
        for i in range(1, cls.MAX_SLOTS + 1):
            optional[f"voice_reference_{i}"] = ("INT", {"default": 1, "min": 0, "max": 64,
                "tooltip": "Saved voice example for this character: 1 = first (default), 2 = second, 0 = all. Selection does not alter the file."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("H3_REF_MODS", "STRING", "CLIP")
    RETURN_NAMES = ("mods", "prompt_hint", "clip")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/mod"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        schema = cls.INPUT_TYPES()
        return _validate_mod_inputs({**schema["required"], **schema.get("optional", {})}, input_types, kwargs)

    @classmethod
    def IS_CHANGED(cls, show_info=False, **kwargs):
        return _mods_changed(kwargs)

    def load(self, show_info=False, max_total_tokens=0, clip=None, **kwargs):
        _check_loader_numbers(self.INPUT_TYPES()["required"], kwargs)
        rows = []  # (mod, strength, copies)
        for i in range(1, self.MAX_SLOTS + 1):
            name = _normalize_mod_name(kwargs.get(f"mod_{i}"))
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            if not name or strength <= 0.0:
                continue
            mod = _load_mod(name)
            if isinstance(mod, ProfileRefMod):
                mod = replace(mod, voice_reference=kwargs.get(f"voice_reference_{i}", 1), subject_slot=i)
                print("[RefMod character] " + mod.report())
            rows.append((mod, min(1.0, max(0.0, strength)),
                         int(kwargs.get(f"copies_{i}", 1))))
        loads = []
        for mod, strength, copies in rows:
            loads.extend([(mod, strength)] * copies)
        _check_token_budget(loads, max_total_tokens)
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
        mapping = reference_map(loads)
        if mapping:
            hint = mapping + ("\n" + hint if hint else "")
            print("[RefMod reference map]\n" + mapping)
        return (loads, hint, prepare_clip(clip, loads))


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
        return {"required": required, "optional": {"max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576})}}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/mod"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        schema = cls.INPUT_TYPES()
        return _validate_mod_inputs({**schema["required"], **schema.get("optional", {})}, input_types, kwargs)

    @classmethod
    def IS_CHANGED(cls, show_info=False, **kwargs):
        return _mods_changed(kwargs)

    def load(self, show_info=False, max_total_tokens=0, **kwargs):
        _check_loader_numbers(self.INPUT_TYPES()["required"], kwargs)
        loads = []
        for i in range(1, self.MAX_SLOTS + 1):
            value = float(kwargs.get(f"value_{i}", 0.0))
            if abs(value) < 1e-6:
                continue
            side = "b" if value > 0 else "a"
            name = _normalize_mod_name(kwargs.get(f"mod_{side}_{i}"))
            if not name:
                continue
            loads.append((_load_mod(name), min(1.0, abs(value))))
        _check_token_budget(loads, max_total_tokens)
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
                io.Boolean.Input("override", default=False,
                    tooltip="Use the config fixed into the mods' own metadata (by 'Fix H3 RefMod "
                            "Config') instead of the widgets below: retention + curve come from "
                            "the first mod in the bundle that carries one. Handy for sharing mods "
                            "whose magic settings took real tuning. Off (default) = use the manual "
                            "parameters. If no mod has a saved config it falls back to the manual "
                            "parameters and prints a note."),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Master reference strength, multiplied with each loader row's "
                             "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                             "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                             "style/attributes, not identity), 0.15 = weak_reference. "
                             "0 = no reference."),
                io.Combo.Input("curve_direction", options=list(CURVE_DIRECTIONS),
                    default="constant",
                    tooltip="Weighting envelope across THIS MOD'S OWN ref frames "
                            "(stacked images / video-ref latent frames) — i.e. WHICH "
                            "reference content dominates, NOT where the concept "
                            "appears in the output video (ref tokens are not bound "
                            "to output time; for output-timing control use the 'H3 "
                            "RefMod Step Curve' node instead, which runs over the "
                            "denoise timeline). 'constant' (default) = every ref "
                            "frame at full strength (official-ref parity). The old "
                            "default 'concept_at_end' fades early stack frames toward "
                            "blur, roughly HALVING average strength on multi-frame "
                            "mods. Old saved workflows keep their saved values."),
                io.Int.Input("scramble_seed", default=-1, min=-1, max=2147483647, step=1,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Ref scrambling seed. -1 (default) = off: all refs in saved order. "
                            "With 2+ refs in the bundle, a seed >= 0 shuffles the ref order and "
                            "keeps a random subset, so a different ref leads each run (a multi-ref "
                            "mod 'pops' a different video/image per seed). Same seed = same "
                            "scramble; set this widget's control-after-generate to 'randomize' "
                            "for per-run variation."),
                io.Combo.Input("curve_shape", options=list(CURVE_SHAPES),
                    default="linear",
                    tooltip="How the weighting travels between its endpoints: 'linear', "
                            "'ease' (smoothstep), 'sigmoid'/'tanh' (S-curves, tanh with a "
                            "steeper knee), 'quadratic', 'cubic', 'exponential', 'stair' "
                            "(stepped), 'elastic' (overshoots), 'bump'/'dip' (peak/trough "
                            "mid-stack). Only matters when curve_direction != constant."),
                io.Float.Input("curve_value", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Endpoint weight ('user input'): both endpoints for 'constant' and "
                            "'concept_at_ends', the start for 'concept_at_start', the end for "
                            "'concept_at_end', the mid peak for 'concept_at_middle'. On "
                            "single-image ('image'-kind) mods this acts as a simple STRENGTH CAP "
                            "(directions are meaningless on one frame): 0.4 = the ref blends "
                            "40% toward its blurred self."),
                io.Combo.Input("graph_preset",
                    options=["(none)"] + _list_graph_presets(), default="(none)",
                    optional=True,
                    tooltip="Optional shared graph preset — leave on '(none)' to use the curve "
                            "widgets above. Selecting one loads direction/shape/value from a "
                            "saved debug-grid PNG (graph embedded in its metadata) or a legacy "
                            ".json, in models/refmods/graph_presets/. Share the preset PNG "
                            "itself to share a curve. New presets appear after a restart."),
                io.Combo.Input("scramble_mode", options=["shuffle", "subset", "legacy_subset"], default="shuffle", optional=True),
                io.Int.Input("scramble_keep", default=1, min=1, max=80, optional=True, tooltip="Refs retained in subset mode; shuffle keeps all refs."),
                io.Int.Input("max_total_tokens", default=0, min=0, max=1048576, optional=True, tooltip="Total reference token budget after copies; 0 disables the limit."),
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
                curve_direction="constant", curve_shape="linear", curve_value=1.0,
                strength_curve=None, scramble_seed=-1, graph_preset="", save_preset_as="",
                override=False, scramble_mode="legacy_subset", scramble_keep=1, max_total_tokens=0):
        native_profiles = validate_apply(conditioning, mods, scramble_seed)
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
        # override: pull retention + curve from the first mod with a fixed config
        if override:
            found = None
            for m, _s in (mods or []):
                saved = _saved_curve(getattr(m, "config", None), "curve")
                if saved is not None:
                    found = (m, saved)
                    break
            if found is None:
                print("[MiniMaxH3RefModApply] override=True but no mod in the bundle "
                      "has a saved config — using the manual parameters.")
            else:
                m, saved = found
                cfg = getattr(m, "config", None) or {}
                curve = saved
                if isinstance(cfg.get("retention"), (int, float)):
                    retention = min(1.0, max(0.0, float(cfg["retention"])))
                print(f"[MiniMaxH3RefModApply] override: using config from '{m.name}' "
                      f"(retention={retention:.2f}, curve={curve[0]} + {curve[1]} "
                      f"@ {float(curve[2]):.2f})")
        if native_profiles:
            factor = RETENTION.get(retention, 1.0) if isinstance(retention, str) else float(retention)
            if not math.isfinite(factor) or factor <= 0:
                raise ValueError("Combined RefMods require retention > 0 after reference labels are encoded. "
                                 "To disable a character, set its strength to 0 in Load H3 RefMods instead.")
        img = render_debug_grid(curve, preset_name)
        if save_preset_as:
            saved = _save_graph_preset(save_preset_as, curve, img)
            if saved:
                print(f"[MiniMaxH3RefModApply] graph preset saved: {saved}.png "
                      f"({curve[0]} + {curve[1]} @ {float(curve[2]):.2f})")
        blocks = _ref_blocks(mods, retention, curve, seed=scramble_seed,
                             scramble_mode=scramble_mode, scramble_keep=scramble_keep, max_total_tokens=max_total_tokens)
        if isinstance(conditioning, list):
            # built-in ComfyUI CONDITIONING (core MiniMaxH3ReferenceToVideo)
            out = []
            for t in conditioning:
                d = dict(t[1])
                if native_profiles:
                    profile_blocks = [b for b in blocks if b.get("refmod_profile")]
                    order = {"image": 0, "video": 1, "video_audio": 1, "audio": 2}
                    profile_blocks.sort(key=lambda b: 0 if b.get("refmod_original_visual") else order[b["kind"]])
                    legacy_blocks = [b for b in blocks if not b.get("refmod_profile")]
                    d["minimax_refs"] = profile_blocks + list(d.get("minimax_refs", [])) + legacy_blocks
                    d[APPLIED_KEY] = True
                else:
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
                "override": ("BOOLEAN", {"default": False,
                    "tooltip": "Use the step_curve fixed into a mod's own metadata (by 'Fix H3 "
                               "RefMod Config') instead of the curve widgets. Needs the mods "
                               "bundle connected to the optional 'mods' input below. Off (default) "
                               "= use the manual widgets. If no mod carries a saved step_curve it "
                               "falls back to the manual widgets and prints a note."}),
                "curve_direction": (list(CURVE_DIRECTIONS), {"default": "constant",
                    "tooltip": "Strength envelope over the DENOISE timeline (the one "
                               "temporal axis refs actually respond to): 'constant' "
                               "(default) = full ref at every step — pure passthrough, "
                               "nothing is re-mixed. 'concept_at_start' holds refs at "
                               "full strength in the early steps (high sigma — "
                               "composition and identity set first) and releases them "
                               "toward the final steps (clean texture, no ref grain); "
                               "'concept_at_end' opens weak and locks full strength in "
                               "the late steps (identity detail refined at the very end "
                               "of denoising); 'concept_at_middle' peaks mid-denoise; "
                               "'concept_at_ends' holds the extremes and dips "
                               "mid-denoise. Old 'decrease'/'increase' values still "
                               "resolve."}),
                "curve_shape": (list(CURVE_SHAPES), {"default": "linear",
                    "tooltip": "How the per-step strength travels between its endpoints "
                               "(linear / ease / sigmoid / tanh / quadratic / cubic / "
                               "exponential / stair / elastic / bump / dip). Only matters "
                               "when curve_direction != constant."}),
                "curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Endpoint strength ('user input'): both endpoints for 'constant' and "
                               "'concept_at_ends', the start for 'concept_at_start', the end for "
                               "'concept_at_end', the mid peak for 'concept_at_middle'. "
                               "1.0 = full ref there."}),
            },
            "optional": {
                "mods": ("H3_REF_MODS", {
                    "tooltip": "Optional bundle, only read when 'override' is on: the first mod "
                               "carrying a saved step_curve config supplies this node's curve. "
                               "Leave unconnected when override is off."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/mod"

    def apply(self, model, mods=None, override=False,
              curve_direction="constant", curve_shape="linear", curve_value=1.0):
        curve = (curve_direction, curve_shape, curve_value)
        if override:
            found = None
            for m, _s in (mods or []):
                saved = _saved_curve(getattr(m, "config", None), "step_curve")
                if saved is not None:
                    found = (m, saved)
                    break
            if found is None:
                print("[MiniMaxH3RefModStepCurve] override=True but no mod in the "
                      "bundle has a saved step_curve config — using the manual "
                      "parameters.")
            else:
                m, curve = found
                print(f"[MiniMaxH3RefModStepCurve] override: using step_curve from "
                      f"'{m.name}' ({curve[0]} + {curve[1]} @ {float(curve[2]):.2f})")
        model = model.clone()
        # unique key per attach: chained Step Curve nodes each get their own
        # wrapper entry (same key would append into the same list slot and
        # double-mix); a no-op curve still attaches harmlessly
        key = f"minimax_h3_refmod_step_curve_{next(_STEP_WRAPPER_SEQ)}"
        progress = ContextVar(key, default=1000.0)
        def schedule(executor, noise, latent_image, sampler, sigmas, *args, **kwargs):
            sampling = executor.class_obj.model_patcher.get_model_object("model_sampling")
            start = float(sampling.timestep(sigmas[0]))
            token = progress.set(start)
            try:
                return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)
            finally:
                progress.reset(token)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key, schedule)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            key, _make_step_wrapper(curve, progress))
        print(f"[MiniMaxH3RefModStepCurve] {curve[0]} + {curve[1]} "
              f"@ {float(curve[2]):.2f} attached — refs re-mixed per denoising step")
        return (model,)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModConfig (fix tuned settings into the mod file)
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModConfig:
    """Fix tuned Apply / Step-Curve settings into a mod's safetensors metadata.

    Every concept has its own "magic settings" — retention + curve on Apply,
    a step curve on H3 RefMod Step Curve — and figuring them out is a
    learning curve of its own.  This node bakes those values into the mod
    file's metadata (re-saving it in place), so a creator tunes once and
    ships the config with the mod.  Newbies then just flip the ``override``
    toggle on Apply / Step Curve and get the exact settings the concept was
    built with, without touching a dial.  A mod without a config falls back
    to the manual widgets (with a console note).

    Wire ``mods`` through this node (Loader -> Config -> Apply); the bundle
    is returned unchanged apart from the config attached to each mod object
    (and written to the file, so it also survives a restart / sharing the
    .safetensors).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mods": ("H3_REF_MODS", {
                    "tooltip": "The mods to fix a config into (e.g. from Load H3 RefMods). "
                               "Each mod's .safetensors is re-saved with the settings below "
                               "embedded in its metadata; the bundle passes through unchanged."}),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "number",
                    "tooltip": "Master reference strength for Apply H3 RefMod — the value that "
                               "makes THIS concept work (1.0 = fully_preserved, 0.7 = "
                               "partially_preserved, 0.4 = attribute_transfer, 0.15 = "
                               "weak_reference)."}),
                "curve_direction": (list(CURVE_DIRECTIONS), {"default": "concept_at_end",
                    "tooltip": "The Apply frame-curve direction this concept needs (same list as "
                               "Apply H3 RefMod)."}),
                "curve_shape": (list(CURVE_SHAPES), {"default": "ease",
                    "tooltip": "The Apply frame-curve shape this concept needs."}),
                "curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "number",
                    "tooltip": "Apply curve endpoint value (1.0 = full strength there)."}),
                "step_curve_direction": (list(CURVE_DIRECTIONS), {"default": "concept_at_end",
                    "tooltip": "The H3 RefMod Step Curve direction this concept needs (over the "
                               "denoise timeline)."}),
                "step_curve_shape": (list(CURVE_SHAPES), {"default": "ease",
                    "tooltip": "The Step Curve shape this concept needs."}),
                "step_curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "number",
                    "tooltip": "Step Curve endpoint value (1.0 = full ref there)."}),
            },
        }

    RETURN_TYPES = ("H3_REF_MODS",)
    RETURN_NAMES = ("mods",)
    FUNCTION = "fix"
    CATEGORY = "MiniMax-H3/mod"

    def fix(self, mods, retention=1.0, curve_direction="concept_at_end",
            curve_shape="ease", curve_value=1.0,
            step_curve_direction="concept_at_end", step_curve_shape="ease",
            step_curve_value=1.0):
        cfg = {
            "retention": min(1.0, max(0.0, float(retention))),
            "curve": [curve_direction, curve_shape,
                      min(1.0, max(0.0, float(curve_value)))],
            "step_curve": [step_curve_direction, step_curve_shape,
                           min(1.0, max(0.0, float(step_curve_value)))],
        }
        out = []
        saved_paths = set()
        for item in mods:
            mod, strength = item if isinstance(item, tuple) else (item, 1.0)
            mod.config = dict(cfg)
            path = getattr(mod, "path", "") or os.path.join(refmods_dir(), mod.name)
            if path not in saved_paths:
                mod.save(path)
                saved_paths.add(path)
            mod.path = path
            _cache_mod(path, mod)
            out.append((mod, strength))
            print(f"[MiniMaxH3RefModConfig] '{mod.name}': config fixed into "
                  f"metadata (retention={cfg['retention']:.2f}, "
                  f"curve={curve_direction} + {curve_shape} @ {float(curve_value):.2f}, "
                  f"step_curve={step_curve_direction} + {step_curve_shape} "
                  f"@ {float(step_curve_value):.2f})")
        if out:
            global _MOD_LIST_CACHE_KEY
            _MOD_LIST_CACHE_KEY = None  # files changed -> refresh listings
        return (out,)


# ═══════════════════════════════════════════════════════════════════════════
# Node: MiniMaxH3RefModContinuumBridge (H3-Continuum integration)
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModContinuumBridge:
    """Attach refs to a cloned MODEL for every sampling call, including Continuum chunks.

    Uses the public OUTER_SAMPLE hook; no global state or Continuum monkey patch.
    Disabling removes this bridge from the clone. Original conditioning is restored
    on completion, exception or cancellation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Cloned model with scoped refs — wire Load Model -> this "
                               "-> Continuum Sampler 'model'. This link is what makes the "
                               "bridge execute before sampling."}),
                "mods": ("H3_REF_MODS", {
                    "tooltip": "Bundle from Load H3 RefMods / Extract H3 RefMod — injected "
                               "into every Continuum chunk while 'enable' is on."}),
                "enable": ("BOOLEAN", {"default": True,
                    "tooltip": "Master switch for the injection. Off = pure passthrough "
                               "(this bridge is removed from the cloned model)."}),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "number",
                    "tooltip": "Master reference strength (same scale as Apply H3 RefMod): "
                               "1.0 = fully_preserved, 0.7 = partially_preserved, "
                               "0.4 = attribute_transfer, 0.15 = weak_reference."}),
                "curve_direction": (list(CURVE_DIRECTIONS), {"default": "constant",
                    "tooltip": "Weighting across each mod's own ref frames (NOT output-video "
                               "time): 'constant' keeps every frame at full strength. Other "
                               "directions fade part of the stack toward blur — mainly useful "
                               "for multi-ref mods."}),
                "curve_shape": (list(CURVE_SHAPES), {"default": "linear",
                    "tooltip": "How that weighting travels between endpoints."}),
                "curve_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "number",
                    "tooltip": "Endpoint weight; on single-image mods this acts as a plain "
                               "strength cap."}),
                "scramble_seed": ("INT", {"default": -1, "min": -1, "max": 2147483647,
                    "step": 1,
                    "tooltip": "-1 = off (all refs, saved order). >= 0 shuffles the bundle "
                               "and keeps a subset, re-drawn on every queued run when the "
                               "widget's control-after-generate is set to 'randomize'."}),
            },
            "optional": {
                "scramble_mode": (["shuffle", "subset", "legacy_subset"], {"default": "shuffle"}),
                "scramble_keep": ("INT", {"default": 1, "min": 1, "max": 80}),
                "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576}),
            },
        }

    RETURN_TYPES = ("MODEL", "H3_REF_MODS", "STRING")
    RETURN_NAMES = ("model", "mods", "status")
    FUNCTION = "arm"
    CATEGORY = "MiniMax-H3/mod"

    def arm(self, model, mods, enable=True, retention=1.0,
            curve_direction="constant", curve_shape="linear", curve_value=1.0,
            scramble_seed=-1, scramble_mode="legacy_subset", scramble_keep=1, max_total_tokens=0):
        if enable and any(isinstance(mod, ProfileRefMod) and strength > 0 for mod, strength in mods):
            raise ValueError("Combined appearance/voice RefMods need the official Reference to Video "
                             "CLIP preparation and Apply H3 RefMod. The Continuum bridge does not prepare reference labels.")
        flat_curve = (curve_direction == "constant"
                      and curve_shape == "linear" and curve_value >= 1.0)
        curve = None if flat_curve else (curve_direction, curve_shape, curve_value)
        blocks = _ref_blocks(mods, retention, curve, seed=scramble_seed,
                             scramble_mode=scramble_mode, scramble_keep=scramble_keep, max_total_tokens=max_total_tokens)
        model = continuum_bridge.attach(model, blocks, enable)
        status = "model-scoped bridge active" if enable else "bridge disabled"
        print(f"[MiniMaxH3RefModContinuumBridge] {status} "
              f"({len(blocks)} block(s) prepared)")
        return (model, mods, status)


class MiniMaxH3RefModBridgeDisarm:
    """Legacy workflow compatibility: global bridge state no longer exists."""
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Any change re-runs the disarm; 'randomize' after "
                               "each generate makes that automatic."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "disarm"
    CATEGORY = "MiniMax-H3/mod"

    def disarm(self, trigger=0):
        del trigger
        status = continuum_bridge.disarm()
        print(f"[MiniMaxH3RefModBridgeDisarm] {status}")
        return {"ui": {}, "result": (status,)}


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

    ``merge`` (training mode only): instead of stacking each ref's own pooled
    latent, one shared grid is refined jointly against *every* full encode
    (mean reconstruction error), so a collection lands on what's common
    across all the views — the cheap multi-exemplar analog of training, at
    one mod's token cost. See ``core.optimize_latent_multi``.

    ``motion_only`` (training mode only, experimental): video refs are
    converted to per-frame temporal differences (|f[t+1] - f[t]|) before
    encoding, so the latent carries where/how things move instead of what
    they look like — the static appearance never enters the mod. For a
    lineart animation this keeps the moving lines and drops the static
    drawing. Image refs have no motion and keep their appearance (warned).

    ``max_tokens`` (0 = off) hard-caps the total injected tokens: when the
    stacked refs exceed it, near-duplicate latent frames are dropped first,
    then frames are resampled to fit (see ``core.fit_token_budget``).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefModExtract",
            is_output_node=True,
            display_name="Extract H3 RefMod",
            description=(
                "Turn one or more references of the same concept into a RefMod. "
                "Stills plug into ref_image_1, video frames into ref_video_1, "
                "and the next slot of that type appears. refs are stacked into "
                "one video-kind mod, so a multi-image moodboard keeps each ref's "
                "own content instead of averaging away. 'training' mode (default) "
                "compresses the refs to a grid and refines it — good identity at "
                "a fraction of the tokens; 'encode' stores the full-res encode "
                "(max identity, MB-size mod). Flip 'merge' on to extract a "
                "collection as ONE consensus latent (joint refinement against "
                "every ref) instead of a stack."
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
                    tooltip="Legacy MiniMax-H3 VAERef. Loads its video checkpoint using the native ComfyUI VAE; prefer vae to share an already loaded VAE."),
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
                io.Boolean.Input("merge", default=False,
                    label_on="merge", label_off="stack",
                    tooltip="Merge mode (training only): instead of stacking each ref's own "
                            "pooled latent, optimize ONE shared grid against every full encode "
                            "jointly — the result lands on what's COMMON across all the views "
                            "(structure, motion, identity) rather than any single shot's "
                            "framing/background. Ideal for a collection: many angles of a "
                            "subject, a folder of similar clips -> one tiny consensus mod, one "
                            "ref block's worth of tokens. Keeps every full encode in VRAM "
                            "during refinement. Off = stack (each ref keeps its own frames). "
                            "Ignored when mode='encode'."),
                io.Boolean.Input("motion_only", default=False,
                    label_on="motion", label_off="full",
                    tooltip="EXPERIMENTAL — extract only what MOVES. Video refs are "
                            "converted to per-frame temporal differences "
                            "(|f[t+1] - f[t]|, normalized) before encoding, so the mod "
                            "carries where/how things move and the static appearance "
                            "(background, the lineart look, an outfit) never enters the "
                            "latent. For a lineart animation this keeps the moving lines "
                            "and drops the static drawing. Needs video refs — a still "
                            "has no motion, so image refs keep their appearance (warned). "
                            "Training mode only; the ref channel is content-based, so "
                            "treat the result as a soft motion guide, not a ControlNet. "
                            "Combines with 'merge'."),
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
                io.Combo.Input("extraction_preset", options=["manual", "identity_encode", "style_experimental", "motion_sequence"], default="manual", optional=True),
                io.String.Input("subfolder", default="", optional=True, tooltip="Optional folder inside models/refmods, for example celebs or voices."),
                io.String.Input("description", default="", multiline=True,
                    tooltip="Optional text describing the concept (e.g. 'a ginger woman with messy "
                            "hair', 'an animation style', 'handheld camera movement'). Stored in "
                            "the mod and printed in the info block — documentation only, no wiring."),
                io.Boolean.Input("save", default=True, label_on="save", label_off="don't save",
                    tooltip="Save the mod to mods/ so Load H3 RefMods can pick it up later."),
                io.Audio.Input("audio", optional=True, tooltip="Voice of this character. Saved in the same file as the visual references."),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE; the existing vae input is the H3 video VAE."),
                io.Autogrow.Input("refs_audio", optional=True,
                    template=io.Autogrow.TemplatePrefix(input=io.Audio.Input("ref_audio"),
                                                       prefix="ref_audio_", min=0, max=10)),
                io.String.Input("video_file", default="", optional=True,
                    tooltip="Optional speaking-video file, relative to ComfyUI/input or absolute. Extracts BOTH its frames and soundtrack together."),
                io.Float.Input("audio_start_seconds", default=0.0, min=0.0, max=86400.0, optional=True),
                io.Float.Input("audio_duration_seconds", default=5.0, min=0.25, max=15.0, optional=True,
                    tooltip="Seconds saved from each connected audio source or speaking video. Existing video-frame inputs are assumed to be 24 fps."),
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
                concept_type="generic", mask=None, background_retention=0.0, subfolder="",
                merge=False, motion_only=False, extraction_preset="manual", audio=None, audio_vae=None,
                refs_audio=None, video_file="", audio_start_seconds=0.0, audio_duration_seconds=5.0,
                **legacy) -> io.NodeOutput:
        name = _sanitize_name(name)
        if extraction_preset == "identity_encode":
            mode, ref_resolution, identity, merge, motion_only = "encode", 1024, 0, False, False
        elif extraction_preset == "style_experimental":
            mode, pool_h, pool_w, identity, merge, motion_only = "training", 8, 8, 150, False, False
        elif extraction_preset == "motion_sequence":
            mode, pool_h, pool_w, latent_frames, merge, motion_only = "training", 16, 16, 16, False, False
        elif extraction_preset != "manual":
            raise ValueError("Unknown extraction preset.")
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
        vae = _resolve_visual_vae(vae, av_encoder)

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
        if audio is not None or any(v is not None for v in (refs_audio or {}).values()) or str(video_file).strip():
            from .refmod_extract_audio import extract_with_original_visual
            ordered = [(_normalize_ref(src, label="character reference"), is_video) for src, is_video in ordered]
            mod = extract_with_original_visual(cls.execute, name, ordered, vae, audio_vae,
                audio=audio, refs_audio=refs_audio, video_file=video_file,
                audio_start_seconds=audio_start_seconds, audio_duration_seconds=audio_duration_seconds,
                visual_options=dict(mode=mode, ref_resolution=ref_resolution, pool_h=pool_h, pool_w=pool_w,
                    identity=identity, latent_frames=latent_frames, description=description, merge=merge,
                    motion_only=motion_only, multiplier=multiplier, mask=mask,
                    background_retention=background_retention, concept_type=concept_type, max_tokens=max_tokens))
            if save:
                mod.path = mod_output_path(name, subfolder)
                mod.save(mod.path)
                _cache_mod(mod.path, mod)
            print("[Extract H3 RefMod] Combined appearance + voice: " + mod.report()
                  + (f"\nSaved: {mod.path}.safetensors" if save else " (not saved)"))
            return io.NodeOutput([(mod, 1.0)])
        if not ordered:
            raise ValueError(
                "MiniMaxH3RefModExtract: connect at least one image to "
                "ref_image_1, or video frames to ref_video_1, or a folder bundle.")
        if merge and mode != "training":
            print("[MiniMaxH3RefModExtract] warning: 'merge' only applies to "
                  "training mode — stacking the refs as usual for mode='encode'.")
        if motion_only and mode != "training":
            print("[MiniMaxH3RefModExtract] warning: 'motion_only' only applies to "
                  "training mode — extracting the full appearance for mode='encode'.")
            motion_only = False
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
        motion_applied = False
        motion_warned = False
        # merge mode: hold each ref's pooled candidate + full encode, then
        # refine one shared latent against all of them jointly after the loop
        merge_refs = [] if (merge and mode == "training" and n_refs > 1) else None
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
            if motion_only and is_video and src.shape[0] > 1:
                # temporal differences: |f[t+1] - f[t]|, normalized by the clip's
                # peak motion so static frames stay dark ("no motion here") and
                # moving parts light up — appearance never enters the latent
                d = (src[1:] - src[:-1]).abs()
                peak = d.max()
                if peak > 1e-6:
                    d = d / peak
                src = d
                motion_applied = True
                print(f"[MiniMaxH3RefModExtract] {label}: motion_only — "
                      f"encoded temporal differences instead of the frames "
                      f"(static appearance stripped)")
            elif motion_only and not is_video and not motion_warned:
                print("[MiniMaxH3RefModExtract] warning: motion_only needs video "
                      "refs — a still has no motion, keeping its appearance.")
                motion_warned = True
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
                mask_px = _resize_mask(mask_batch[src_idx:src_idx + 1], src.shape[1], src.shape[2],
                                       "center" if mode == "encode" and canvas is not None else "disabled")
            if src.shape[1] <= 0 or src.shape[2] <= 0:
                raise ValueError(
                    f"MiniMaxH3RefModExtract: reference {src_idx + 1} "
                    f"({'video' if is_video else 'image'}) has an empty frame "
                    f"{tuple(src.shape)} right before VAE encode (mode={mode}, "
                    f"ref_resolution={ref_resolution}, canvas={canvas}). "
                    f"Check that this specific reference's source image/video "
                    f"is valid.")
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
                if merge_refs is not None:
                    merge_refs.append((pooled.cpu(), z.float().cpu()))
                elif identity > 0:
                    print(f"[MiniMaxH3RefModExtract] {label}: refining identity "
                          f"({int(identity)} gradient steps)...")
                    pooled = optimize_latent(pooled, z.float(), steps=int(identity),
                                              progress_every=100)
                    print(f"[MiniMaxH3RefModExtract] {label}: identity refinement done")
            if merge_refs is None:
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

        merged_n = 0
        if merge_refs is not None:
            # one shared grid refined against every full encode jointly, so the
            # result is the consensus of the collection, not any single shot
            common_t = max(p.shape[2] for p, _ in merge_refs)
            init = torch.stack([
                pool_latent(p, common_t, gh, gw) for p, _ in merge_refs
            ]).mean(0)
            print(f"[MiniMaxH3RefModExtract] merging {len(merge_refs)} references "
                  f"into one shared {common_t}x{gh}x{gw} latent"
                  + (f", {int(identity)} joint gradient steps" if identity > 0 else
                     " (pure pooling mean — identity=0)"))
            if identity > 0:
                init = optimize_latent_multi(
                    init, [f for _, f in merge_refs],
                    steps=int(identity), progress_every=100)
                print("[MiniMaxH3RefModExtract] merge refinement done")
            latent = init.to(torch.float16)
            merged_n = len(merge_refs)
        else:
            latent = torch.cat(frames, dim=2)  # [1, 24, total_t, h, w]
        if multiplier > 1:
            latent = latent.repeat(1, 1, multiplier, 1, 1)  # data multiplier
        if max_tokens > 0:
            latent = fit_token_budget(latent, max_tokens, name)
        total_t = latent.shape[2]
        kind = "video" if total_t > 1 else "image"
        # the VAE encodes at 16x spatial scale, so a latent of 40x20 = 640x320 px
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
        if merged_n:
            src = f"merge {merged_n} refs"
        elif len(frames) > 1:
            src = "stack"
        else:
            src = "video" if n_vid else "image"
        tags = [f"{n_img} img, {n_vid} vid"]
        if merged_n:
            tags.append(f"merged {merged_n} refs")
        if motion_applied:
            tags.append("motion_only")
        if multiplier > 1:
            tags.append(f"x{multiplier} repeat")
        if mask_batch is not None:
            tags.append(f"masked (bg_retention={background_retention})")
        mod = H3RefMod(
            name=name,
            kind=kind,
            latent=latent,
            latent_h=latent.shape[3],
            latent_w=latent.shape[4],
            latent_t=total_t,
            mode=mode,
            source=src,
            source_shape=" +".join(source_shapes),
            pool=f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode" else f"{total_t}x{gh}x{gw}",
            optimize_steps=int(identity) if mode == "training" else 0,
            tags=tags,
            description=(description or "").strip(),
            concept_type=concept_type,
        )

        if save:
            path_no_ext = mod_output_path(name, subfolder)
            path = mod.save(path_no_ext)
            mod.path = path_no_ext  # so Fix H3 RefMod Config can re-save in place
            _cache_mod(path_no_ext, mod)
            print(f"[MiniMaxH3RefModExtract] saved {_summarize(mod)} -> {path}")
        else:
            print(f"[MiniMaxH3RefModExtract] {_summarize(mod)} (not saved)")
        if mod.description:
            print(f"[MiniMaxH3RefModExtract] description: {mod.description}")
        return io.NodeOutput([(mod, 1.0)])


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

class MiniMaxH3RefModInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mods": ("H3_REF_MODS",),
            "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "preview": (["off", "stored", "compare_strength"],),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
        }, "optional": {"vae": ("VAE",)}}
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("details", "image_preview", "audio_preview")
    FUNCTION = "inspect"
    CATEGORY = "MiniMax-H3/mod"

    def inspect(self, mods, index=0, preview="off", strength=0.5, vae=None):
        error = _number_error("strength", strength, "FLOAT", {"min":0, "max":1})
        if error:
            raise ValueError(error)
        details = []
        for mod, row_strength in mods:
            details.append({"name": mod.name, "path": mod.path, "kind": mod.kind,
                            "shape": list(mod.latent.shape), "tokens": mod.token_count,
                            "strength": row_strength, "description": mod.description,
                            "concept_type": mod.concept_type, "saved_config": mod.config})
            if isinstance(mod, ProfileRefMod):
                details[-1].update(selection=mod.report(), stored_references=len(mod.profile.references),
                                   selected_references=[{"kind": r.kind, "duration": r.duration,
                                       "visual_shape": list(r.visual.shape) if r.visual is not None else None,
                                       "audio_shape": list(r.audio.shape) if r.audio is not None else None}
                                       for r in mod.selected_references()])
        report = json.dumps({"total_tokens": _check_token_budget(mods, 0), "refs": details}, indent=2, ensure_ascii=False)
        images = torch.zeros(1,64,64,3)
        audio = {"waveform": torch.zeros(1,2,1), "sample_rate": 32000}
        if preview != "off":
            if vae is None:
                raise ValueError("Connect the matching H3 VAE to enable previews.")
            if not 0 <= index < len(mods):
                raise ValueError("Preview index is outside the RefMod bundle.")
            mod = mods[index][0]
            z = mod.latent[..., :80] if mod.kind == "audio" else mod.latent[:, :, :1]
            variants = [z]
            if preview == "compare_strength":
                variants.append(strength * z + (1.0 - strength) * _blur_latent(z))
            elif preview != "stored":
                raise ValueError("Unknown preview mode.")
            if mod.kind == "audio":
                from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
                if not isinstance(vae.first_stage_model, MiniMaxH3AudioVAE):
                    raise ValueError("Audio previews require the MiniMax H3 audio VAE.")
                decoded = [vae_decode_audio(vae, {"samples": value})["waveform"] for value in variants]
                audio = {"waveform": torch.cat(decoded, dim=-1), "sample_rate": 32000}
            else:
                images = torch.cat([vae.decode(value) for value in variants], dim=0)
        return (report, images, audio)


class MiniMaxH3RefModAudioExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",), "audio_vae": ("VAE",),
            "name": ("STRING", {"default": "audio_refmod"}),
            "max_seconds": ("FLOAT", {"default": 30.0, "min": 0.025, "max": 600.0}),
            "max_tokens": ("INT", {"default": 5120, "min": 0, "max": 65536}),
            "budget_policy": (["error", "truncate"],),
            "concept_type": (["voice", "singing", "music_style", "sound_fx", "ambience"],),
            "description": ("STRING", {"default": "", "multiline": True}),
            "subfolder": ("STRING", {"default": ""}),
            "save": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("H3_REF_MODS",)
    FUNCTION = "extract"
    CATEGORY = "MiniMax-H3/mod"

    def extract(self, audio, audio_vae, name="audio_refmod", max_seconds=30.0,
                max_tokens=5120, budget_policy="error", concept_type="voice",
                description="", subfolder="", save=True):
        name = _sanitize_name(name)
        mod = make_audio_mod(audio_vae, audio, name, max_seconds, max_tokens,
                             budget_policy, description, concept_type)
        if save:
            mod.path = mod_output_path(name, subfolder)
            mod.save(mod.path)
            _cache_mod(mod.path, mod)
        print(f"[RefMod audio] {name}: {mod.latent_t / 40:.2f}s, {mod.token_count} tokens")
        return ([(mod, 1.0)],)


class MiniMaxH3RefModSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mods": ("H3_REF_MODS",),
            "filename_prefix": ("STRING", {"default": ""}),
            "subfolder": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "saved_paths")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax-H3/mod"
    DESCRIPTION = "Save each distinct RefMod in a bundle. Runs without a connected output; existing destination files are replaced. Loader strengths are preserved in the output bundle, not baked into files."

    def save(self, mods, filename_prefix="", subfolder=""):
        if not mods:
            raise ValueError("The RefMod bundle is empty; nothing to save.")
        planned, used_paths = {}, set()
        for mod, _strength in mods:
            if id(mod) in planned:
                continue
            base = _sanitize_name(filename_prefix + mod.name)
            name = base
            index = 2
            path = mod_output_path(name, subfolder)
            while os.path.normcase(path) in used_paths:
                name = f"{base}_{index}"
                index += 1
                path = mod_output_path(name, subfolder)
            used_paths.add(os.path.normcase(path))
            planned[id(mod)] = replace(mod, name=name, path=path)
        paths = []
        for mod in planned.values():
            paths.append(mod.save(mod.path))
            _cache_mod(mod.path, mod)
        out = [(planned[id(mod)], strength) for mod, strength in mods]
        report = "\n".join(paths)
        print(f"[RefMod Save] Saved {len(paths)} file(s):\n{report}")
        return {"ui": {"text": paths}, "result": (out, report)}


class MiniMaxH3RefModMasterExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = MiniMaxH3RefModExtract.define_schema()
        # Master remains the upstream separate-file workflow; combined files
        # use the original Extract node. Avoid duplicate inherited audio ports.
        schema.inputs = [port for port in schema.inputs if port.id not in {
            "audio", "audio_vae", "refs_audio", "video_file", "audio_start_seconds", "audio_duration_seconds"}]
        schema.node_id = "MiniMaxH3RefModMasterExtract"
        schema.display_name = "Extract H3 RefMod Master"
        schema.description = (
            "Extract appearance and/or audio into one RefMod bundle. Visual and audio "
            "VAEs run sequentially. Saves separate name_visual and name_audio files. "
            "This extracts references; it does not jointly train identity and voice.")
        schema.inputs.extend([
            io.Audio.Input("audio", optional=True),
            io.Vae.Input("audio_vae", optional=True,
                         tooltip="MiniMax H3 audio VAE. The visual vae socket remains separate."),
            io.Float.Input("audio_max_seconds", default=30.0, min=0.025, max=600.0),
            io.Int.Input("audio_max_tokens", default=5120, min=0, max=65536),
            io.Combo.Input("audio_budget_policy", options=["error", "truncate"], default="error"),
            io.Combo.Input("audio_concept_type", options=["voice", "singing", "music_style", "sound_fx", "ambience"], default="voice"),
            io.Int.Input("max_total_tokens", default=0, min=0, max=1048576,
                         tooltip="Combined visual and audio budget. 0 disables this extra limit."),
        ])
        schema.outputs = [io.Custom("H3_REF_MODS").Output("mods"), io.String.Output("details")]
        return schema

    @classmethod
    def execute(cls, name, mode="training", audio=None, audio_vae=None,
                audio_max_seconds=30.0, audio_max_tokens=5120,
                audio_budget_policy="error", audio_concept_type="voice",
                max_total_tokens=0, save=True, subfolder="", description="", **visual):
        has_visual = any(
            value is not None
            for group in (visual.get("refs_image"), visual.get("refs_video"))
            for value in (group or {}).values()
        ) or any(value is not None for value in (visual.get("refs_bundle") or []))
        has_visual = has_visual or any(
            value is not None and (key == "image" or key.startswith(("ref_image_", "ref_video_")))
            for key, value in visual.items())
        if not has_visual and audio is None:
            raise ValueError("Connect at least one image, video, reference bundle or audio to the Master.")
        if has_visual and visual.get("vae") is None and visual.get("av_encoder") is None:
            raise ValueError("Visual references require the video VAE or av_encoder.")
        if audio is not None and audio_vae is None:
            raise ValueError("Audio references require the MiniMax H3 audio VAE in audio_vae.")
        _check_token_budget([], max_total_tokens)
        name = _sanitize_name(name)
        paths = {kind: mod_output_path(name + "_" + kind, subfolder)
                 for kind, enabled in (("visual", has_visual), ("audio", audio is not None))
                 if enabled and save}
        if save:
            print("[RefMod Master] Preparing references in memory; saving follows both extractions and the token-budget check.")
        else:
            print("[RefMod Master] save=False: output bundle only; no files will be written by Master.")
        mods = []
        if has_visual:
            mods.extend(MiniMaxH3RefModExtract.execute(
                name=name + "_visual", mode=mode, save=False,
                description=description, **visual)[0])
        if audio is not None:
            mods.extend(MiniMaxH3RefModAudioExtract().extract(
                audio=audio, audio_vae=audio_vae, name=name + "_audio",
                max_seconds=audio_max_seconds, max_tokens=audio_max_tokens,
                budget_policy=audio_budget_policy, concept_type=audio_concept_type,
                description=description, save=False)[0])
        total = _check_token_budget(mods, max_total_tokens)
        saved_paths = []
        for mod, _strength in mods:
            if save:
                mod.path = paths["audio" if mod.kind == "audio" else "visual"]
                existed = os.path.isfile(mod.path + ".safetensors")
                destination = mod.save(mod.path)
                _cache_mod(mod.path, mod)
                saved_paths.append(destination)
                action = "Replaced" if existed else "Created"
                print(f"[RefMod Master] {action}: {destination}")
        print(f"[RefMod Master] Complete: {len(mods)} references, {total} tokens, {len(saved_paths)} files saved.")
        details = json.dumps({"name": name, "total_tokens": total, "saved_paths": saved_paths,
                              "refs": [{"name":m.name, "kind":m.kind, "tokens":m.token_count,
                                        "path":m.path} for m, _ in mods]}, indent=2, ensure_ascii=False)
        return io.NodeOutput(mods, details)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RefModSave": MiniMaxH3RefModSave,
    "MiniMaxH3RefModMasterExtract": MiniMaxH3RefModMasterExtract,
    "MiniMaxH3RefModInspect": MiniMaxH3RefModInspect,
    "MiniMaxH3RefModAudioExtract": MiniMaxH3RefModAudioExtract,
    "MiniMaxH3RefModExtract": MiniMaxH3RefModExtract,
    "MiniMaxH3RefModFolderLoader": MiniMaxH3RefModFolderLoader,
    "MiniMaxH3RefModsLoader": MiniMaxH3RefModsLoader,
    "MiniMaxH3RefModsAxis": MiniMaxH3RefModsAxis,
    "MiniMaxH3RefModApply": MiniMaxH3RefModApply,
    "MiniMaxH3RefModStepCurve": MiniMaxH3RefModStepCurve,
    "MiniMaxH3RefModConfig": MiniMaxH3RefModConfig,
    "MiniMaxH3RefModContinuumBridge": MiniMaxH3RefModContinuumBridge,
    "MiniMaxH3RefModBridgeDisarm": MiniMaxH3RefModBridgeDisarm,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefModSave": "Save H3 RefMods",
    "MiniMaxH3RefModMasterExtract": "Extract H3 RefMod Master",
    "MiniMaxH3RefModInspect": "Inspect H3 RefMod",
    "MiniMaxH3RefModAudioExtract": "Extract H3 Audio RefMod",
    "MiniMaxH3RefModExtract": "Extract H3 RefMod",
    "MiniMaxH3RefModFolderLoader": "Load H3 RefMod Folder",
    "MiniMaxH3RefModsLoader": "Load H3 RefMods",
    "MiniMaxH3RefModsAxis": "Load H3 RefMod Axis",
    "MiniMaxH3RefModApply": "Apply H3 RefMod",
    "MiniMaxH3RefModStepCurve": "H3 RefMod Step Curve",
    "MiniMaxH3RefModConfig": "Fix H3 RefMod Config",
    "MiniMaxH3RefModContinuumBridge": "Continuum RefMod Bridge",
    "MiniMaxH3RefModBridgeDisarm": "Disarm H3 RefMod Bridge",
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
        register_routes(PromptServer.instance, _list_mod_names, _find_mod_path, read_refmod_meta)
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
