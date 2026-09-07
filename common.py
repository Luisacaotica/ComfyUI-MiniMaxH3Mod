"""Shared helpers for the RefMod pack: media loading and the refmods folder."""

from __future__ import annotations

import os
import ntpath
import math
import random
from typing import List, Optional, Tuple

import torch
import comfy.utils

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def refmods_dir() -> str:
    """First registered RefMod root, or the default; selection does not write."""
    import folder_paths
    try:
        roots = folder_paths.get_folder_paths("refmods")
    except KeyError:
        roots = []
    return roots[0] if roots else os.path.join(folder_paths.models_dir, "refmods")


def list_media_files(folder: str) -> Tuple[List[str], List[str]]:
    """(images, videos) directly under ``folder`` (top level only), sorted by name."""
    images, videos = [], []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                if ext in IMAGE_EXTS:
                    images.append(p)
                elif ext in VIDEO_EXTS:
                    videos.append(p)
    return images, videos


def load_image_file(path: str, max_edge: Optional[int] = None) -> torch.Tensor:
    """Load one image file -> [1, H, W, 3] float32 in [0, 1]."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max_edge is not None:
            scale = min(1.0, max_edge / max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    return arr.unsqueeze(0)  # [1, H, W, 3]


def _sample_video_frames(frames, total, max_frames, max_edge):
    """Uniform known-length sampling; bounded deterministic reservoir otherwise."""
    import numpy as np
    from PIL import Image
    known = math.isfinite(total) and total > 0
    total = int(total) if known else 0
    targets = None
    if total > max_frames:
        targets = {round(i * (total - 1) / max(1, max_frames - 1)) for i in range(max_frames)}
    kept = []
    rng = random.Random(0)
    for index, frame in enumerate(frames):
        if targets is not None and index not in targets:
            continue
        # Treat underestimated counts as unknown after filling the reservoir.
        slot = len(kept) if len(kept) < max_frames else rng.randrange(index + 1)
        if slot >= max_frames:
            continue
        frame = np.asarray(frame)
        if max_edge is not None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_edge / max(h, w))
            if scale < 1.0:
                frame = np.asarray(Image.fromarray(frame).resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR))
        item = (index, frame.copy())
        if slot == len(kept):
            kept.append(item)
        else:
            kept[slot] = item
        if targets is not None and len(kept) == len(targets):
            break
    if not kept:
        raise ValueError("Video contains no decodable frames.")
    return torch.from_numpy(np.stack([frame for _, frame in sorted(kept)])).float() / 255.0


def load_video_file(path: str, max_frames: int = 240,
                    max_edge: Optional[int] = None) -> torch.Tensor:
    """Load bounded, ordered RGB frames; release either decoder on every exit."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    errors = []
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            def frames():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return _sample_video_frames(frames(), total, max_frames, max_edge)
        finally:
            cap.release()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        try:
            total = reader.get_meta_data().get("nframes", 0) or 0
            return _sample_video_frames(iter(reader), total, max_frames, max_edge)
        finally:
            reader.close()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    raise RuntimeError(f"Cannot decode video '{path}': " + "; ".join(errors))


def mod_output_path(name, subfolder=""):
    relative = "/".join(part for part in (subfolder.replace("\\", "/"), name) if part)
    if (ntpath.splitdrive(relative)[0] or relative.startswith("/") or
            any(p in ("..", ".", "graph_presets", ".git", "__pycache__") for p in relative.split("/"))):
        raise ValueError("Use a relative RefMod subfolder without '..' or reserved directories.")
    root = os.path.realpath(refmods_dir())
    path = os.path.join(root, relative)
    if os.path.commonpath((root, os.path.realpath(path + ".safetensors"))) != root:
        raise ValueError("RefMod save path leaves the model folder.")
    return path


def resize_ref(image, short_edge: int, canvas=None):
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


def snap_to_causal_grid(n_frames: int) -> int:
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




def ensure_min_size(image, floor: int = 320):
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
