"""Shared helpers for the RefMod pack: media loading and the refmods folder."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def refmods_dir() -> str:
    """ComfyUI models/refmods — the mod storage folder, created on first use.

    Sits next to loras/, unet/ etc. instead of inside the pack folder, so mods
    live in the standard model tree.
    """
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "refmods")
    os.makedirs(d, exist_ok=True)
    return d


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


def load_video_file(path: str, max_frames: int = 240,
                    max_edge: Optional[int] = None) -> torch.Tensor:
    """Load one video file -> [T, H, W, 3] float32 in [0, 1].

    Uses opencv if available, else imageio. ``max_frames`` caps how many
    frames are ever buffered (uniformly sampled for long videos, *during*
    decode, not after), and ``max_edge`` downscales each frame — so a folder
    of long high-res videos stays memory-bounded instead of decoding the
    whole thing at native resolution first and OOM-ing ComfyUI.
    """
    frames = None
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > max_frames:
            # uniform target indices up front: skip frames while decoding so
            # we never hold more than max_frames decoded frames at once
            targets = set(np.linspace(0, total - 1, max_frames).round().astype(int).tolist())
        else:
            targets = None
        out = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if targets is not None and idx not in targets:
                idx += 1
                continue
            idx += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if max_edge is not None:
                scale = min(1.0, max_edge / max(w, h))
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                        interpolation=cv2.INTER_LINEAR)
            out.append(frame)
        cap.release()
        if out:
            frames = torch.from_numpy(np.stack(out)).float() / 255.0
    except Exception:
        frames = None

    if frames is None:
        try:
            import imageio.v2 as imageio
            import numpy as np
            from PIL import Image
            reader = imageio.get_reader(path)
            out = []
            for i, frame in enumerate(reader):
                if max_frames and i >= max_frames:
                    break
                frame = np.asarray(frame)
                h, w = frame.shape[:2]
                if max_edge is not None:
                    scale = min(1.0, max_edge / max(w, h))
                    if scale < 1.0:
                        frame = np.asarray(Image.fromarray(frame).resize(
                            (max(1, round(w * scale)), max(1, round(h * scale))),
                            Image.LANCZOS))
                out.append(frame)
            reader.close()
            if out:
                frames = torch.from_numpy(np.stack(out)).float() / 255.0
        except Exception:
            frames = None

    if frames is None:
        raise RuntimeError(
            f"No video loader available for {path} (tried opencv and imageio).")

    n = frames.shape[0]
    if n > max_frames:
        idx = torch.linspace(0, n - 1, max_frames).round().long()
        frames = frames[idx]
    return frames
