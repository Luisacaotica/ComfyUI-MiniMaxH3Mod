"""
extract_mod.py — standalone MiniMax H3 RefMod extractor (no training)

Turns reference images/videos into a ``.safetensors`` mod that the ComfyUI
Load H3 RefMods / Apply nodes can inject into MiniMax H3 generation.  Only
the H3 video VAE is needed; the 29B DiT is never loaded.

Two modes:

  * ``full`` (default) — refs are resized to ``--resolution`` short edge (down
    only) and encoded at that resolution, exactly like the official ref2video
    node.  This is what carries identity (a face, an outfit); files are
    ~0.2-1 MB per frame.
  * ``pooled`` — the latent is average-pooled to a small grid (4x4 by default
    = 4 tokens per latent frame) and optionally refined with a few gradient
    steps (still model-free).  Nearly free to inject but only carries concept
    / motion, not fine identity.

Usage
-----
  # full-res identity mod (recommended for characters)
  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --image char.png --vae path/to/h3_video_vae.safetensors \
      --name my_character --mode full --resolution 1024

  # tiny concept/motion mod
  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name dance --mode pooled --pool 4 --latent-frames 2

Multi-reference concept (each ref becomes its own latent frame):

  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --image face_a.png --image face_b.png --image full.png \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name disney_char --mode full --resolution 1024

Output goes to ``ComfyUI/models/refmods/`` by default (created on first run,
next to loras/ and unet/).

Run it with the same Python/venv that runs ComfyUI (it imports ``comfy``
from the install the script lives in).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from common import load_image_file, load_video_file, refmods_dir
from core import H3RefMod, optimize_latent, pool_latent

MAX_VIDEO_FRAMES = 60  # uniform sample cap; temporal pooling averages anyway


# ═══════════════════════════════════════════════════════════════════════════
# Media loading (delegates to common.py: image/video -> [B, H, W, 3] float32)
# ═══════════════════════════════════════════════════════════════════════════

def _load_image(path: str, max_edge: int) -> torch.Tensor:
    return load_image_file(path, max_edge=max_edge)


def _resize_ref(image: torch.Tensor, short_edge: int, canvas=None) -> torch.Tensor:
    """Aspect-preserving downscale (never upscale) to ``short_edge`` px; dims to /32.

    When several refs are stacked into one mod they must share a single spatial
    canvas, so ``canvas`` (tw, th) cover-crops each ref to it.
    """
    import comfy.utils
    h, w = image.shape[1], image.shape[2]
    scale = min(1.0, short_edge / min(h, w))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    crop = "disabled"
    if canvas is not None:
        tw, th = canvas
        crop = "center"
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", crop)
    return samples.movedim(1, -1)


def _load_video(path: str, max_edge: int, max_frames: int = MAX_VIDEO_FRAMES) -> torch.Tensor:
    return load_video_file(path, max_frames=max_frames, max_edge=max_edge)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Extract a MiniMax H3 RefMod from one or more images/videos "
                    "of the same concept (they are stacked into a video-kind mod)")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="reference image path (repeatable for multi-view concepts)")
    ap.add_argument("--video", action="append", default=[], metavar="PATH",
                    help="reference video path (repeatable)")
    ap.add_argument("--vae", required=True, help="MiniMax H3 video VAE .safetensors")
    ap.add_argument("--name", default=None, help="mod name (default: first source file stem)")
    ap.add_argument("--output", default=None,
                    help="output dir (default: custom_nodes/ComfyUI-MiniMaxH3Mod/mods)")
    ap.add_argument("--mode", choices=["full", "pooled"], default="pooled",
                    help="pooled = compressed grid refined by --identity (default, good balance); full = full-res encode (~1K tokens/img)")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="full mode: target short edge in px, downscale only (default 1024; 2048 = max fidelity)")
    ap.add_argument("--pool", type=int, default=16, help="pooled mode: spatial latent grid (even, default 16)")
    ap.add_argument("--pool-w", type=int, default=None, help="pool width (default: == --pool)")
    ap.add_argument("--latent-frames", type=int, default=16,
                    help="frames kept per video ref (default 16; images use 1): pooled mode pools them, full mode samples them")
    ap.add_argument("--identity", type=int, default=500,
                    help="pooled mode: how tightly the mod clings to the reference (gradient refinement steps; default 500, 0 = pure pooling)")
    ap.add_argument("--multiplier", type=int, default=1,
                    help="data multiplier: repeat the extracted ref N times along time so a short video/GIF isn't drowned out by the main video's tokens (default 1 = no repeat)")
    ap.add_argument("--max-edge", type=int, default=1536,
                    help="resize source so the longest edge is <= this before encoding (default 1536)")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="max video frames sampled (uniform) before encoding (default 60; lower for CPU)")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu (VAE device)")
    args = ap.parse_args()

    if not args.image and not args.video:
        ap.error("provide at least one --image or --video")

    import comfy.model_management
    import comfy.sd
    import comfy.utils

    device = comfy.model_management.get_torch_device() if args.device == "auto" \
        else torch.device(args.device)

    # ── load VAE ─────────────────────────────────────────────────────
    print(f"[extract] loading VAE {args.vae} (device={args.device})")
    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()

    # ── encode each source (full-res or pooled), then stack along time ──
    pool_w = args.pool_w or args.pool
    # in full mode --resolution is the target short edge, so keep the load-time
    # longest-edge cap from interfering with it
    load_max_edge = args.resolution * 2 if args.mode == "full" else args.max_edge
    # full-res refs must share one spatial canvas so the stacked latent has a
    # single H/W: anchor on the first source, cover-crop the rest to it
    canvas = None
    if args.mode == "full" and len(args.image) + len(args.video) > 1:
        if args.image:
            first = _load_image(args.image[0], load_max_edge)
        else:
            first = _load_video(args.video[0], load_max_edge, args.max_frames)
        h, w = first.shape[1], first.shape[2]
        scale = min(1.0, args.resolution / min(h, w))
        canvas = (max(32, round(w * scale / 32) * 32),
                  max(32, round(h * scale / 32) * 32))
    frames = []
    shapes = []
    n_img = n_vid = 0
    for path in args.image:
        src = _load_image(path, load_max_edge)
        if args.mode == "full":
            src = _resize_ref(src, args.resolution, canvas)
        print(f"[extract] image {path}: {tuple(src.shape)} (mode={args.mode})")
        with torch.no_grad():
            z = vae.encode(src.to(device)).float().cpu()
        if args.mode == "full":
            pooled = z.to(torch.float16)
        else:
            pooled = pool_latent(z, 1, args.pool, pool_w).to(torch.float16)
            if args.identity > 0:
                pooled = optimize_latent(pooled, z, steps=args.identity, device=device)
        frames.append(pooled)
        shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
        n_img += 1
    for path in args.video:
        src = _load_video(path, load_max_edge, args.max_frames)
        if args.mode == "full":
            n_src = src.shape[0]
            if args.latent_frames < n_src:
                idx = torch.linspace(0, n_src - 1, args.latent_frames).round().long()
                src = src[idx]
            src = _resize_ref(src, args.resolution, canvas)
        print(f"[extract] video {path}: {tuple(src.shape)} (mode={args.mode})")
        with torch.no_grad():
            z = vae.encode(src.to(device)).float().cpu()
        if args.mode == "full":
            pooled = z.to(torch.float16)
        else:
            pool_t = min(args.latent_frames, z.shape[2])
            pooled = pool_latent(z, pool_t, args.pool, pool_w).to(torch.float16)
            if args.identity > 0:
                pooled = optimize_latent(pooled, z, steps=args.identity, device=device)
        frames.append(pooled)
        shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
        n_vid += 1

    if args.mode == "full" and args.identity > 0:
        print(f"[extract] warning: --identity only applies to pooled mode — "
              f"full mode stores the actual encode, so --identity {args.identity} "
              f"was ignored.")

    latent = torch.cat(frames, dim=2)
    if args.multiplier > 1:
        latent = latent.repeat(1, 1, args.multiplier, 1, 1)
    total_t = latent.shape[2]
    kind = "video" if total_t > 1 else "image"

    name = args.name or (os.path.splitext(os.path.basename(args.image[0]))[0]
                         if args.image else os.path.splitext(os.path.basename(args.video[0]))[0])
    out_dir = args.output or refmods_dir()
    px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
    mod = H3RefMod(
        name=name,
        kind=kind,
        latent=latent,
        latent_h=latent.shape[3],
        latent_w=latent.shape[4],
        latent_t=total_t,
        mode=args.mode,
        source="stack" if len(frames) > 1 else ("video" if n_vid else "image"),
        source_shape=" +".join(shapes),
        pool=f"full-res {px_w}x{px_h}px (short-edge cap {args.resolution}px)" if args.mode == "full" else f"{total_t}x{args.pool}x{pool_w}",
        optimize_steps=args.identity if args.mode == "pooled" else 0,
        tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{args.multiplier} repeat"] if args.multiplier > 1 else []),
    )
    path = mod.save(os.path.join(out_dir, name))
    mb = latent.numel() * latent.element_size() / 1024 / 1024
    print(f"[extract] saved {kind} mod '{name}' "
          f"({mod.token_count} tokens, {mb:.2f} MB) -> {path}")
    print(f"[extract] load it in ComfyUI with the Load H3 RefMods node "
          f"(dropdown '{name}', strength 1.0 = full ref), then chain "
          f"Apply H3 RefMod (Cond) into your sampling.")


if __name__ == "__main__":
    main()
