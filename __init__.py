"""
ComfyUI-MiniMaxH3Mod — no-training "RefMod" reference adapters for MiniMax H3

Reference videos are expensive because they inject thousands of tokens into
the H3 packed sequence.  A RefMod compresses a reference image/video into a
tiny pooled latent (~4-16 tokens) that still rides the model's native ref2va
path — the DiT attends to it through all 50 blocks like a real reference, but
at a fraction of the compute.  Extraction needs only the H3 VAE: no diffusion
model load, no training.

Nodes
─────
  MiniMaxH3RefModExtract       — ref_image_1..N stills + ref_video_1..N clips -> saved mod
  MiniMaxH3RefModFolderLoader  — load every image/video in a folder as an ordered ref list
  MiniMaxH3RefModsLoader       — load 1-8 mods with a typed strength each (LoRA-style)
  MiniMaxH3RefModsAxis         — A/B mod pairs on one signed slider each (negative -> A, positive -> B)
  MiniMaxH3RefModApply         — inject the bundle into MINIMAX_H3_COND (pack) or CONDITIONING (built-in);
                                 the old split Apply/ApplyCond merged into one node (old workflows migrate)

Standalone extraction (image/video files): see extract_mod.py
"""

__author__ = "Luisa (luisacaotica)"

import os
import sys

# The av_encoder input on Extract and the pack-conditioning Apply node need the
# ComfyUI-MiniMaxH3 pack (a sibling folder in custom_nodes/).  Everything else
# — Extract with a plain VAE, the folder loader, both mod loaders, and the
# built-in CONDITIONING Apply — works without it.  Check once at import time so
# the missing pack shows a short, actionable warning instead of a raw traceback
# the first time someone uses av_encoder.
PACK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI-MiniMaxH3")
H3_PACK_AVAILABLE = os.path.isdir(PACK_DIR)
if not H3_PACK_AVAILABLE:
    print(
        "[MiniMaxH3Mod] ComfyUI-MiniMaxH3 pack not found — install it first "
        "(ComfyUI Manager: search 'MiniMax H3', or git clone "
        "https://github.com/xiaolibai-sys/ComfyUI-MiniMaxH3 into custom_nodes/). "
        "Required only for the av_encoder input on Extract H3 RefMod and the "
        "pack-conditioning Apply H3 RefMod node; the rest works without it."
    )

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
