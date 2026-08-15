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
  MiniMaxH3RefModApply         — inject the bundle into MINIMAX_H3_COND (pack)
  MiniMaxH3RefModApplyCond     — inject the bundle into CONDITIONING (built-in)

Standalone extraction (image/video files): see extract_mod.py
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
