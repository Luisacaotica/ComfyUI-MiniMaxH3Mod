# Changelog

All notable changes are tracked here. Each version is also published as a
[GitHub Release](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/releases),
so you can keep using an older version if a new one changes something you
rely on.

## v0.1.0 — 2026-08-16

First tagged release. Extract references once as tiny `.safetensors`
"mods" and inject them through conditioning — no full video/image loading
every generation, no training.

### Nodes

- **Extract H3 RefMod** — image / video / GIF → one small mod file. Modes:
  `training` (pooled concept/identity thumbnails; the `pool` dial trades
  concept ↔ identity) and `encode` (full-resolution VAE encode). Identity
  refinement steps, token cap with dedup, data multiplier for short clips,
  optional `av_encoder` input, and folder bulk-loading.
- **Load H3 RefMods** — LoRA-loader-style rows with a typed strength and a
  `copies` multiplier (2-10x — the manual row-duplication trick as a knob).
- **Load H3 RefMod Axis** — signed A/B sliders: negative picks the A mod,
  positive the B mod, one dial controls both.
- **Load H3 RefMod Folder** — every image/video in a folder as an ordered
  ref list.
- **Apply H3 RefMod** — one node for the pack's `MINIMAX_H3_COND` and the
  built-in `CONDITIONING`. `retention` master strength; a curve split into
  `curve_direction` (constant / concept_at_start / concept_at_middle /
  concept_at_end / concept_at_ends) + `curve_shape` (linear / ease /
  sigmoid / tanh / quadratic / cubic / exponential / stair / elastic /
  bump / dip) + `curve_value`; `scramble_seed`; optional curve-graph
  `debug` IMAGE output; shareable PNG graph presets (graph embedded in the
  image metadata, legacy `.json` still loads).
- **H3 RefMod Step Curve** — the same curve widgets, but over the **denoise
  timeline**: re-mixes every ref latent once per step (early steps lock
  composition/identity, late steps stay clean or refine detail) via a
  ComfyUI `DIFFUSION_MODEL` wrapper, attached between the model loader and
  the sampler.

### Reference math

- Weakening a ref blends toward a blurred copy of itself instead of noise
  or zero — stays on the latent manifold, so no grey/static output.
- The per-frame curve mixes each ref latent frame with
  `retention * curve(x)` instead of one flat strength.
- Greedy temporal dedup + budget-fit resampling make the token cap cheap.

### Misc

- `extract_mod.py` standalone CLI.
- Mods live in `ComfyUI/models/refmods/` (created on first run, next to
  loras/ and unet/); older mods in the pack's `mods/` folder still load.
