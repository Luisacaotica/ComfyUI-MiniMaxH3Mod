# Changelog

All notable changes are tracked here. Each version is also published as a
[GitHub Release](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/releases),
so you can keep using an older version if a new one changes something you
rely on.

## v0.3.0 — unreleased character extension

- Integrate upstream v0.2.0 at `7604ef4`, preserving Master/audio/library/loader
  fixes and upstream node IDs.
- Add **H3 Character Reference Conditioning** for unchanged full prompts,
  native reference presentation, diagnostic AV packing controls, and content
  fingerprints for matched trials.
- Add **Import H3 RefMods as Character** for upstream image/audio files, using
  original source images to restore vision presentation without re-encoding
  latents. Reject pooled visuals, duplicate copies and unpairable video stacks.
- Introduce character format v2 with float vision pixels and v1 read support;
  match native image preparation and standalone-audio resampling. Honor
  registered RefMod roots while retaining default-folder discovery.
- Add import and one/two/three-character comparison workflows, a blank 45-run
  sheet, and a results tool that rejects unmatched runs and includes failures.

- Add **H3 Character Voice Binding (Experimental)**: soft attention bias from
  each reference voice to its character's visuals, and optional timed bias
  from generated audio to its assigned reference voice. Uses H3's native
  attention hook on a cloned model, with an exact off mode and compatibility
  guards. No weights or stored latents are changed.
- Complete the six-section reference prompt and number speakers by actual
  speaking order. Add optional validated speaking windows shared by the
  prompt and attention controls.
- Add two-/three-character comparison workflows and a blank trial sheet;
  test numerical attention behavior and tiny native H3 forwards. Voice-swap
  reduction with released H3 weights remains unmeasured.
- Add **Load H3 Character Clip**, **Extract H3 Character**, **Load H3
  Character**, and **H3 Character Dialogue Conditioning**.
- Save one character's image references and separate voice examples, or a
  synchronized speaking video/audio pair, in a versioned safetensors asset.
- Preserve source audio timing, normalize voice input to stereo 32 kHz, and
  align paired clips to the native H3 temporal grid. Add PyAV dependency.
- Build native H3 reference labels and conditioning payloads together for up
  to three characters, with separate new dialogue/emotion/delivery inputs.
- Validate shapes, timing, and reference budgets; save atomically; detect
  modified files on reload. Preserve the original visual RefMod nodes.
- Include four workflow JSON examples, a setup guide, CPU regression tests,
  and a smoke check against the actual ComfyUI H3 implementation.
- This is reusable reference conditioning, not learned speaker training or
  a verified fix for voice swapping. Render quality still requires H3 testing.

## v0.2.0 — 2026-09-07

### Known limitation: voice transfer

Audio reference support includes an observed music-transfer example, but current
speaker-identity tests failed. This release does not provide working voice cloning.
The cause has not been isolated between H3/checkpoint behavior and the reference
presentation difference in the current RefMod integration.

### Follow-up implementation (2026-09-07)

- Clarify Master saving logs: deferred extraction, Created/Replaced destinations,
  explicit save=False and a final token/file summary; report saved_paths in details.

- Add Save H3 RefMods as an output node: save mixed bundles without connecting
  a Preview or sampler, with prefix/subfolder options and copy deduplication.

- Save new mods/presets in the first registered refmods root (extra_model_paths);
  keep the standard models/refmods fallback and existing-file Config locations.
- Remove sibling-pack imports and startup dependency warning. Legacy av_encoder
  inputs use native ComfyUI video VAE loading; standard VAE inputs are preferred.

- Add Extract H3 RefMod Master: optional image/video and audio inputs, sequential
  extraction through the shared extractors, combined token budget, one bundle
  output and separate compatible visual/audio files under a shared name.

- Step Curve now mixes the current payload without retaining tensors and handles
  keyframe offsets, audio refs, empty conditioning and chained curves.
- Bridge uses the MODEL outer-sampling hook with finally cleanup; removes global
  state and third-party monkey patches. The old Disarm node is deprecated.
- Align image/mask center crops; share resize and causal-frame helpers with CLI.
- Invalidate loaded files and queue cache on changes; cap cache at 256 MiB;
  enforce copies/numeric limits; bound video decode buffers with unknown lengths.
- Add native audio RefMods, legacy AudioMod import, audio Extract, library UI,
  inspection/optional previews, save subfolders, explicit scramble modes,
  extraction presets and total-token budgets.
- Atomic saves, deduplicated config writes and grouped mean-MSE refinement.
- Historical entries below describe earlier unreleased revisions; the behavior
  above and the current README supersede their bridge/cache descriptions.

### Fixed: linked loader inputs and subfolders

- Both loaders accept unresolved/empty mod names and connected STRING/combo
  outputs. Fixed missing names still fail validation; connected names resolve
  at execution with a missing-file error when necessary.
- Discover RefMods recursively as `folder/name`, preserving metadata filtering
  and skipping graph presets. Listing cache tracks nested files, nanosecond
  timestamps, and legacy JSON sidecars.
- Normalize Windows separators and check relative paths at execution. Cache
  mods by file path so identical metadata names in different folders do not
  collide when saving their configuration.
- Add regression tests using real safetensors and ComfyUI queue validation.

### New: Continuum integration

- **Continuum RefMod Bridge** node — injects your mods into
  [ComfyUI-H3-Continuum](https://github.com/xiaolibai-sys/ComfyUI-H3-Continuum)'s
  chunked long-video samplers (V2/V3/V3.4), which build their conditioning
  internally and have no CONDITIONING socket. While armed, every chunk
  carries the bundle's ref blocks as native `minimax_refs` — persistent
  across all chunks, no Continuum markers so its continuity layout never
  re-times them. Version-checked against one public seam; if Continuum
  changes internally the bridge reports and degrades to no-injection rather
  than breaking a run. One active bundle per graph; per-run scramble seeds
  work normally. Wire it on the sampler's model line (Load Model -> bridge
  -> sampler `model`): ComfyUI never executes fully disconnected nodes, so
  the earlier "anywhere upstream" placement silently did nothing.

### Fixed: why insertions felt weak, random, or "possessed"

Three defects were found and fixed; each is reproducible with the new
`gauntlet_harness.py` (run it with ComfyUI's Python — no server needed).

- **Apply's default curve silently HALVED every multi-frame mod.** The old
  default (`concept_at_end` + `ease`) runs its envelope across the mod's own
  ref frames — mean multiplier ≈ **0.50** at any frame count — so video/
  stacked mods shipped half-blurred unless you touched widgets. Worse, ref
  tokens are not bound to output-video time at all: no frame curve can place
  a concept "at the end of the video"; it only weights which reference
  content dominates. Defaults are now `constant` + `linear` + `1.0`
  (= full strength, official-ref parity); saved workflows keep their saved
  values. The curve widgets remain for stack weighting, with honest tooltips.
  For real output-timing control use **H3 RefMod Step Curve** (denoise
  timeline).
- **Image-kind mods ignored every curve dial silently** (`curve_strengths`
  short-circuits at 1 latent frame), so single-image identity mods always
  injected at full strength — the "concept bleeds into everything" case.
  Now `curve_value` acts as a plain strength cap on single-frame mods
  (e.g. value 0.4 = the ref blends 40% toward its blurred self), with a
  console note.
- **H3 RefMod Step Curve leaked state across generations.** The wrapper
  cached pristine/blurred latents plus the marked-ref index mapping on the
  first forward and never reset them — but ComfyUI reuses the patched MODEL
  and conditioning payload across queued runs, so run #2 mixed run #1's
  cached tensors whenever the scramble seed/mods/strengths changed
  (corrupted, "ghost" output). The wrapper now keys its state on the
  injected refs' identities (which it never mutates), re-keying cleanly on
  every new run while keeping pristine stable within one; chaining multiple
  Step Curve nodes composes deterministically, and progress is normalized to
  each run's actual schedule start (partial denoise included).

Also new: Apply now prints each row's *effective* strength (row × retention
× curve-mean) and flags anything below 0.30 — the usual suspect when an
insertion feels weak.

### New: fixed configs, shipped with the mod

- **Fix H3 RefMod Config** node — bakes tuned Apply/Step-Curve settings
  (`retention` + `curve` + `step_curve`) into a mod's safetensors metadata
  and re-saves the file in place, so a concept ships with the settings that
  make it work.
- **`override` toggle** on Apply H3 RefMod and H3 RefMod Step Curve — reads
  the first mod in the bundle that carries a fixed config and uses those
  settings instead of the widgets; falls back to the manual parameters
  (with a console note) when no mod has one.
- The debug curve graph reflects the overridden curve, so what you see is
  what actually ran.

### New: merge mode on Extract H3 RefMod

- **`merge` toggle** (training mode): one shared latent refined jointly
  against every reference's full encode (mean reconstruction error) instead
  of stacking each ref's own pooled latent. A whole collection becomes ONE
  tiny consensus mod — pulled toward what's common across all the views
  (structure/motion/identity) rather than any single shot's framing or
  background, at one ref block's worth of tokens. `identity` dials the
  joint refinement; masks and `background_retention` still apply per ref.
- **`motion_only` toggle** (training mode, experimental): video refs are
  converted to per-frame temporal differences before encoding, so the mod
  carries where/how things move instead of what they look like — static
  appearance (lineart look, background) never enters the latent. A soft
  motion guide, not a ControlNet (the ref channel is content-based).

### Fixed

- **Curve directions now mean what they say.** `concept_at_end` actually
  puts the concept at the **end** of the video (and at the end of the
  denoise timeline on H3 RefMod Step Curve); `concept_at_start` puts it at
  the start. The old envelopes were inverted relative to the names. Legacy
  `decrease`/`increase` workflow values and the preset names
  (`fade_in`/`fade_out`/`bump`/`dip`) still resolve to their original
  envelopes, so old workflows keep their behavior.
- **H3 RefMod Step Curve no longer touches your input video.** The wrapper
  used to re-mix *every* ref latent, including the original video's
  identity anchor — that blurred the source subject and let a different
  person "pop". It now identifies the refs injected by Apply H3 RefMod (via
  a marker on the blocks) and only re-mixes those; native ref2va refs pass
  through untouched, and a console note explains when there is nothing to
  re-mix.

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
