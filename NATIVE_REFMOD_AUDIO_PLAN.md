# RefMod audio with the official ComfyUI workflow

Implemented locally as v0.3.1. CPU regression and integration checks passed
on 2026-09-08; see [validation details](NATIVE_REFMOD_VALIDATION.md).
The next validation step is rendering with the user's H3 checkpoint and profiles.

## User-facing contract

Extend Extract H3 RefMod, Load H3 RefMods and Apply H3 RefMod. Keep the
unmodified official MiniMax H3 Reference to Video node and its full Ref2VA
prompt. Do not require the experimental character dialogue or binding nodes.

Extraction saves one file containing visual references, voice recordings,
presentation images and genuine video/audio timing. Visual-only legacy
extraction keeps its current behavior. Existing character files remain usable.

Generation wiring:

    CLIP Loader -> Load H3 RefMods.clip -> official Reference to Video.clip
    Load H3 RefMods.mods -> Apply H3 RefMod.mods
    official Reference to Video.positive -> Apply H3 RefMod.conditioning -> guider
    official Reference to Video.latent -> sampler

The complete user prompt stays in the official node. The loader's CLIP output
is a local adapter around a cloned encoder: it adds cached reference presentation
to tokenization and records the exact source bundle in conditioning metadata.
Apply checks that metadata, then attaches the matching saved latents. No core
ComfyUI edits, node replacement, global tokenizer patches or second text encode.

## Changes

1. Extend the RefMod reader/writer to recognize combined profiles and earlier
   h3_character_meta files, including recursive discovery, cache accounting and
   resaving all stored recordings even when only one is selected for generation.
2. Extend the original extractor with optional audio/audio VAE, multiple audio
   inputs and a speaking-video file input. Keep each reference separately;
   preserve a matched time interval for paired clips. Compression, if used,
   must not resample speech or destroy a paired video's temporal grid.
3. Add a CLIP input/output to the existing loader, keeping existing output
   indices. Select one voice recording per profile by default. Report stable
   Picture/Video/Audio labels and the selected voice; all source recordings
   remain in the file. Advanced selection can use another recording or all.
4. Extend Apply to expand combined entries into native reference blocks. Check
   encoder/bundle correspondence, avoid double application, and reject
   post-encoding scrambling of combined references rather than silently
   changing assignments. Preserve original behavior for ordinary RefMods.
5. Replace the previous blanket 15-second aggregate rejection with explicit
   selected-reference reporting and configurable token budgets. Distinguish
   native UI slot counts from demonstrated model limits. More reference audio
   is not an automatic quality improvement.
6. Supply minimal extraction and one/two-character generation examples using
   the official node. Keep old node IDs for saved workflow compatibility, but
   direct users to the simpler workflow.

## Acceptance checks

- The original extractor saves one combined safetensors file, and the original
  loader and Apply consume it. Legacy visual/audio files still work.
- Earlier character files load without recreating profiles. Two profiles with
  two recordings each select one voice each by default; their 18.10 seconds of
  stored audio do not trigger the previous blanket error.
- Official-node prompt text reaches its tokenizer unchanged. Cached and
  equivalent native references have matching presentation and latent payloads.
- Matched speaking-video timing survives storage and Apply. Independent images
  and audio are never mislabeled as a synchronized performance.
- The base CLIP and other branches are untouched. Wrong bundles, duplicate
  Apply and incompatible scrambling fail before sampling.
- All new examples validate against real ComfyUI node schemas. Regression
  tests use real serialization and ComfyUI code; GPU render quality remains a
  separate user test. The user's successful single-character render is the
  baseline, not proof of multi-character voice locking.

Reference: MiniMaxAI's VIDEO_PROMPT_WRITING_GUIDE_ref_en.md; the six sections,
global speaker IDs and dialogue tags stay in the official prompt field.
