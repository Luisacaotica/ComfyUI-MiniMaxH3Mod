# Native RefMod validation — 2026-09-08

## Environment

- Extension: local v0.3.1 changes on `codex/character-voice-conditioning`.
- Upstream RefMod baseline: `7604ef4690168365b9db35c13a69e4c99421ace4`.
- ComfyUI: v0.34.0, `fbed745c8d7d62573b099cd61fe51cb64b9b807e`.
- Windows, Python 3.13.9, PyTorch 2.14.0+cpu, PyAV 18.1.0.
- No released H3 DiT, Qwen or VAE checkpoint was loaded for a quality render.

## Results

`python tools/run_tests.py /path/to/ComfyUI`: **93 tests passed**.

The new integration tests run the actual official `MiniMaxH3ReferenceToVideo`
node and MiniMax tokenizer with ComfyUI's bundled Qwen vocabulary. Large VAE and
text-model forwards use input-dependent codec doubles and synthetic text
embeddings. Tests also run real safetensors serialization, video/audio demux,
H3 packed layouts and the model wrapper's conditioning preparation.

Covered cases:

- Original Extract saves images and multiple recordings in one file; the
  original loader and Apply consume it. Resaving preserves every recording.
- Earlier character files load without conversion. Two profiles with three
  images and two recordings each retain all stored references but select one
  recording each by default: six images, two voices, 8.10 seconds in the fixture.
- Explicitly selecting all four fixture recordings packs 18.10 seconds without
  the previous blanket aggregate-duration rejection.
- Cached image/audio conditioning matches equivalent direct native reference
  tokens, latent blocks and empty AV output; the full prompt is unchanged.
- A real generated test video is demuxed through the original Extract node.
  Its saved video and soundtrack share the same H3-aligned interval.
- A compressed speaking-video reference preserves temporal geometry and its
  paired soundtrack. Spatial pooling does not resample the voice latent.
- Independent CLIP branches remain untouched. Additional ordinary references
  follow cached references in consistent presentation/block order.
- Wrong bundles, changed voice choices, duplicate Apply, scrambling, removing
  already-numbered profiles with zero retention and unsupported Continuum
  injection fail before sampling.
- Copies, strengths and selected-reference token budgets work without changing
  stored tensors. Ten saved recordings do not become ten render inputs by default.

`python tests/comfy_smoke.py /path/to/ComfyUI`: **passed**.

- All 15 character/native workflow JSONs match node schemas and graph links;
  the five new native workflows also match current widget ordering.
- All 20 existing/new node registrations import.
- Earlier native parity and experimental two-/three-character tiny-transformer
  regressions pass. These tiny random-weight forwards validate integration,
  not voice identity or generation quality.

`git diff --check`: **passed**. The five new workflow layouts have no overlapping
saved node bounds, including title spacing. Their frontend appearance has not
been inspected in a live ComfyUI browser session.

## Remaining render validation

Load the same profiles and new dialogue used for the user's successful
single-character test in the new native workflow. Then load two and three
profiles, using `voice_reference_N=1`, `copies=1`, `retention=1` and
`scramble_seed=-1`. Update the prompt's Picture/Video/Audio references from the
loader's printed map. Compare voice assignment, new-word accuracy, lip movement
and appearance across fixed seeds.

Correct storage and reference presentation do not establish a measured reduction
in voice swapping. That remains a checkpoint-level render test on the user's
ComfyUI machine. These local results do not claim a hard speaker-identity lock.
