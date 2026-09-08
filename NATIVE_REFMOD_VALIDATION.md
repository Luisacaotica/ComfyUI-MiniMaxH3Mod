# Native RefMod validation — v0.3.2

## Environment

- Date: 2026-09-08.
- Branch: codex/character-voice-conditioning, local v0.3.2 changes.
- Upstream RefMod baseline: 7604ef4690168365b9db35c13a69e4c99421ace4.
- ComfyUI v0.34.0: fbed745c8d7d62573b099cd61fe51cb64b9b807e.
- Windows, Python 3.13.9, PyTorch 2.14.0+cpu, PyAV 18.1.0.

## Checks

Results: **98 regression tests passed**, **16 workflow schema/link checks
passed**, all 20 node registrations imported, and `git diff --check` passed.
All six native workflow layouts have non-overlapping saved node bounds and
their generation prompts contain Subject tags without manual Picture/Audio tags.

Run with ComfyUI's Python:

```text
python tools/run_tests.py /path/to/ComfyUI
python tests/comfy_smoke.py /path/to/ComfyUI
```

The regression suite covers:

- Original visual extraction versus extraction with audio: exact tensor/dtype
  equality for encode, mixed image sizes, repeat, token budget, pooling,
  refinement, merge, masking and motion-only inputs. Visual strength/curve
  outputs match the author's path, including after a safetensors round trip.
- One saved file retaining the original identity stack and all voice examples.
  Selected generation recordings do not remove unselected stored examples.
- Automatic slot-to-Subject assignment, sparse slots, copies, disabled slots,
  differing subject/speaker order, unchanged dialogue, wrong-bundle rejection,
  duplicate Apply and no mutation of the source CLIP.
- Earlier character files and native presentation parity with the same internally
  resolved prompt. Character table versions 1 and 2 remain readable.
- Real PyAV video demux with synchronized soundtrack. The author's pooled
  identity and full timed performance coexist without sharing a false clock.
- Eight distinct five-second voice references: 3,200 audio-reference tokens,
  eight separate owner labels, unchanged individual audio tensors, real H3
  PackedLayout and model-wrapper conditioning preparation.
- Three video_audio references plus three independent audio references: six
  reference voices with the correct internal Subject/Audio associations.
- Actual H3 transformer forward with eight reference voices, using a tiny model
  with random weights and small test tensors. Both output streams have the
  expected shape and finite values. This checks execution, not voice quality.
- Legacy RefMod nodes, metadata, caches, audio and previous binding experiments.

The smoke check validates all 16 workflow JSONs against actual node schemas and
links, including widget order for the six native examples. It also runs the
previous tiny two-/three-character attention regressions and all 20 node imports.

## Limits of these checks

The tokenizer vocabulary, node code, serialization, media decoding and H3 layout
are real. Large visual/audio VAE and Qwen forwards use CPU test doubles. The
small transformer has random weights. No released H3 checkpoint was used for a
render. Workflow node geometry is checked locally; the frontend has not been
inspected in a live ComfyUI session.

MiniMax documents three audio clips and 15 seconds total audio. The implementation
allows larger local experiments; successful packing/forward execution does not
establish reliable eight-speaker generation. The earlier description of that
15-second number as merely an invented local policy was incorrect.

Start rendering with two profiles, one clean recording per profile, copies=1,
retention=1 and scramble_seed=-1. Use only Subject labels in the visible prompt.
Then compare three and eight profiles while recording voice assignment, dialogue
accuracy, lip movement and appearance. Original latent equality does not imply
identical generated visuals after adding voice references and new conditioning.
