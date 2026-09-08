# Character extension validation

Date: 2026-09-08. Extension version: 0.3.0, unreleased.
Upstream base: `7604ef4690168365b9db35c13a69e4c99421ace4`.
Local branch: `codex/character-voice-conditioning`.

## Environment

- Windows, CPU-only verification.
- Native ComfyUI 0.34.0, commit `fbed745c8d7d62573b099cd61fe51cb64b9b807e`.
- Real PyTorch, safetensors, PyAV, torchaudio, native H3 tokenizer/model/layout
  code and ComfyUI node schemas.
- No released H3 diffusion, audio VAE, video VAE or Qwen model weights loaded.

## Results

| Check | Result | What it establishes |
| --- | --- | --- |
| Combined upstream + character suite | 81 tests passed | Upstream loader/audio/Master/bridge behavior; character media/storage/import/binding and trial validation. |
| Native cached-reference comparison | Passed | Same prepared source intervals produce equal reference tensors, Qwen token IDs/vision entries, native reference position IDs and AV output geometry after character save/load. |
| Workflow schemas and links | 10 workflows passed | Existing and new graphs match the installed node input/output schemas and have consistent connections. |
| Node registrations | 20 registered | Upstream nodes coexist with all seven character nodes. |
| Tiny native H3 forward passes | Passed | Two/three-character attention controls change finite outputs; off/zero is exact bypass; weights and unpatched options remain unchanged; incompatible/bypassed hooks fail explicitly. |
| Real voice fidelity/assignment | Not run | Requires the user's H3 checkpoint and reference clips on their ComfyUI machine. |

The native comparison uses the actual Qwen vocabulary shipped with ComfyUI.
Text-model forward and VAE encoding use deterministic test doubles. The VAE
doubles depend on the supplied pixels/waveform to catch mismatched preparation
and routing, but cannot establish real-codec numerical equivalence or audio
quality. Image preparation is checked against native Lanczos, and standalone
resampling against native torchaudio. Tiny transformer checks use random weights.

Old v1 profile reads, exact v2 pixel/tensor round trips, synchronized media
offsets, malformed files, token limits, ownership after sorting, stereo speaker
windows, save isolation, imported legacy audio metadata, registered folders and
results-sheet rejection of mismatched runs are covered by automated checks.

## Reproduce

Using ComfyUI's Python, from this repository:

```sh
python tools/run_tests.py /path/to/ComfyUI
python tests/comfy_smoke.py /path/to/ComfyUI
python tools/voice_trials.py summarize examples/characters/voice_trials.csv
```

The supplied trial sheet contains no completed results. Its summary should
report zero matched completed groups and list pending groups explicitly.

## Still to measure

Follow [VOICE_TESTING.md](VOICE_TESTING.md). Compare the same seeds, source
profiles, prompt, target dialogue, timing and model/sampling configuration across
the five supplied variants. Record failures, incorrect voices, dialogue errors,
reference repetition, mouth alignment, emotion and runtime. The current local
checks establish implementation behavior; they are not a measured reduction in
voice swapping or proof of a trained identity lock.
