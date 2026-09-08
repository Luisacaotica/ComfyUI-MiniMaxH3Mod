# Test character voices with the upstream update

This extension includes upstream commit `7604ef4` and adds complete reusable
character conditioning. H3 still generates the scene, speech and lip movement
together. Start with a single voice and new words, then test two and three voices.

## Install and choose a source

Install this complete checkout in `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3Mod`,
install its `requirements.txt` with ComfyUI's Python, and restart ComfyUI and
refresh the browser. Keep one installation of this node pack. The ordinary
upstream checkout alone does not contain the character extension.

Choose the workflow matching your files:

| Source | Workflow | Result |
| --- | --- | --- |
| Images and voice recordings | [01_extract_images_and_audio.json](examples/characters/01_extract_images_and_audio.json) | Independent image/audio references owned by one character. |
| Speaking video with its audio | [02_extract_speaking_video.json](examples/characters/02_extract_speaking_video.json) | Synchronized native `video_audio` reference. |
| Upstream image + audio safetensors | [07_import_upstream_refmods.json](examples/characters/07_import_upstream_refmods.json) | Reuses their latents and restores the image presentation from your original image. |
| Our existing character safetensors | Load H3 Character | v1 files remain readable; no conversion is required to load them. |

The importer accepts one character per invocation, encode-mode image RefMods,
and audio RefMods. Select both visual and audio files with strength 1 and copies
1. Connect the original source images in the visual files' loader order. The
importer needs no VAE, because it reuses the saved latents. Matching the correct
source image is your responsibility: old files contain no source-image hash.
Masked, edited or differently cropped sources may not match the supplied image.

An upstream video/stacked-image RefMod cannot safely be converted into a paired
speaking reference: source timing and original presentation may have been lost.
Re-extract that speaking video using workflow 02. Matching `_visual` and `_audio`
filenames is insufficient to reconstruct synchronization.

New extractions use the native H3 Lanczos image preparation and torchaudio
resampling for standalone audio. New v2 files preserve prepared float RGB pixels
alongside the audio/video latents. Earlier v1 profiles keep their original
preparation and byte-frame data; re-extract them for comparisons of the new
preprocessing. Saving v1 as v2 cannot restore information already discarded.

Character saves use `characters/` in the first registered `refmods` root, or
`ComfyUI/models/refmods/characters/` by default. Loading also searches other
registered roots and the default location; earlier roots take precedence for
duplicate relative filenames.

## First render

1. Open [08_compare_1_characters.json](examples/characters/08_compare_1_characters.json).
2. Select your character, H3 **Ref2VA** checkpoint, `minimax` CLIP encoder and both
   H3 VAEs. Select `voice_reference=first` to test one saved voice example.
3. Begin with `reference_presentation=native`, `av_layout=paired`, Voice Binding
   `mode=off`, and the fixed seed. Replace the short diagnostic dialogue with
   new words before beginning the comparison set.
4. Queue and listen for the source speaker's voice saying the new words. Also
   check intelligibility, expression and the correct face's mouth movement.

The new **H3 Character Reference Conditioning** accepts your complete prompt
unchanged. `<Subject 1>` means the first character input, but source asset labels
follow H3's modality order. The console and `reference_info` report the actual
mapping. The supplied comparison prompts assume **one speaking-video reference
per character**. For image/audio profiles, use the reported `<Picture N>` and
`<Audio N>` labels. Settle the prompt once before comparing mechanisms.

You may also keep using **H3 Character Dialogue Conditioning** and workflows
03–06 for automatic dialogue/reference definitions. Both conditioners use the
same native reference assembler and work with Voice Binding.

## Five controlled variants

Use a speaking-video profile for this experiment. Change only the switches in
this table, retaining the prompt, source profiles, seed, output dimensions,
frame count, model files, sampler, steps, shifts, and speaking windows.

| Trial variant | Reference presentation | AV layout | Voice Binding mode | What the comparison tests |
| --- | --- | --- | --- | --- |
| `latent_separate` | `latent_only` | `separate` | `off` | Diagnostic baseline without native reference presentation or shared AV packing. |
| `native_separate` | `native` | `separate` | `off` | Adds native reference presentation. |
| `native_paired` | `native` | `paired` | `off` | Adds the shared video/audio reference timeline. Recommended normal configuration. |
| `pair_bias` | `native` | `paired` | `pair_only` | Adds soft reference-audio attention toward its character's visual references. |
| `scheduled_bias` | `native` | `paired` | `scheduled` | Also biases generated audio toward the assigned voice during explicit speaking windows. |

Start with `pair_bias=0.5`, `voice_bias=0.5`, `boundary_fade_seconds=0.1` in all
five runs. Off/pair-only modes ignore the inapplicable controls. The full-prompt
conditioner does not insert speaking windows into your prompt: keep the same
timing in both the prompt and `turn_schedule`. Blank schedules are valid with
binding off or pair-only. Scheduled mode needs nonoverlapping windows; it cannot
detect when H3 actually speaks or force its dialogue to fit those windows.

`latent_only` is an ablation of this character pipeline, not a claim of exact
parity with every upstream Apply configuration. Splitting AV changes both the
reference positions and the downstream target origin, as native packing dictates.
The stored tensors and prompt stay identical. With image/audio-only profiles,
paired and separate have the same layout and are not distinct experiments.

The case fingerprint covers characters, source tensor contents, prompt, output
geometry and schedule, and should remain identical across these variants. It
does not include the deliberately changed switches. It also cannot identify
model weights or sampler settings: keep those fixed and record them separately.

After a successful one-voice check, use:

- [09_compare_2_characters.json](examples/characters/09_compare_2_characters.json)
- [10_compare_3_characters.json](examples/characters/10_compare_3_characters.json)

Use one short voice reference per character and at most 15 seconds total audio.
Start with the same three seeds for each variant. Disable conflicting attention
overrides, block replacements and compilation while evaluating Voice Binding.
See [VOICE_BINDING.md](VOICE_BINDING.md) for its mechanism and boundaries.

## Record and compare results

[voice_trials.csv](examples/characters/voice_trials.csv) contains 45 **pending**
rows: three cast sizes, three seeds, five variants. Complete one case/seed's five
variants first. Do not discard failed renders.

For each row, copy `case_fingerprint` from the console or `reference_info`, and
record `run_settings` using the same descriptive identifier for an unchanged
model/sampler configuration. Keep the full workflow with the video. Use status
`complete` for an evaluated render, or `failed` with the error/reason in `notes`.
The quality fields are manual listening/visual assessments, not automatic scores.

Record the number of correctly assigned voices and yes/no answers for exact new
dialogue, copied reference words, intelligibility, lip sync/correct speaking face,
and requested emotion. Enter render time and the output filename. Reference
repetition means **source_repeated=yes**; the other quality fields are positive
when they are **yes**.

Run with any Python 3 installation, from this repository:

```sh
python tools/voice_trials.py summarize examples/characters/voice_trials.csv
```

The report compares only groups with all five variants finished. It explicitly
lists incomplete groups, rejects mismatched fingerprints/settings and duplicate
runs, counts failed renders as unsuccessful, and reports voice accuracy separately
from whether all quality checks passed. A correct voice with gibberish or copied
words does not pass the full quality check. No results are prefilled.

Create a separate fresh sheet without overwriting an existing one:

```sh
python tools/voice_trials.py init my_next_voice_trials.csv
```

If a setting helps, test it on fresh dialogue and seeds. Compare smaller/larger
bias values in separate matched sheets. A persistent native-equivalent failure
would motivate actual H3 adapter/latent training with speaker and dialogue losses,
as described in [the plan](VOICE_IMPLEMENTATION_PLAN.md); that training is a
separate research stage and is not implemented here.

## Verification available locally

See [CHARACTER_VALIDATION.md](CHARACTER_VALIDATION.md). The automated checks cover
storage, preprocessing, reference presentation, ownership, native packing, graph
schemas and the attention mechanism. They do not establish voice fidelity using
the released H3 weights. Your ComfyUI renders supply that evidence.
