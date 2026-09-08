# Appearance and voice with the official H3 node

Use **Extract H3 RefMod → Load H3 RefMods → Apply H3 RefMod**, together with
ComfyUI's original **MiniMax H3 Reference to Video**. Your entire Ref2VA prompt,
dialogue and emotions belong in that official node's `prompt` field.
The older experimental character dialogue/binding nodes are not needed.
In v0.3.2, **loader slot N supplies Subject N automatically**. Write your scene
with Subject tags; you do not need to write Picture/Audio associations.

## Ready-to-open workflows

Use the files in **[examples/native_refmods](examples/native_refmods)**:

| File | Purpose |
| --- | --- |
| [01_extract_images_and_audio.json](examples/native_refmods/01_extract_images_and_audio.json) | Original Extract node: 1–3 images and 1–2 audio examples, saved together. More inputs can be added with its audio/image input controls. |
| [02_extract_speaking_video.json](examples/native_refmods/02_extract_speaking_video.json) | Original Extract node: a video file including its soundtrack, saved together. |
| [03_generate_one_character.json](examples/native_refmods/03_generate_one_character.json) | Load one profile and generate through the official node. |
| [04_generate_two_characters.json](examples/native_refmods/04_generate_two_characters.json) | Load two profiles in one ordinary RefMod loader. |
| [05_generate_three_characters.json](examples/native_refmods/05_generate_three_characters.json) | The same workflow with three profiles selected. |
| [06_generate_eight_characters_experimental.json](examples/native_refmods/06_generate_eight_characters_experimental.json) | Eight distinct profiles/voices; experimental capacity test, not proven voice accuracy. |

Install this complete version of the pack, restart ComfyUI and refresh the
browser. Drag the appropriate JSON onto the canvas. Select your actual model
files and character files. Cached profiles do not need their original media
uploaded again during generation.

## Save a character

In **Extract H3 RefMod**, select `mode=encode`, `ref_resolution=768`,
`identity=0`, `multiplier=1`, `merge=False`, `motion_only=False` for initial tests.
Visual processing now calls the author's original extractor. Its resizing,
mask, pooling, identity refinement, merge, latent_frames, multiplier and
max_tokens controls retain their visual behavior. max_tokens limits the
original visual stack; loader/Apply max_total_tokens limits the full selected
bundle including audio and paired performances. The examples disable these
optional caps with 0. Audio is encoded separately, never averaged into the
visual latent or trained into a speaker embedding.

Connect `minimax_h3_video_vae_fp16.safetensors` to `vae` and
`minimax_h3_audio_vae_fp32.safetensors` to `audio_vae` (or equivalent H3 files).

- **Images plus voice recordings:** connect images to the image inputs and
  one recording to `audio`. Additional recordings go to the growing audio
  inputs. All references belong to the same character. Unused Load Image/Audio
  nodes can be deleted. `audio_start_seconds=0`, `audio_duration_seconds=5`
  takes up to the first five seconds of each recording independently. Use
  prepared/trimmed clips if different recordings need different start times.
- **Speaking video:** upload the file into the cloud ComfyUI input directory
  and enter its filename in `video_file`, or use its server-side absolute path.
  This reads both video and sound from the same selected interval. Frame and
  audio durations are aligned to H3's valid 24 fps temporal grid. Plain IMAGE
  video-frame inputs cannot carry sound; use `video_file` for synchronized
  evidence. Separate frame and audio inputs are treated as independent references.
  The file contains the original sampled/compressed identity stack **plus** a
  full synchronized video/audio performance. These require two visual encodes
  at extraction and additional reference tokens at generation. This preserves
  the author's identity processing without pretending sampled identity frames
  still have the source soundtrack's timing. The file reader caps decoded video
  at a 2048-pixel long edge before extraction; connected IMAGE inputs retain
  their original preprocessing path.

Set `name`, `subfolder=characters` and `save=True`, then Queue. The extractor
runs without requiring an extra downstream Save or Preview node. It writes one
`models/refmods/characters/name.safetensors` (or your registered RefMod root).
As with the original extractor, an existing destination is replaced.

## Generate with your existing prompt

Wire these connections (already made in workflows 03–06):

```text
CLIP Loader ──> Load H3 RefMods.clip ──> MiniMax H3 Reference to Video.clip
                        │                           │ positive
                        └── mods ───────────────> Apply H3 RefMod ──> guider
                                                    
MiniMax H3 Reference to Video.latent ──> sampler.latent_image
```

The loader's new CLIP output prepares the saved image/video presentation and
numbered audio labels and subject associations for the **official** node's text encoding. Apply supplies
the corresponding saved visual/audio latents. There is no duplicate VAE encode
or additional text-model encode for cached references. No ComfyUI core file,
global tokenizer, official node class or model weight is modified.

Use the same loader's `mods` and `clip` outputs. Keep `copies=1` and
`scramble_seed=-1` initially. Scrambling after text encoding is rejected for
combined profiles because it would change the relationship between labels and
latents. Do not connect the same character's original images/audio to the
official node again; that would add a second set of references.
Keep `retention=1` for initial tests. To disable a character, set its loader
strength to zero so both its presentation and latents are removed together.
The Continuum bridge cannot prepare these labels; combined profiles use Apply.

Write your prompt normally:

```text
subject_definitions:
<Subject 1> is Character A.
<Subject 2> is Character B.

retention_analysis:
<Subject 1>: fully_preserved - retain appearance and voice identity.
<Subject 2>: fully_preserved - retain appearance and voice identity.
```

Fill in your remaining scene sections and dialogue in the official node. The
loader associates the references internally, for example:

```text
mod_1 -> Subject 1 -> saved appearance + selected voice for Character A
mod_2 -> Subject 2 -> saved appearance + selected voice for Character B
```

The visible prompt remains unchanged. Before text encoding, the loader adds
appearance/voice associations inside subject_definitions (or creates that
section when absent). Dialogue, scene text and explicit S speaker IDs are not
rewritten. Slot numbers are not compacted: mod_8 is Subject 8 even if slots 1–7
are empty. Repeating a slot adds references for the same subject. The same file
in two different slots intentionally defines two subjects with the same identity.

The diagnostic preview shows actual internal labels. Conditioning metadata
`h3_refmod_native` records user_prompt, resolved_prompt and assignments for
debugging. It is not an extra required prompt field. Existing prompts written
with manual Picture/Audio associations should have those old mapping lines
removed if they conflict with the slot-to-Subject convention.

New original-stack files store a first-view image for text-encoder presentation;
the original visual latent retains the author's encoded stack/merge. Earlier
files keep their original saved presentation. A generated scene can change
because audio and conditioning have been added, even when the original visual
latent is numerically identical.
Additional raw references in the official node are numbered after cached
references of the same type. Legacy latent-only RefMods remain usable through
the usual Apply path, but they do not acquire missing reference presentation.

Follow [MiniMaxAI's Ref2VA guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape`, `non_diegetic_music`. Dialogue remains inside `<d>` tags
in the detailed description. Our sample is a short diagnostic; replace it with
your complete scene prompt.

## Multiple recordings and the reported 18.10-second error

Each loader slot has `voice_reference_N`:

- **1 (default):** use the first stored recording for the character in slot N.
- **2, 3, …:** use that numbered stored recording.
- **0:** explicitly use all stored recordings; this can increase cost and has
  not been shown to improve speaker matching.

Selecting a recording affects generation, not the saved file. Saving/configuring
a loaded profile retains every recording. A selected paired video keeps its
own soundtrack; other paired performances are not silently split into stills.

Your old error counted six images, four recordings and 18.10 seconds from two
profiles. The default selects one recording per profile. MiniMax's model card
does document at most three audio clips and 15 seconds total audio. ComfyUI also
accepts three video soundtrack inputs separately from its standalone audio
inputs, and its lower-level packing has no fixed three-speaker table. The cached
path permits larger experiments with explicit token budgets. Exceeding documented
input specifications is not established support or a guarantee of more voices.
See [MiniMax's input specifications](https://huggingface.co/MiniMaxAI/MiniMax-H3#model-variants-and-input-specifications).

The older `examples/characters` workflows keep their separate character-node
interfaces for compatibility. Their three-audio-count check remains, and can
be satisfied with `voice_reference=first`; use `examples/native_refmods` for
the simplified workflow above.

## Existing files and validation

Earlier `h3_character_meta` files, including your existing character profiles,
appear in **Load H3 RefMods** without re-extraction and gain automatic subject
association. Their saved latents are not rewritten. To use the restored original
visual extraction on an old profile, re-extract it from its original sources.
New combined saves use `refmod_meta` format 5 with character table version 3;
table versions 1/2 remain readable. Ordinary older visual/audio RefMods continue
using their original format.
Install the updated pack to read combined files; the unmodified author's
loader does not understand them.

The plan is in [NATIVE_REFMOD_AUDIO_PLAN.md](NATIVE_REFMOD_AUDIO_PLAN.md).
The tested environment and results are in [NATIVE_REFMOD_VALIDATION.md](NATIVE_REFMOD_VALIDATION.md).
Tests cover the official node, actual H3 tokenizer vocabulary, packed layouts,
model conditioning, file roundtrips, source-video demux and branch isolation.
Large VAE/text-model forwards use CPU doubles; this is not GPU render-quality
validation. The user's single-character result is encouraging evidence for
their configuration; multi-character voice accuracy still needs render tests.
Automatic association uses H3's existing conditioning, not LoRA training or a
hard attention mask. The separate experimental Voice Binding nodes are not
automatically applied. No claim of eight-speaker voice accuracy has been measured.
