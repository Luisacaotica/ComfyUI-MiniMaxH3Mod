# Character RefMods: appearance and voice

**For the simpler v0.3.2 workflow, use [NATIVE_REFMODS.md](NATIVE_REFMODS.md).**
It saves and loads combined appearance/voice files through the original RefMod
nodes and ComfyUI's official Reference to Video node. Your existing character
files work there. The separate character nodes documented below remain for
older workflows and optional experiments.

For the v0.3.0 extension on upstream `7604ef4`, start with
[the updated import and comparison guide](VOICE_TESTING.md). It adds a full-prompt
conditioner, an upstream image/audio RefMod importer, and three controlled
generation workflows while preserving the original workflows below.

The **MiniMax-H3 / character** nodes save one character's visual and audio
references together in a `.safetensors` file. You can create it from either:

- **Three images + two voice clips** of the same character. Each image and each
  recording is encoded separately and saved under the same character identity.
- **One speaking video with a soundtrack.** Video and audio are decoded from the
  same time interval and encoded as an H3 `video_audio` reference.

This is cached native H3 reference conditioning. It does not train a speaker
embedding, merge the modalities into a learned identity, or guarantee that H3
will never swap voices. The new conditioner keeps reference labels and tensors
in the same order and explicitly assigns each voice to its character. Real H3
renders are needed to measure whether that improves your scenes.

To test stronger association **during generation**, see the additional
[H3 Character Voice Binding experiment](VOICE_BINDING.md). It changes attention
inside H3 and includes two- and three-character comparison workflows. Saved
character files remain compatible.

## Install for testing

1. Update ComfyUI to a version with native **MiniMax H3 Reference to Video**,
   the `minimax` CLIP loader type, and H3 video/audio VAE support. Integration
   checks for this change used ComfyUI **0.34.0**, commit **fbed745**.
2. Put this repository, including the new files, in
   `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3Mod/`. If testing your fork, clone your
   fork there. Use one installation of this pack to avoid duplicate node IDs.
3. Install requirements using **the Python that runs ComfyUI**.

   Standard installation, from the ComfyUI directory:

   ```sh
   python -m pip install -r custom_nodes/ComfyUI-MiniMaxH3Mod/requirements.txt
   ```

   Windows portable, from the portable installation directory:

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3Mod\requirements.txt
   ```

4. Restart ComfyUI and refresh the browser. Search for **Extract H3 Character**.
   The added dependency is `av` (PyAV); its wheels include FFmpeg libraries.
   The companion `ComfyUI-MiniMaxH3` pack is optional for these new nodes.

Use your existing compatible **H3 Ref2VA diffusion checkpoint**, **H3 Qwen text
encoder**, **H3 video VAE**, and **H3 audio VAE**. The models are not included or
downloaded by this pack. Extraction needs only the two VAEs. Generation needs
the diffusion model, CLIP, and both VAEs for decoding.

## 1. Create a character from images and audio

Drag [01_extract_images_and_audio.json](examples/characters/01_extract_images_and_audio.json)
onto ComfyUI.

1. Select the H3 video VAE and H3 audio VAE in the two VAE Loader nodes.
   Check the actual `vae_name` dropdowns, not just the node titles:
   `minimax_h3_video_vae_fp16.safetensors` goes to `video_vae` and
   `minimax_h3_audio_vae_fp32.safetensors` goes to `audio_vae` (or equivalent
   H3 codec files). Still images require the video VAE too. An audio VAE
   connected to `video_vae` caused `too many values to unpack (expected 3)`
   in earlier builds; extraction now checks codec metadata before encoding.
2. Upload the three character images and two recordings. All five inputs
   should represent **one character and one voice**. One image and one audio
   clip also work; the additional inputs are optional.
3. Set `name`, for example `character_a`, and an appearance description.
   Leave recorded dialogue out of the description.
4. Choose `audio_start_seconds` and `audio_duration_seconds`. They apply to
   **each** audio clip separately; defaults select its first five seconds.
5. Queue the workflow. `save = true` writes:

   ```text
   ComfyUI/models/refmods/characters/character_a.safetensors
   ```

With registered `refmods` folders in `extra_model_paths.yaml`, new files instead
use `characters/` in the first registered root. Loading searches all registered
roots and the default location, including subfolders.

The two recordings stay separate inside the file. They are not averaged or
concatenated. Mono recordings become stereo and audio is resampled to 32 kHz
before encoding. `reference_short_edge` controls visual reference resolution;
larger references cost more memory and sampling time.

An existing filename raises an error. Enable `overwrite` to replace it, or
choose another name. Writes are atomic so an interrupted save cannot leave a
partially written character at the final filename.

## 2. Create a character from a speaking video

Drag [02_extract_speaking_video.json](examples/characters/02_extract_speaking_video.json)
onto ComfyUI.

1. Select both H3 VAEs.
2. Put the video in `ComfyUI/input/` and enter its filename in **Load H3
   Character Clip**, or enter its absolute filesystem path.
3. Select `start_seconds` and `duration_seconds`. Use a section showing one
   character speaking in their own voice.
4. Name the character in **Extract H3 Character**, then Queue.

You only connect `character_clip` to the extractor. That connection carries
both streams; separate image/audio connections are unnecessary. The loader's
other outputs are a preview frame, the selected audio, and interval information.

The loader samples video at 24 fps using source timestamps. It keeps audio
stream offsets, crops both streams to the same interval, and rounds duration
**down** to H3's valid `5 + 17*k` frame grid. For example, a requested five-second
clip becomes 107 frames, about **4.458 seconds**, with the matching soundtrack.
It does not squeeze a long recording into a short sequence of frames.

The saved file includes both VAE latents, sparse RGB frames for H3's vision
text encoder, their timestamps, and character metadata. After extraction,
generation can use the saved file without reopening the source video.

## 3. Generate new speech with a saved character

Drag [03_generate_one_character.json](examples/characters/03_generate_one_character.json)
onto ComfyUI. This includes sampling, video/audio decoding, and Save Video.

1. Refresh ComfyUI's node definitions after extraction, then select your file
   in **Load H3 Character**. Restart/refresh ComfyUI if the dropdown is stale.
2. Select your H3 Ref2VA checkpoint, H3 text encoder, and the two H3 VAEs.
   **CLIP Loader type must be `minimax`.** Replace every `SELECT_H3_...` placeholder.
3. Enter the scene, **new dialogue**, emotion, delivery, and language in
   **H3 Character Dialogue Conditioning**.
4. Queue. Generated video and speech are combined at 24 fps and saved under
   `ComfyUI/output/video/`.

You can also insert the two nodes into your existing native H3 workflow:
connect the new `positive` output to your guider and its `latent` output to
your sampler. Keep your proven H3 sampler/model settings for comparisons.
The example starts at 768 x 512; set your usual output resolution if different.

Reference recordings guide vocal identity. The `dialogue` field supplies the
requested new words; `emotion` and `delivery` describe how to speak them. The
node does not use the recordings as a target audio track or add them as
keyframe audio guides. Their acoustic content is still present in the reference
latents, so H3 can still copy words or produce incorrect speech.

The `resolved_prompt` and `reference_info` outputs show the final request and
assignments. Assignments are also printed in the ComfyUI console.

## Three characters

Use [04_generate_three_characters.json](examples/characters/04_generate_three_characters.json).
Create one file per character, select those three files, and enter each
character's new dialogue, emotion, and delivery. Blank dialogue keeps a
connected character silent. The example requests sequential turns in one shot.

For files made from **three images + two audio clips**, set each Load H3
Character node's `voice_reference` to **first** (already set in the example).
This uses three images and one saved voice clip per character, for totals of
nine images and three recordings. You can select `second` for a different saved
voice example. With a paired speaking video, `first` keeps its complete
video/audio pair.

These older character nodes check a combined budget of **nine images, three
videos and three audio-bearing references**. The previous 15-second aggregate
check has been removed from these local experiments. MiniMax does document
three audio clips and 15 seconds total; accepting more is experimental, not a
proven expansion of supported inputs or speaker capacity. Exceeding a count or configured token budget gives an
error instead of dropping another character's data. `max_reference_tokens = 0`
disables the additional user token cap. The simplified native workflow has
explicit per-character recording selection; see [NATIVE_REFMODS.md](NATIVE_REFMODS.md).

Try one character with new words first. Then compare the three-character
workflow against your native workflow using the same characters, dialogue,
output dimensions, and several fixed seeds. Record correct speaker/voice
assignment, exact dialogue, reference-word repetition, and lip sync. Saving
the references together is not evidence that those generation errors are solved.

## Compatibility and format

- Use **Extract/Load H3 Character** for the new files. The original visual
  RefMod nodes and the bundled example mod keep their existing format and use.
- The new `H3_CHARACTER` object connects to **H3 Character Dialogue
  Conditioning** or **H3 Character Reference Conditioning**. It does not connect to the companion pack's
  `MINIMAX_H3_COND` Apply input or the old visual strength/curve nodes.
- Character files retain full encoded reference latents at the chosen
  resolution. They can be much larger and contribute more sampling tokens
  than the old pooled visual mods. No joint training/refinement is implemented.
- Metadata is versioned (`h3_character_meta`, format version 2,
  `minimax_h3_ref2va`), with continued v1 reading. New profiles preserve prepared
  float32 vision pixels; v1 uint8 frames are converted when loaded. Loading validates tensor shapes, finite values, and
  paired timing. Corrupt or incompatible files are rejected.
- Saved references are cloned for each conditioning payload. Changing the
  selected file, dialogue, or cast rebuilds the mapping without a shared
  character cache. Reload detects changes to the saved file. Model attention
  changes only when you explicitly add and enable the experimental Voice
  Binding node.

## Developer checks

Run the combined upstream and character CPU tests with ComfyUI's Python and
dependencies. Tensor math, safetensors, media decoding, and resampling use real
libraries; extraction tests use small VAE test doubles:

```sh
python tools/run_tests.py /path/to/ComfyUI
```

With a current ComfyUI installation and its dependencies, run:

```sh
python tests/comfy_smoke.py /path/to/ComfyUI
```

The smoke check imports the whole node pack, checks the ten workflow graphs
against actual ComfyUI node schemas, and passes saved references through H3's
real presentation tokenizer, model conditioning adapter, and packed layout.
It compares the native node's prepared reference tensors, Qwen vocabulary tokens,
vision entries, positional layout, and output AV latent geometry against a saved
mixed cast. VAE and text-model forward passes are replaced by test doubles. An additional binding
check runs a tiny native H3 transformer with random test weights; these checks
do not load the released H3 weights or establish rendered image/voice quality.
