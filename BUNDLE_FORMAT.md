# RefMod single-file bundle format

Version 5 is a container of independent H3 references. It does not concatenate
audio with visual latents or change H3 conditioning semantics. Existing
standalone version-4 files remain supported and keep their `latent` tensor key.

## Detection and layout

Read JSON from the safetensors header's `refmod_meta` key without loading tensors.
A bundle has `_format_version: 5`, `kind: "bundle"`, a display `name`, and an
ordered `members` array containing 1–256 ordinary RefMod metadata objects.
Nested bundles are not supported. A member's `kind` is `image`, `video`, or `audio`.

```json
{
  "_format_version": 5,
  "kind": "bundle",
  "name": "hero",
  "members": [
    {
      "_format_version": 4,
      "name": "hero_visual",
      "kind": "video",
      "latent_t": 3,
      "latent_h": 32,
      "latent_w": 32
    },
    {
      "_format_version": 4,
      "name": "hero_audio",
      "kind": "audio",
      "latent_t": 200,
      "sample_rate": 32000
    }
  ]
}
```

This example omits optional member metadata. Each member keeps the usual mode,
source, tags, description, concept type and optional `refmod_config` JSON string.
Tensor `ref_0` belongs to member 0, `ref_1` to member 1, and so on.

| Kind | Tensor layout | Tokens |
| --- | --- | --- |
| image | `[1,24,1,H,W]` | `(H/2) × (W/2)` |
| video | `[1,24,T,H,W]` | `T × (H/2) × (W/2)` |
| audio | `[1,32,2,T]` | `2 × T` |

Visual H/W must be positive and even; T must be positive. Audio uses the existing
32 kHz H3 codec convention. Each member retains its own latent dimensions.

## Picker and loader integration

1. Detect `kind == "bundle"` and validate the supported format version.
2. Inspect members to display whether the file contains visuals, audio, or both.
3. Filter by **All**, **Visual** (`image` + `video`), or **Audio** before reading
   tensor data. Skip members whose modality strength is zero.
4. Load the selected `ref_i` tensors and expand them into ordinary
   `(H3RefMod, strength)` entries, preserving relative member order.
5. Multiply the slot strength by the modality strength. Account for each selected
   member's token cost and runtime copies.

In this pack, `bundle.load_bundle(path_without_extension, selection,
visual_strength, audio_strength)` performs selection and returns ordinary
entries with modality strengths. The caller applies the slot strength/copies.
Members retain their container `path` and `bundle_index` for Config updates.

Save writes each distinct reference object once; runtime strengths and duplicate
copies are not stored. Repacking a loaded bundle flattens its selected members.
Config updates preserve other members and unknown container metadata/tensors.

## Compatibility and limits

New readers load older standalone files. Older readers expecting only a `latent`
tensor will need bundle support; there is no promise of forward compatibility.
The standalone Save node remains available for exporting members.

The container carries encoded references, not original audio bytes, explicit
cross-modal timestamps, or speaker-to-character assignments. Shared storage does
not guarantee synchronized generation or stop multiple references from mixing.
Per-slot Audio strength affects all audio members in that slot; this initial UI
does not select individual audio tracks within a multi-audio file.
