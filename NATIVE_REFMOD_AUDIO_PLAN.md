# RefMod audio integration plan — v0.3.2

## Contract

Keep the author's visual extraction and ordinary Subject-based prompting.
Extend the existing Extract H3 RefMod, Load H3 RefMods and Apply H3 RefMod.
Use the unchanged official MiniMax H3 Reference to Video node. No new required
node types, dialogue fields, manual Picture/Audio mapping or model weight edits.

## Implemented design

1. Call the original extractor with the same ordered visual sources and visual
   settings. Save its resulting latent unchanged in the combined character file.
   Reuse the original visual strength/curve operation during Apply. Audio inputs
   are encoded separately and do not participate in visual refinement.
2. Keep multiple encoded voice examples in that same safetensor. Select one per
   character by default; selecting another or all never deletes recordings.
3. For a speaking file, decode a matched interval. Store both the original
   sampled/compressed identity stack and a separate native video_audio performance
   with a real shared timeline. This requires extra visual encoding and tokens;
   a pooled identity stack is never falsely treated as lip-synchronized video.
4. Map runtime loader slot N to Subject N. Preserve gaps, disabled slots, copies
   and the user's S speaker IDs. Add native appearance/voice associations inside
   subject_definitions before the official text encode; leave the visible prompt,
   scene description and dialogue intact. Store original/resolved text in metadata.
5. Supply saved reference presentation through a scoped CLIP clone, then inject
   matching blocks through Apply. Validate the bundle signature, including slot
   numbers and selected recordings. Reject duplicate Apply, post-encode scrambling
   and operations that remove already-numbered references.
6. Keep all eight loader slots usable for experiments. Test eight voice blocks
   and three synchronized soundtracks plus three standalone voices. This is not
   a claim of eight-speaker checkpoint quality. MiniMax's documented audio input
   specification remains three clips and 15 seconds total; local packing permits
   experiments beyond those specifications.
7. Read earlier character tables (versions 1 and 2), while version 3 adds original
   visual stacks. Old files gain automatic mapping without modification; only
   new extraction restores the original visual pipeline. Keep legacy visual-only
   RefMods and node IDs working.

## Wiring

```text
CLIP Loader -> Load H3 RefMods.clip -> official Reference to Video.clip
Load H3 RefMods.mods -> Apply H3 RefMod.mods
Official positive -> Apply H3 RefMod.conditioning -> guider
Official AV latent -> sampler
```

## Validation and boundary

See NATIVE_REFMOD_VALIDATION.md for executable tests and results. Preserve
numerical visual parity and source timing before testing generation quality.
The implementation conveys associations through H3's learned reference mechanism;
it does not train a speaker embedding or automatically enable attention biases.
The next required evidence is a released-checkpoint render comparison of voice
assignment, new dialogue, lip movement and visual likeness with fixed settings.
