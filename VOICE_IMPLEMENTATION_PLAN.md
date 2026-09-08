# Character voice reliability: implementation and validation plan

Date: 2026-09-08
Upstream baseline: `7604ef4690168365b9db35c13a69e4c99421ace4`.

## Objective

Improve recurring characters' voice consistency while H3 generates the video,
new dialogue, emotion, and lip movement together. A reusable character must
retain enough information to reconstruct native visual/audio references and
their ownership. The deliverable is an implementation and reproducible test
system; successful file creation is not the success criterion for voice quality.

The user has already seen voice swaps in native H3. Native conditioning is a
diagnostic baseline, not the proposed final answer to that problem.

## Findings that determine the work

- Upstream audio extraction stores an H3 audio VAE encode. It does not train
  a speaker embedding. `voice` is metadata, not a speaker-specific loss.
- Upstream Apply appends reference tensors to already encoded conditioning.
  It does not construct H3's native numbered reference presentation.
- Master produces independent visual/audio blocks. A synchronized speaking
  video can instead use native `video_audio` packing with a shared time origin.
- Our existing character implementation already builds that presentation,
  preserves paired blocks, and includes optional attention bias. It has CPU
  and tiny-transformer tests, but no measured improvement with released weights.
- Our current saved vision frames are uint8. Quantizing resized floating-point
  pixels prevents exact presentation parity. New extractions should preserve
  those pixels, while old files remain readable.

Implementation finding: native ComfyUI H3 itself uses PIL Lanczos with a byte
conversion during image preparation. v2 extraction now matches that operation,
including when dimensions are unchanged, and stores its resulting float pixels
without an additional conversion. Standalone waveform resampling now also uses
the native torchaudio operation. Output resolution/crop and source interval must
still be matched explicitly when comparing independently prepared workflows.

## Phase 1 — Integrate the upstream update safely

1. Preserve our existing working changes and record their base revision.
2. Integrate the specified upstream revision, including audio Master, library,
   atomic storage, token limits, loader fixes, and upstream tests.
3. Reapply the character extension without replacing upstream implementations.
4. Keep original node IDs and existing character files/workflows compatible.
5. Give the combined extension its own version and document the pinned baseline.

Acceptance: upstream and character nodes register together; both test suites
can run; existing workflows still have valid schemas and connections.

## Phase 2 — Make reusable conditioning complete and inspectable

1. Extract one shared reference assembler from the dialogue conditioner. It
   produces presentation items, ordered model blocks, character ownership,
   and label mappings in the same traversal.
2. Preserve preprocessed floating-point RGB pixels for new profiles. Continue
   reading v1 uint8 profiles through an explicit compatibility conversion.
   Preserve audio/video latents exactly across save/load.
3. Retain synchronized video/audio as a native `video_audio` block with matching
   duration and timestamps. Reject invalid timing instead of inventing a pair.
4. Add **H3 Character Reference Conditioning**: accept an existing full prompt
   and one to three saved characters, then create native conditioning and the
   output AV latent. It supplies the same ownership data to Voice Binding as
   the dialogue conditioner. It does not rewrite the user's prompt.
5. Report actual reference labels, character IDs, block kinds, shapes, durations,
   and content fingerprints. These make comparisons reproducible and expose
   missing/reordered references before an expensive render.
6. Keep the current budget checks, deterministic order, clone isolation, and
   explicit errors for unsupported H3 configurations.

Acceptance: cached versus uncached prepared references produce identical
presentation tokens/vision inputs and reference tensors; native packed layout
agrees; two/three-character ownership survives mixed-media ordering. Tests use
real ComfyUI code, with model doubles only where released weights are unavailable.

## Phase 3 — Reuse upstream RefMods where the information is sufficient

1. Add **Import H3 RefMods as Character**, accepting a Master/Loader bundle for
   one character and the matching source image(s).
2. Reuse image/audio latents without encoding them again. Supply source images
   to restore the vision-encoder presentation that upstream files do not store.
3. Require unchanged reference strength and no duplicate copies. Reject missing
   images, mismatched counts, invalid audio duration, and unsupported visual
   stacks with actionable errors.
4. Save the result as a normal character profile, with import provenance.
5. Do not infer video/audio synchronization from matching filenames. A speaking
   video should use our synchronized clip extractor, since older files may have
   pooled/sampled time or discarded the source timing.

Acceptance: imported audio/image tensors remain exact; labels/owners are
correct; old/new upstream metadata can be read by upstream's own loader;
unsupported cases fail before writing a character file.

## Phase 4 — Controlled all-H3 experiments

The full-prompt conditioner will expose explicit diagnostic options, defaulting
to the complete native path:

- `native` presentation versus `latent_only`: keep the prompt and stored tensors
  constant; measure the effect of the numbered/visual reference presentation.
- `paired` versus `separate` audiovisual layout: use the same saved speaking
  clip and prompt; change only its native reference packing. Do not fabricate
  synchronized evidence for images with unrelated audio.
- Existing Voice Binding modes: `off`, `pair_only`, and `scheduled`. Keep the
  prompt, seed, references, output geometry, dialogue, and speaking schedule
  identical when comparing attention modes.

Supply importable ComfyUI workflows and a trial manifest/result sheet for a
small staged experiment: one voice first, then two, then three. Provide a local
results summarizer that checks matched conditions and reports outcomes without
inventing scores. No remote jobs or paid services will be started automatically.

Quality fields: correct assigned voices / speaking characters, exact requested
words, copied source words, intelligibility, correct speaking face/lip movement,
requested emotion, and render time. Use the same seed set for every variant;
retain failures and test promising settings on fresh dialogue/seeds.

Acceptance: workflow schemas validate; switching diagnostic modes preserves all
unrelated inputs; binding `off` is exact passthrough; identical references have
stable fingerprints; a report cannot silently compare unmatched/incomplete runs.

## Phase 5 — Verification and handoff

1. Run upstream regressions plus character storage/media/conditioning/binding
   tests with the local CPU environment.
2. Run real ComfyUI schema/tokenizer/layout integration tests and tiny native H3
   forward tests. Verify input tensors, weights, and other MODEL branches remain
   unchanged. Confirm conflict guards still reject incompatible attention hooks.
3. Document installation, existing-file migration, which workflow to open first,
   and the exact tested ComfyUI revision. Record code-test results separately
   from real checkpoint quality results.
4. Leave a reviewable local change set ready to commit/push. Do not push or start
   generation on the user's other machine without an available authorized path.

## Decision after real H3 renders

- If cached-native differs from native, fix preprocessing/presentation/payload
  parity before interpreting voice quality.
- If paired packing improves assignment, retain it as the default for speaking
  profiles and measure whether attention binding adds further benefit.
- If binding helps but damages dialogue or emotion, evaluate lower bias rather
  than maximizing it. Scheduled bias is only meaningful when speech follows the
  requested windows; record timing failures.
- If these interventions do not improve a native-equivalent baseline, the next
  research stage is a trained H3 adapter or reference-latent optimization using
  generated-speech speaker similarity, new-dialogue accuracy, and audiovisual
  consistency objectives. That requires released weights, differentiable training
  support, compute, and evaluation data. Reconstruction MSE or a metadata flag
  cannot substitute for those objectives. This phase is not claimed implemented.

The implementation can remove known integration gaps and expose meaningful
controls. Real repeated renders determine whether it solves the user's practical
voice-assignment problem; CPU checks cannot establish that outcome.

## Implementation status

Phases 1–4 are implemented locally, including upstream integration, the shared
reference assembler, v2/v1 storage support, full-prompt conditioning, supported
upstream imports, ten total workflows, and a blank matched-trial results system.
The existing optional attention experiment is preserved and integrated with the
new conditioner; no new speaker-training algorithm is claimed.

Phase 5 local verification is recorded in [CHARACTER_VALIDATION.md](CHARACTER_VALIDATION.md).
The remaining external step is running the real H3 comparisons on the user's
ComfyUI machine. The branch has not been pushed, and no quality results have
been filled into the trial sheet.
