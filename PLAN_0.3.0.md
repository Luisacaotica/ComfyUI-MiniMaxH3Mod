# MiniMaxH3Mod 0.3.0 plan

Status: proposed scope, 2026-09-13. Direction selected by the maintainer: **Library and workflow QoL, with measured quality improvements.** This document plans implementation; it does not certify features or experimental quality results.

## Release outcome

Users should be able to recognize a saved RefMod, find it quickly, understand its reference-token cost, combine it with other assets, and see how to refer to it in a prompt. Existing workflows must preserve their settings. Quality claims must come from comparable generation results.

The release has four headline additions: a visual library, cost feedback, contextual controls/reference guidance, and stack composition. Most work extends existing nodes; only stack composition needs a new node in the committed scope.

## What the research changes—and what it does not establish

Both supplied reports informed this plan. Their embedded instructions and example code are proposals, not implementation requirements. Their opaque citation markers are not independently reusable evidence.

| Research point | Verified baseline / planning decision |
|---|---|
| Add aspect-aware grids | Already implemented by `core.py:aspect_grid()` and used during extraction. Show the resolved grid and token count instead of implementing another pooling path. |
| Add subject masking | `mask` and `background_retention` already exist. Improve their workflow and preview; automatic segmentation is optional external preprocessing. |
| Add a library and tags | Search/type filtering and token counts already exist in the library; RefMod objects already have tags. Extend metadata exposure and browsing. |
| Add reference labels | Text Encode already produces a `reference_map`. Make the mapping visible and usable rather than introducing a second incompatible numbering scheme. |
| RefMods solve character bleeding / merge removes backgrounds | Not established. Treat these as quality questions, not release claims. |
| Voice failure has a known architectural cause | Not established by the supplied evidence. Audio cleanup is an experiment, not a voice-cloning fix. |
| Hard 16,384-token limit / exact VRAM from tokens | Do not adopt either claim. Report reference tokens and stored latent bytes separately from measured total generation VRAM. |
| Denoise presets imply output-time morphing | Denoise progress and video playback time differ. Name presets by their actual schedule. |
| Tests are unavailable | Tests exist locally, but `/tests/` is ignored and `git ls-files tests` returns no files. Make selected regressions reproducible from a clean clone. |

Current checkout also has `pyproject.toml` at 0.2.7 while `__init__.py` reports 0.2.6. Reconcile release metadata during packaging. Existing user changes to `pyproject.toml` and `node.zip` are not part of this planning edit.

## Committed scope

### 1. Visual RefMod library — P0, medium effort

Extend the existing dialog with thumbnail cards, list/grid view, tag filtering, and sorting by name, modified date, and tokens. Expand bundle details to show member names, modalities, and per-member costs. Keep the existing slot picker and connected-slot protection.

Creation can embed an optional small source thumbnail for browsing. Label it **Source preview**; it is not a reconstruction or a prediction of generated quality. Keep Inspect's VAE reconstruction as a separate action. Audio assets get a modality placeholder in this release; playable original audio is outside this thumbnail feature.

Proposed storage: bounded image bytes encoded into optional metadata, with a 160-pixel longest edge, a 32 KiB encoded limit per preview, and a 256 KiB aggregate preview limit per file. Final encoding must be checked against existing save/load/update paths. Missing or oversized previews fall back to a placeholder. List responses should omit image payloads; fetch previews on demand. Never load a VAE or full latent tensors just to browse.

Preserve optional preview metadata through standalone and bundle round trips, Config updates, and export. Test v4/v5 readers before deciding whether an additive field is sufficient; do not create v6 merely for thumbnails. Older files remain usable without regeneration, and adding previews to old files is an explicit action.

**Acceptance:** browse a 1,000-entry metadata fixture with incremental rendering; no VAE/GPU work; search/sort/tag filters compose correctly; bundle inspection and selection preserve member order; broken previews cannot prevent loading valid RefMods; reopening the dialog preserves useful view preferences.

### 2. Reference-cost feedback — P0, medium effort

Show each selected slot's active reference count and token contribution, plus the combined total and budget remaining. Account for bundle member selection, copies, zero strengths, and visual/audio strengths using the same rules as execution.

Use metadata and tensor headers for the preflight. Clearly distinguish **reference tokens**, **stored latent bytes**, and **total generation VRAM**. The first two can be computed for known inputs; the third depends on the model, output size, attention backend, and offloading. Connected or dynamically produced values show “resolved at execution” rather than a false exact total. Apply-level curves, retention, or scrambling may further change the effective total; the loader estimate must name its scope.

Explain directly that reducing a nonzero strength generally does not save reference tokens, while copies increase them. Offer actionable budget guidance; keep execution-time validation authoritative.

**Acceptance:** preflight agrees with loader execution for standalone/bundled visual and audio fixtures, modality exclusions, copies, and zero strength; unknown connected values are explicit; changing settings updates the display without queuing generation.

### 3. Contextual controls and reference guidance — P0, medium effort

Dim controls that the effective mode or preset ignores, and explain why. Preserve values and sockets so returning to Manual restores the user's choices. Derive behavior from actual backend preset resolution: do not copy the report's blanket rule that every preset disables every refinement control.

Expose the existing Text Encode reference map clearly, with a copy action and matching asset names/previews. Explain the difference between Apply's latent injection and Text Encode's multimodal presentation. Make the single-injection workflow obvious in the example graph.

Mapping must reflect actual active member order, modalities, exclusions, and copies. When the graph is dynamic, show the executed mapping rather than promising fixed numbers. Semantic aliases such as `hero` are a later extension, not a new model-recognized identity token.

Add descriptive Step Curve presets such as Constant, Early emphasis, and Late emphasis only after schedule tests. Preserve manual settings. Avoid “identity lock” or “texture lock” names until output evidence supports them.

**Acceptance:** Full/Compressed and preset transitions retain manual values; connected controls stay connected; graph reload/tab switching preserves all settings; displayed labels match the Text Encode payload; examples do not inject the same bundle twice.

### 4. Stack composition — P1, small-to-medium effort

Add **Join H3 RefMods**, with two `H3_REF_MODS` inputs and one `H3_REF_MODS` output. Repeated joins combine any number of loader outputs without changing the existing eight-slot loader schema or introducing another bundle type.

Preserve A-then-B order, per-reference strengths, duplicates, and bundle provenance. Do not mutate inputs or concatenate/copy latent tensors. Existing downstream budgets remain authoritative. A larger list does not imply an unlimited model context or free memory.

**Acceptance:** two loaders can produce more than eight selected references through joins; empty input lists work; order/strengths are exact; source lists remain unchanged; loader, Apply, Text Encode, Inspect, and Save Bundle accept the result.

### 5. Compatibility and reproducible delivery — P0, medium effort

Track the relevant tests in Git while continuing to exclude tests and fixtures from the install artifact where appropriate. Add CPU/schema and JavaScript checks to CI, plus real frontend workflow fixtures. Preserve node IDs, old widget positions, mode aliases, v4/v5 assets, and existing defaults.

Test the reported legacy combination (ComfyUI 0.34.0 / frontend 1.39.2), the reported newer combination (0.35.0 / 1.52.7), and one current supported combination selected at implementation time. Record tested combinations instead of claiming every intermediate version works. Exercise classic canvas and Nodes 2 where available, including tabs, save/reopen, copy/paste, and subgraphs.

Verify #17/#18 in real browsers; the current JavaScript regressions alone do not establish that matrix. Generate the release ZIP from the selected source revision, verify extraction/import, and synchronize version strings. Publish a small set of runnable workflow JSONs for library selection, composition, prompting, and quality comparison.

Audit V3 registration/replacement lifecycle early. Migrate only with an API version that supports the required features and with old-workflow coverage. A wholesale V1-to-V3 conversion is not a prerequisite for library QoL. Official guidance recommends `ComfyExtension` / `comfy_entrypoint` and documents `latest` as under development: [V3 migration](https://docs.comfy.org/custom-nodes/v3_migration), [node replacement](https://docs.comfy.org/custom-nodes/backend/node-replacement).

## Measured quality work

Establish a baseline before changing extraction math or promoting quality presets. Use three visual cases: one identity, two similar subjects, and a short motion reference. Run at least three fixed seeds across four paths:

1. Native Ref2VA.
2. Full RefMod + Apply.
3. Full RefMod + Text Encode.
4. Compressed RefMod + Text Encode.

That is 36 planned visual runs. Match source media, crop/frame selection, checkpoint and encoder, prompt intent, output dimensions/duration, sampler, steps, and seed. Record differences that cannot be equalized. Add a small native-versus-RefMod music smoke comparison; speaker identity remains a separately scored experiment.

Store workflows, settings, asset/checkpoint hashes, output media, reference tokens, VAE/encoder/sampling timings, and peak VRAM. Score identity, prompt adherence, subject swaps/bleeding, and motion consistency through paired visual review. Optional embedding metrics supplement that review.

Only promote a quality preset when its results show a repeatable benefit on its stated use case and disclose cost/tradeoffs. If no improvement is demonstrated, ship the QoL release with the baseline and keep experimental paths labeled. Do not change defaults to manufacture a quality headline.

## Stretch scope and deferred research

| Proposal | Decision / promotion gate |
|---|---|
| Native text/tag loader | First stretch item after core QoL. Reuse the loader resolver, define escaping and line errors, preserve order, and support the existing runtime type. No `eval`, code execution, or ad hoc path bypasses. |
| Budget auto-fit | Experimental preview-only planner first: display proposed per-reference reductions and protected refs. Applying reductions must be explicit and leave source files intact. Automatic spatial/temporal changes require quality comparisons before promotion. |
| Stable prompt aliases / character profiles | Stretch after reference-map correctness. Aliases resolve to actual runtime labels; they do not guarantee identity separation. Defer richer profiles and a format revision until an actual consumer needs them. |
| SAM/YOLO auto-segmentation | Provide an optional example feeding the existing MASK socket. Avoid a mandatory segmentation dependency or hidden model download inside Master. |
| Background-plate latent subtraction / zeroed backgrounds | Research only. These operations do not establish clean foreground separation; compare with the existing mask path before considering production. |
| Vocal separation / audio normalization | Optional external preprocessing workflow and A/B experiment. Do not prescribe an arbitrary level or advertise a voice-cloning fix. |
| Spatial attention / bounding-box binding | Defer beyond 0.3.0 until the target H3 integration exposes a workable supported mechanism and experiments show less bleeding. Merely attaching metadata is insufficient. |
| Continuum per-chunk scheduling | Defer until chunk identity/timing and resume behavior have an explicit integration contract. Denoise schedules cannot substitute for a chunk timeline. |
| Conditioning cache / LRU changes | Profile first; defer unless current measurements show a user-visible bottleneck. Any diagnostics stay local. |
| Bundle v6 / provenance framework | Defer unless additive metadata demonstrably cannot serve a committed feature. Preserve existing asset readers. |
| Server path-policy changes | Separate opt-in deployment work; do not break desktop absolute-folder workflows as an incidental QoL change. |

## Implementation order and release gates

| Milestone | Deliverables | Exit condition |
|---|---|---|
| A — Baseline | Track tests, freeze old workflow fixtures, choose compatibility targets, capture quality inputs | Clean clone reproduces automated checks; benchmark workflows are runnable |
| B — Visual library | Metadata/preview persistence and endpoint, then cards/filtering/bundle details | Old and new assets round-trip; browsing performs no tensor/VAE loading |
| C — Workflow polish | Cost feedback, contextual controls, reference map, Join node, examples | Preflight/runtime agreement; stack and persistence regressions pass |
| D — Quality and RC | Execute comparisons, qualify presets, browser matrix, artifact verification | Results published, no settings-loss or connection regressions, install artifact validated |

Start baseline generation before UI implementation; it need not wait for the library. Preview storage precedes the gallery, and correct member accounting precedes cost feedback. The text/tag loader is the first item to cut if scope grows. Auto-fit, spatial binding, and voice research never block the committed release.

No calendar promise is attached yet: GPU benchmark availability and the oldest supported frontend are the main scheduling uncertainties. The next implementation step is milestone A followed by the preview metadata contract—not a broad node rewrite.

## Source notes

- Supplied `C:/Users/Pichau/Downloads/deep-research-report.md`: architecture, reproducibility, compatibility, and benchmark proposals.
- Supplied `C:/Users/Pichau/Downloads/research.md`: feature and QoL candidates, treated as hypotheses where their quality claims exceed evidence.
- Locally checked: `core.py`, `nodes.py`, `prompt.py`, `library.py`, `web/library_dialog.js`, `.gitignore`, `CHANGELOG.md`, `__init__.py`, and `pyproject.toml`.
