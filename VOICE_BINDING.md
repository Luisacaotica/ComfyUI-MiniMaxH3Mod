# Experimental character-to-voice association inside H3

For the integrated upstream update and full-prompt comparison workflows, see
[VOICE_TESTING.md](VOICE_TESTING.md). Both character conditioners supply the
same ownership information to this attention control.

This experiment changes **attention during H3 generation**. It keeps H3's joint
video, speech, expression, and lip-motion generation. No separate speech model
or replacement soundtrack is used. Existing character files work without
re-extraction; synchronized single-speaker video profiles are the preferred
starting point for this test.

The hypothesis is that reducing ambiguous reference interactions can improve
speaker assignment. **A reduction in real voice swaps has not been measured.**
This is an implementation to test that hypothesis, not a trained voice lock.

## What changes

Add **H3 Character Voice Binding (Experimental)** after your H3 model's sigma
shift node. Connect `positive` from **H3 Character Dialogue Conditioning** to
both Voice Binding and **Basic Guider**. Send Voice Binding's `model` output
to the guider and scheduler. The examples already have these connections.

| Mode | Action inside the joint H3 transformer |
| --- | --- |
| `off` | Returns the original model. No attention patch is installed. |
| `pair_only` | Reference-audio queries favor the visual references belonging to their own character and discourage other characters' visual references. |
| `scheduled` | Adds a speaking timeline: generated audio queries favor the assigned character's audio reference during that character's window and discourage other voices. Also applies `pair_bias` if nonzero. |

For a saved speaking video, the pairing comes from the video's own visual and
audio tracks. H3's existing shared reference timeline and positional encoding
remain intact. The additional bias strengthens the reference audio's access
to its own character's visuals. It does not realign phonemes, detect a face,
or create a new identity embedding.

For images plus separate audio, the same control uses the ownership recorded
by our character node. These inputs do not supply synchronized lip/voice
evidence, but the ownership mapping is still available.

`pair_bias` and `voice_bias` are **additive attention-logit biases**, not latent
volume, denoise strength, or voice similarity scores. At 0.5, an affected edge
to an assigned reference gets +0.5 and one to another character gets -0.5.
That changes their pairwise attention odds by a factor of approximately 2.72
before accounting for the original logits. This is a numerical description,
not a predicted improvement in render quality.

Reference-audio queries and timed target-audio queries are the only queries
directly biased. Text and target-video queries use the original attention
backend. Later transformer layers still exchange information between all
streams. Cross-character information can therefore still pass indirectly.

## Start with two characters

1. Update this pack in your ComfyUI installation and restart ComfyUI. Use a
   current native H3 installation; development used ComfyUI 0.34.0 / `fbed745`.
2. Drag [05_binding_two_characters.json](examples/characters/05_binding_two_characters.json)
   onto ComfyUI. Choose two character files and your H3 Ref2VA models/VAEs.
3. Prefer clear single-speaker reference clips, with comparable duration and
   reference resolution across characters. Pick `first` voice when a file
   contains two audio examples.
4. Enter short new dialogue. Keep the supplied timing and fixed seed initially.
5. Render `off`, then `pair_only`, then `scheduled`, **keeping the seed, prompt,
   schedule, reference files, models, resolution, and sampler settings identical**.
   Start with both biases at 0.5. Change Save Video's prefix for each mode.

The schedule stays in the prompt even when mode is `off`. This makes the
comparison isolate the attention change, rather than changing the prompt and
the model intervention together. The updated prompt builder supplies all six
reference-prompt sections, separates audio reference from copying, and assigns
speaker IDs by actual speaking order following the
[official H3 reference guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt).
Prompt formatting itself is shared by all three modes.

Then try [06_binding_three_characters.json](examples/characters/06_binding_three_characters.json),
which uses a roughly ten-second output and three speaking windows.

`pair_only` also works with **blank** `turn_schedule` if you want H3 to choose
when the characters speak. Scheduled mode with a nonzero `voice_bias` requires
an explicit schedule.

## Speaking windows

The dialogue node's optional `turn_schedule` uses one line per speaking
character:

```text
1,0.5,2.5
2,3.0,5.0
3,6.0,8.5
```

Each line means `character input number, start seconds, end seconds` within
the **generated** video. These numbers do not refer to the source recording
or to denoising steps. The conditioner uses the same schedule in the prompt
and in the attention mapping. Character input numbers stay stable if the
speaking order changes; prompt speaker IDs are assigned in speaking order.

The first implementation supports one window per character with nonblank
dialogue and non-overlapping turns. Pauses are allowed and receive no target
voice bias. A connected character with blank dialogue stays silent and must
not have a speaking window. Overlap, missing windows, duplicate turns, and
windows beyond the generated duration raise errors.

Both stereo channels use identical ownership timing. `boundary_fade_seconds`
(default 0.1) softens the bias near the edges of a speaking window. This reduces
an abrupt control change; it does not force speech to fit the interval. Leave
enough time for each line. If H3 speaks outside a requested window, the control
can favor the wrong voice at that time, which is a failure to record.

## Evaluate improvement

Use the same small seed set for every mode. The included
[trial sheet](examples/characters/binding_trial.csv) has 18 blank result rows:
two cast sizes, three seeds, and three modes. It contains no generated results.

Record the number of characters with the correct assigned voice, whether the
new dialogue is correct, whether source words are copied, and whether the
correct visible speaker's mouth matches the speech. Include failures and
render time. Keep reference copies and prompts for reproducibility.

Compare modes **within each seed and scene**. A useful result is more correctly
assigned voices without a corresponding increase in gibberish, reference-word
copying, lost emotions, or incorrect mouth movement. A lucky clip does not
establish improvement across the cast. If 0.5 helps, test 0.25 and 1.0 on a
small comparison set, then check the selected value on new lines/seeds. Higher
bias is not necessarily better.

Set `pair_bias = 0` in scheduled mode to isolate the timing control. Set both
biases to zero, or select `off`, for the original model path. Do not chain
Voice Binding nodes.

## Runtime boundaries

- Uses the **native ComfyUI H3** attention-override hook, on a cloned model
  branch. Weights and saved reference tensors are unchanged. Unpatched model
  branches keep their original behavior.
- Use **Basic Guider** for these CFG-distilled H3 comparison workflows. The
  patch expects exactly the character conditioning that configured it, not
  separate negative or combined conditioning branches.
- Different casts, rearranged/appended references, or a mismatched output
  duration fail explicitly. Do not append old RefMods after character
  conditioning while testing this control.
- Other optimized-attention overrides and DiT/VSA block replacements are
  rejected. Disable compilation/replacement accelerators for the initial
  experiment. If the expected hook is bypassed, the node raises an error
  instead of reporting success.
- Changed queries use PyTorch scaled-dot-product attention with a bias. The
  rest use the selected native backend. This can be slower than an optimized
  unmasked run. `query_chunk` controls chunk size; the code also limits an
  estimate of the math backend's temporary score workspace. It never allocates
  a full sequence-by-sequence bias matrix, but this does not cap total VRAM.
- There is no face tracking, spatial mask routing, overlapping speech support,
  learned character adapter, or automatic recognition of the actual speaker.
  Those are subsequent experiments if this simpler intervention is useful.

## What was actually verified

CPU tests check reference ownership after mixed-media sorting, speaking order,
stereo timing, gaps, fades, wrong-condition rejection, and numerical agreement
with dense additive-bias attention. Tests confirm the bias changes attention
in the intended direction and preserves its inputs.

The ComfyUI smoke test runs the actual native H3 transformer with a small,
randomly initialized configuration, using two and three characters and two
noise levels. It checks both modes, finite output changes, exact bypass,
unchanged weights, and conflicts/bypassed hooks. Workflow graphs are checked
against current node schemas. This verifies the mechanism is connected and
operates; it does not test the trained H3 checkpoint's voice or visual quality.

Next evidence needed: the paired `off` / `pair_only` / `scheduled` renders on
your recurring characters. Improvement in those renders, rather than file
creation or attention-unit-test success, is the objective of this experiment.
