"""Experimental attention bias for character association, using H3's own AV model.

No weights, latents or global attention functions are modified. Positive biases
favor an attention edge; negative biases discourage it. These are soft controls,
not speaker embeddings or guarantees about the rendered scene.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BINDING_KEY = "h3_character_binding"
OWNER_KEY = "h3_character_owner"
ID_KEY = "h3_character_id"
RUN_KEY = "h3_character_condition_id"
WRAPPER_KEY = "h3_character_voice_binding"


@dataclass(frozen=True)
class SpeakerTurn:
    owner: int  # 1-based character input slot, NOT the prompt's S-number
    start: float
    end: float


def parse_turns(text, cast, duration, require_dialogue_coverage=True):
    """One CSV line per speaking character: character,start_seconds,end_seconds.

    A blank schedule leaves timing to H3. Explicit schedules must cover exactly
    the speaking characters and may leave pauses. Overlap/repeated turns are
    rejected in this first experiment because the output is one stereo stream.
    """
    if not str(text).strip():
        return ()
    turns = []
    for line_number, line in enumerate(str(text).splitlines(), 1):
        if not line.strip():
            continue
        try:
            owner, start, end = (value.strip() for value in line.split(","))
            turn = SpeakerTurn(int(owner), float(start), float(end))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Turn schedule line {line_number}: use character,start_seconds,end_seconds.") from exc
        if (not 1 <= turn.owner <= len(cast) or not math.isfinite(turn.start)
                or not math.isfinite(turn.end) or not 0 <= turn.start < turn.end <= duration + 1e-6):
            raise ValueError(f"Turn schedule line {line_number}: invalid character or interval (output {duration:.3f}s).")
        if turn.end - turn.start < 0.25:
            raise ValueError("Each speaking window must be at least 0.25 seconds.")
        turns.append(turn)
    turns.sort(key=lambda t: t.start)
    owners = [t.owner for t in turns]
    speaking = {i for i, row in enumerate(cast, 1) if str(row[1]).strip()}
    if len(owners) != len(set(owners)) or (require_dialogue_coverage and set(owners) != speaking):
        raise ValueError("Provide exactly one window for each character with dialogue, and none for silent characters.")
    if any(a.end > b.start for a, b in zip(turns, turns[1:])):
        raise ValueError("Speaking windows must not overlap in this experiment.")
    return tuple(turns)


def ref_signature(ref):
    def shape(key):
        value = ref.get(key)
        return None if value is None else tuple(value.shape)
    return (ref.get("kind"), ref.get(OWNER_KEY), ref.get(ID_KEY), shape("latent"), shape("audio_latent"),
            ref.get("latent_t"), ref.get("latent_h"), ref.get("latent_w"), ref.get("ref_audio_t"))


@dataclass(frozen=True)
class BindingSpec:
    condition_id: str
    references: tuple
    turns: tuple[SpeakerTurn, ...]
    frame_count: int

    @property
    def audio_t(self):
        return round(self.frame_count / 24 * 40)

    def validate_refs(self, refs):
        if tuple(ref_signature(r) for r in refs) != self.references or any(
                r.get(RUN_KEY) != self.condition_id for r in refs):
            raise ValueError("Voice binding received different references. Connect the SAME character positive "
                             "to Voice Binding and the guider; do not append/reorder references afterward.")


def make_binding(blocks, turns, frame_count):
    condition_id = str(uuid.uuid4())
    for ref in blocks:
        ref[RUN_KEY] = condition_id
    return BindingSpec(condition_id, tuple(ref_signature(r) for r in blocks), tuple(turns), frame_count)


@dataclass(frozen=True)
class BiasGroup:
    start: int
    stop: int
    keys: tuple  # (start, stop, additive logit bias)
    weights: tuple = ()  # optional per-query fade at a speaking-window boundary


def build_bias_groups(layout, refs, binding, pair_bias, voice_bias, fade_seconds):
    """Map real packed rows to character owners, including channel-major stereo.

    Only reference-audio and scheduled target-audio queries are biased.
    Text and target-video queries keep their original attention calculation.
    Both streams still interact through H3's joint transformer.
    """
    binding.validate_refs(refs)
    if layout.signature[-1] != binding.audio_t or layout.signature[1] != 2 + (binding.frame_count - 5) // 17 * 5:
        raise ValueError("Voice binding output duration differs from character conditioning.")
    segments = [s for s in layout.segments if s[2] in ("ref_img", "ref_audio")]
    cursor, visual, audio = 0, [], []
    for ref in refs:
        owner = ref[OWNER_KEY]
        expected = []
        if ref.get("audio_latent") is not None:
            expected.append(("ref_audio", ref["audio_latent"].shape[-1] * 2))
        if ref.get("latent") is not None:
            z = ref["latent"]
            expected.append(("ref_img", z.shape[2] * (z.shape[3] // 2) * (z.shape[4] // 2)))
        for kind, count in expected:
            if cursor >= len(segments):
                raise ValueError("H3 packed reference layout is incomplete.")
            start, stop, actual = segments[cursor]
            cursor += 1
            if actual != kind or stop - start != count:
                raise ValueError("H3 packed reference order/geometry changed; voice binding cannot be applied safely.")
            (audio if kind == "ref_audio" else visual).append((start, stop, owner))
    if cursor != len(segments):
        raise ValueError("Unexpected additional reference rows in H3's layout.")
    groups = []
    if pair_bias:
        for start, stop, owner in audio:
            keys = tuple((a, b, pair_bias if other == owner else -pair_bias) for a, b, other in visual)
            groups.append(BiasGroup(start, stop, keys))
    if voice_bias and binding.turns:
        target = [s for s in layout.segments if s[2] == "audio"]
        if len(target) != 1 or target[0][1] - target[0][0] != 2 * binding.audio_t:
            raise ValueError("Expected one channel-major stereo H3 target audio stream.")
        origin = target[0][0]
        for turn in binding.turns:
            keys = tuple((a, b, voice_bias if other == turn.owner else -voice_bias) for a, b, other in audio)
            # Classify each 40 Hz audio token by its center, identically in both channels.
            left = max(0, math.ceil(turn.start * 40 - 0.5))
            right = min(binding.audio_t, math.ceil(turn.end * 40 - 0.5))
            weights = tuple(min(1.0, max(0.0, min((i + 0.5) / 40 - turn.start,
                           turn.end - (i + 0.5) / 40) / fade_seconds))
                            for i in range(left, right)) if fade_seconds else ()
            for channel in (0, 1):
                offset = origin + channel * binding.audio_t
                groups.append(BiasGroup(offset + left, offset + right, keys, weights))
    groups.sort(key=lambda g: g.start)
    if any(g.start >= g.stop for g in groups) or any(a.stop > b.start for a, b in zip(groups, groups[1:])):
        raise ValueError("Overlapping or empty attention bias regions.")
    return tuple(groups)


def biased_attention(original, q, k, v, heads, groups, query_chunk=64, **kwargs):
    """Exact additive-bias SDPA on affected queries, original backend elsewhere.

    Never constructs a full sequence-by-sequence bias matrix. Chunking also
    bounds the approximate score workspace when PyTorch uses its math backend.
    The model's Q/K normalization, RoPE, V, output projection and residuals are
    kept. Bias 0 bypasses this function entirely at the model-node level.
    """
    if not groups:
        return original(q, k, v, heads, **kwargs)
    if (not kwargs.get("skip_reshape") or kwargs.get("skip_output_reshape", False)
            or kwargs.get("mask") is not None or q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
            or q.shape[1] != heads):
        raise ValueError("Unsupported H3 attention shape/mask; disable conflicting attention patches.")
    batch, _, tokens, dim = q.shape
    # Approximate 64 MiB for three fp32 score/probability workspaces per chunk.
    chunk = min(query_chunk, max(1, 64 * 1024**2 // (batch * heads * tokens * 4 * 3)))
    out = torch.empty((batch, tokens, heads * dim), device=q.device, dtype=q.dtype)
    cursor = 0
    for group in groups:
        if group.start < cursor or group.stop > tokens:
            raise ValueError("Attention bias range falls outside the actual sequence.")
        if cursor < group.start:
            out[:, cursor:group.start] = original(q[:, :, cursor:group.start], k, v, heads, **kwargs)
        bias = torch.zeros(tokens, device=q.device, dtype=q.dtype)
        for a, b, value in group.keys:
            if not 0 <= a < b <= tokens:
                raise ValueError("Reference key range falls outside the actual sequence.")
            bias[a:b] += value
        weights = torch.tensor(group.weights, device=q.device, dtype=q.dtype) if group.weights else None
        for start in range(group.start, group.stop, chunk):
            end = min(group.stop, start + chunk)
            mask = bias.view(1, 1, 1, -1)
            if weights is not None:
                mask = mask * weights[start - group.start:end - group.start].view(1, 1, -1, 1)
            extra = {"scale": kwargs["scale"]} if kwargs.get("scale") is not None else {}
            result = F.scaled_dot_product_attention(q[:, :, start:end], k, v, attn_mask=mask,
                                                    dropout_p=0.0, is_causal=False, **extra)
            out[:, start:end] = result.transpose(1, 2).reshape(batch, end - start, heads * dim)
        cursor = group.stop
    if cursor < tokens:
        out[:, cursor:] = original(q[:, :, cursor:], k, v, heads, **kwargs)
    return out


def make_binding_wrapper(binding, pair_bias, voice_bias, fade_seconds, query_chunk):
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        refs = payload.get("refs", [])
        binding.validate_refs(refs)
        if transformer_options.get("optimized_attention_override") is not None:
            raise ValueError("Voice Binding cannot be combined with another optimized-attention override.")
        if transformer_options.get("patches_replace", {}).get("dit"):
            raise ValueError("Disable H3 block/VSA replacement patches when testing Voice Binding.")
        options = transformer_options.copy()
        calls, last_layout, groups = 0, None, ()

        def attention(original, q, k, v, heads, **attn_kwargs):
            nonlocal calls, last_layout, groups
            layout = attn_kwargs.get("transformer_options", {}).get("minimax_h3_layout")
            # Text-only refinement is intentionally outside the experiment.
            if layout is None or q.ndim != 4 or q.shape[-2] != layout.seq_len:
                return original(q, k, v, heads, **attn_kwargs)
            if layout is not last_layout:
                groups = build_bias_groups(layout, refs, binding, pair_bias, voice_bias, fade_seconds)
                last_layout = layout
            calls += 1
            return biased_attention(original, q, k, v, heads, groups, query_chunk, **attn_kwargs)

        options["optimized_attention_override"] = attention
        result = executor(x, timestep, context, options, **kwargs)
        if not calls:
            raise ValueError("This H3 implementation bypassed the attention hook. Update ComfyUI and disable "
                             "compiled/replacement attention for this experiment.")
        return result
    return wrapper


class H3CharacterVoiceBinding:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",),
            "mode": (["off", "pair_only", "scheduled"], {"default": "pair_only"}),
            "pair_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1,
                "tooltip": "Experimental: reference audio favors its own character's visual references and discourages others."}),
            "voice_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1,
                "tooltip": "Scheduled mode: target audio favors its assigned voice reference during each speaking window."}),
            "boundary_fade_seconds": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.025}),
            "query_chunk": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1, "advanced": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/character/experimental"

    def apply(self, model, positive, mode="pair_only", pair_bias=0.5, voice_bias=0.5,
              boundary_fade_seconds=0.1, query_chunk=64):
        if mode not in ("off", "pair_only", "scheduled"):
            raise ValueError("Unknown voice binding mode.")
        for value, maximum in ((pair_bias, 3), (voice_bias, 3), (boundary_fade_seconds, 1)):
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError("Invalid voice binding strength/fade.")
        if not isinstance(query_chunk, int) or not 1 <= query_chunk <= 256:
            raise ValueError("query_chunk must be between 1 and 256.")
        effective_voice = voice_bias if mode == "scheduled" else 0.0
        from comfy.patcher_extension import WrappersMP
        if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY):
            raise ValueError("Use one Voice Binding node on an unbound model branch; do not chain these nodes.")
        if mode == "off" or (pair_bias == 0 and effective_voice == 0):
            return model, "Voice binding OFF: original model and attention path."
        if len(positive) != 1 or not isinstance(positive[0][1].get(BINDING_KEY), BindingSpec):
            raise ValueError("Connect positive directly from H3 Character Dialogue or Reference Conditioning.")
        binding = positive[0][1][BINDING_KEY]
        binding.validate_refs(positive[0][1].get("minimax_refs", []))
        if effective_voice and not binding.turns:
            raise ValueError("Scheduled mode needs turn_schedule in the H3 Character conditioner. "
                             "Enter one line per speaker: character,start_seconds,end_seconds.")
        from comfy.ldm.minimax.model import MiniMaxH3Model
        if not isinstance(model.get_model_object("diffusion_model"), MiniMaxH3Model):
            raise ValueError("Voice Binding requires the native ComfyUI MiniMax H3 model.")
        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
            make_binding_wrapper(binding, pair_bias, effective_voice, boundary_fade_seconds, query_chunk))
        info = (f"EXPERIMENTAL {mode}: pair bias {pair_bias:.2f}, voice bias {effective_voice:.2f}; "
                f"{len(binding.turns)} speaking windows. Uses H3 joint AV attention, with soft biases; "
                "render improvement is unverified. Same positive must feed the guider.")
        print(f"[H3 Character] {info}")
        return patched, info
