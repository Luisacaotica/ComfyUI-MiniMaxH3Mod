"""Scoped CLIP adapter for the unmodified MiniMax H3 Reference to Video node.

The official node still creates conditioning and the empty AV latent. This
adapter supplies cached presentation to that encode; Apply supplies the latents.
No shared tokenizer, encoder weights, node classes or ComfyUI files are patched.
"""
import hashlib
import json

from .refmod_profile import ProfileRefMod
from .character_references import tensor_digest

MANIFEST_KEY = "h3_refmod_native"
APPLIED_KEY = "h3_refmod_native_applied"


def profiles_in(mods):
    return [(m, s) for m, s in (mods or []) if isinstance(m, ProfileRefMod) and s > 0]


def bundle_signature(mods):
    rows = []
    for mod, strength in profiles_in(mods):
        refs = []
        for ref in mod.selected_references():
            refs.append({"kind": ref.kind, "duration": ref.duration,
                         **{k: tensor_digest(getattr(ref, k)) for k in
                            ("visual", "audio", "frames", "timestamps")}})
        rows.append({"id": mod.profile.character_id, "name": mod.name,
                     "strength": strength, "voice": mod.voice_reference, "refs": refs})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_map(mods):
    counts = {"image": 0, "video": 0, "audio": 0}
    tag = {"image": "Picture", "video": "Video", "audio": "Audio"}
    profiles = profiles_in(mods)
    labels = [[] for _ in profiles]
    for owner, mod, ref in ordered_profile_refs(mods):
        for item in presentation_for(ref):
            kind = item["type"]
            counts[kind] += 1
            labels[owner].append(f"<{tag[kind]} {counts[kind]}>")
    return "\n".join(f"{mod.name}: {', '.join(labels[i])}. {mod.report()}"
                     for i, (mod, _) in enumerate(profiles))


def ordered_profile_refs(mods):
    refs = [(i, mod, ref) for i, (mod, _) in enumerate(profiles_in(mods)) for ref in mod.selected_references()]
    order = {"image": 0, "video": 1, "video_audio": 1, "audio": 2}
    return sorted(refs, key=lambda row: order[row[2].kind])


def presentation_for(ref):
    items = [{"type": "audio"}] if ref.audio is not None else []
    if ref.kind == "image":
        items.append({"type": "image", "data": ref.vision_pixels()})
    elif ref.kind in ("video", "video_audio"):
        items.append({"type": "video", "data": ref.vision_pixels(), "timestamps": ref.timestamps.tolist()})
    return items


class RefModTokens(dict):
    """Per-call metadata travels with tokens, never as mutable adapter state."""
    def __init__(self, tokens, manifest):
        super().__init__(tokens)
        self.manifest = manifest


class RefModCLIP:
    def __init__(self, clip, mods, signature=None):
        self.base = clip
        self.mods = tuple(profiles_in(mods))
        self.signature = signature or bundle_signature(self.mods)

    def __getattr__(self, key):
        return getattr(self.base, key)

    def clone(self, *args, **kwargs):
        return RefModCLIP(self.base.clone(*args, **kwargs), self.mods, self.signature)

    def tokenize(self, text, return_word_ids=False, **kwargs):
        if "minimax_ref_items" not in kwargs:
            raise ValueError("Connect Load H3 RefMods.clip to the official MiniMax H3 Reference to Video node.")
        # Cache references first so their numbering stays stable when ordinary
        # references are also connected to the official node. Apply uses this
        # same order for reference latents; native references follow them.
        items = [item for _, _, ref in ordered_profile_refs(self.mods) for item in presentation_for(ref)]
        native_items = list(kwargs.pop("minimax_ref_items") or [])
        tokens = self.base.tokenize(text, return_word_ids=return_word_ids,
                                   minimax_ref_items=items + native_items, **kwargs)
        if not isinstance(tokens, dict):
            raise ValueError("Expected the MiniMax H3 text encoder's token dictionary.")
        return RefModTokens(tokens, {"signature": self.signature,
                                    "reference_map": reference_map(self.mods)})

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        if not isinstance(tokens, RefModTokens) or tokens.manifest["signature"] != self.signature:
            raise ValueError("RefMod tokens belong to a different loader/CLIP branch.")
        positive = self.base.encode_from_tokens_scheduled(dict(tokens), *args, **kwargs)
        if not positive or any("minimax_token_tags" not in row[1] for row in positive):
            raise ValueError("Use the MiniMax H3 text encoder (CLIP loader type minimax).")
        return [[row[0], {**row[1], MANIFEST_KEY: dict(tokens.manifest)}] for row in positive]


def prepare_clip(clip, mods):
    if not profiles_in(mods):
        return clip
    if clip is None:
        return None  # Apply explains the missing CLIP connection if queued.
    if isinstance(clip, RefModCLIP):
        raise ValueError("Use the original CLIP Loader output as Load H3 RefMods.clip; do not chain prepared CLIPs.")
    return RefModCLIP(clip.clone(), mods)


def validate_apply(conditioning, mods, scramble_seed):
    profiles = profiles_in(mods)
    native = isinstance(conditioning, list)
    has_manifest = native and any(MANIFEST_KEY in row[1] for row in conditioning)
    if not profiles and not has_manifest:
        return False
    if not profiles or not native or not conditioning:
        raise ValueError("Use the same combined RefMods in Load H3 RefMods.clip and Apply H3 RefMod.mods.")
    if scramble_seed is not None and scramble_seed >= 0:
        raise ValueError("Combined character references require scramble_seed=-1 (Fixed), so voice/image labels remain aligned.")
    signature = bundle_signature(mods)
    for _, metadata in conditioning:
        if metadata.get(APPLIED_KEY):
            raise ValueError("Combined RefMods have already been applied to this conditioning. Use one Apply node.")
        manifest = metadata.get(MANIFEST_KEY)
        if manifest is None:
            raise ValueError("Combined RefMods need native reference presentation. Connect CLIP Loader -> "
                             "Load H3 RefMods.clip -> official MiniMax H3 Reference to Video.clip, "
                             "then its positive output -> Apply H3 RefMod.conditioning.")
        if manifest.get("signature") != signature:
            raise ValueError("The RefMods/voice selections differ from the CLIP branch. Connect both mods and clip from the same loader.")
    return True
