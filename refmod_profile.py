"""Combined appearance/audio assets inside the existing H3_REF_MODS contract."""
from dataclasses import dataclass, replace
import json

import torch

from .core import H3RefMod, _blur_latent, curve_strengths
from .character_core import H3CharacterMod


@dataclass
class ProfileRefMod(H3RefMod):
    profile: H3CharacterMod | None = None
    voice_reference: int = 1  # 1-based; 0 explicitly selects all stored voices.
    subject_slot: int = 0  # Runtime loader slot; never persisted as character identity.

    def __post_init__(self):
        if self.profile is None:
            raise ValueError("Combined RefMod is missing its character references.")
        self.profile.validate()
        self.kind = "character"

    @classmethod
    def from_profile(cls, profile, path="", config=None):
        visual = next(r.visual for r in profile.references if r.visual is not None)
        meta = profile.provenance.get("visual_metadata", {})
        return cls(name=profile.name, kind="character", latent=visual,
                   latent_t=visual.shape[2], latent_h=visual.shape[3], latent_w=visual.shape[4],
                   mode=meta.get("mode", "encode"), source=meta.get("source", "character"),
                   pool=meta.get("pool", ""), optimize_steps=meta.get("optimize_steps", 0),
                   description=profile.description, concept_type=meta.get("concept_type", "identity"),
                   profile=profile, path=path, config=config or {})

    @classmethod
    def load_profile(cls, path, meta):
        profile = H3CharacterMod.load(path + ".safetensors")
        config = meta.get("refmod_config", {})
        if isinstance(config, str):
            config = json.loads(config)
        return cls.from_profile(profile, path, config if isinstance(config, dict) else {})

    def selected_references(self):
        if isinstance(self.voice_reference, bool) or not isinstance(self.voice_reference, int) or self.voice_reference < 0:
            raise ValueError("Voice reference must be 1-based, or 0 for all recordings.")
        voices = [r for r in self.profile.references if r.audio is not None]
        if self.voice_reference == 0:
            return self.profile.references
        if self.voice_reference > len(voices):
            raise ValueError(f"{self.name} contains {len(voices)} voice recording(s); select 1..{len(voices)} or 0 for all.")
        selected = voices[self.voice_reference - 1]
        return [r for r in self.profile.references if r.audio is None or r is selected]

    @property
    def token_count(self):
        return sum(r.token_count for r in self.selected_references())

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for r in self.profile.references
                   for t in (r.visual, r.audio, r.frames, r.timestamps) if t is not None)

    def save(self, path_no_ext):
        # A render-time selection must never discard the unselected recordings.
        profile = replace(self.profile, name=self.name, description=self.description)
        first_voice = next(r for r in profile.references if r.audio is not None)
        default_tokens = sum(r.token_count for r in profile.references if r.audio is None or r is first_voice)
        meta = {"kind": "character", "name": self.name, "description": self.description,
                "concept_type": self.concept_type, "_format_version": 5,
                "refmod_config": self.config, "stored_references": len(profile.references),
                "selected_tokens": default_tokens}
        return profile.save(path_no_ext + ".safetensors", overwrite=True, refmod_metadata=meta)

    def ref_block(self, *args, **kwargs):
        raise ValueError("A combined RefMod contains several blocks. Use the updated Apply H3 RefMod.")

    def ref_blocks(self, strength=1.0, curve=None):
        if strength <= 0:
            return []
        blocks = []
        for ref in self.selected_references():
            block = ref.block()
            if ref.kind == "refmod_visual":
                # Use exactly the author's strength/curve operation on the
                # original visual latent, including its single-frame rule.
                original = H3RefMod(self.name, block["kind"], ref.visual.clone(),
                    latent_t=ref.visual.shape[2], latent_h=ref.visual.shape[3], latent_w=ref.visual.shape[4])
                used_curve, visual_strength = curve, strength
                if (original.latent_t <= 1 and isinstance(curve, tuple) and len(curve) == 3
                        and isinstance(curve[0], str) and curve[0] != "constant"):
                    visual_strength = min(strength, max(0., min(1., float(curve[2]))))
                    used_curve = None
                if visual_strength <= 0:
                    raise ValueError("This curve removes a numbered character visual. Use a positive curve_value, "
                                     "or disable the character at its loader slot before text encoding.")
                block = original.ref_block(visual_strength, used_curve)
                block.update(refmod=True, refmod_profile=True, refmod_original_visual=True,
                             character_id=self.profile.character_id, refmod_subject=self.subject_slot)
                blocks.append(block)
                continue
            for key in ("latent", "audio_latent"):
                z = block.get(key)
                if z is None:
                    continue
                t = z.shape[2] if key == "latent" else z.shape[-1]
                values = curve_strengths(curve, t) if curve is not None else [1.0] * t
                if values is None:
                    values = [1.0] * t
                weights = z.new_tensor([max(0.0, min(1.0, strength * v)) for v in values])
                if torch.any(weights < 1):
                    weights = weights.view((1, 1, t, 1, 1) if key == "latent" else (1, 1, 1, t))
                    block[key] = weights * z + (1 - weights) * _blur_latent(z)
            block.update(refmod=True, refmod_profile=True, character_id=self.profile.character_id,
                         refmod_subject=self.subject_slot)
            blocks.append(block)
        return blocks

    def report(self):
        selected = [r for r in self.selected_references() if r.audio is not None]
        stored = sum(r.audio is not None for r in self.profile.references)
        duration = sum(r.duration for r in selected)
        choice = "all" if self.voice_reference == 0 else str(self.voice_reference)
        return f"{self.name}: voice example {choice}; {len(selected)}/{stored} recordings selected, {duration:.2f}s, {self.token_count:,} tokens"
