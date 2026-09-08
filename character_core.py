"""Portable character references. No ComfyUI imports or model-weight patches."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file

META_KEY = "h3_character_meta"
FORMAT_VERSION = 3
MODEL_FAMILY = "minimax_h3_ref2va"


def safe_name(name: str) -> str:
    name = str(name).strip()
    if name.lower().endswith(".safetensors"):
        name = name[:-12]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
    if not name or name.split(".")[0].upper() in reserved:
        raise ValueError("Enter a valid character filename.")
    return name


def _tensor(z, shape, label, floating=True):
    if not isinstance(z, torch.Tensor) or z.ndim != len(shape):
        raise ValueError(f"{label}: expected tensor shape {shape}.")
    if any(size <= 0 or (wanted is not None and size != wanted)
           for size, wanted in zip(z.shape, shape)):
        raise ValueError(f"{label}: expected {shape}, got {tuple(z.shape)}.")
    if floating and (not z.is_floating_point() or not torch.isfinite(z).all()):
        raise ValueError(f"{label}: expected finite floating-point values.")


@dataclass
class CharacterReference:
    kind: str
    visual: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    frames: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None
    duration: float = 0.0

    def validate(self):
        if self.kind not in ("image", "audio", "video", "video_audio", "refmod_visual"):
            raise ValueError(f"Unsupported character reference kind: {self.kind!r}")
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("Reference duration must be finite and non-negative.")
        if self.kind in ("image", "video", "video_audio", "refmod_visual"):
            _tensor(self.visual, (1, 24, None, None, None), "H3 video latent")
            if self.visual.shape[-1] % 2 or self.visual.shape[-2] % 2:
                raise ValueError("H3 video latent spatial dimensions must be even.")
            _tensor(self.frames, (None, None, None, 3), "Vision frames", floating=False)
            if self.frames.dtype != torch.uint8:
                if (self.frames.dtype != torch.float32 or not torch.isfinite(self.frames).all()
                        or self.frames.min() < 0 or self.frames.max() > 1):
                    raise ValueError("Vision frames must be uint8 RGB or finite float32 RGB in [0,1].")
            if self.kind == "image" and (self.visual.shape[2] != 1 or len(self.frames) != 1):
                raise ValueError("An image reference must contain one frame.")
            if self.kind == "refmod_visual" and (len(self.frames) != 1 or self.duration != 0):
                raise ValueError("An original RefMod stack has one presentation image and no physical duration.")
        elif self.visual is not None or self.frames is not None:
            raise ValueError("Standalone audio references cannot contain video tensors.")
        if self.kind in ("audio", "video_audio"):
            _tensor(self.audio, (1, 32, 2, None), "H3 audio latent")
            if self.duration <= 0:
                raise ValueError("Audio reference duration must be positive.")
            if abs(self.audio.shape[-1] / 40.0 - self.duration) > 0.026:
                raise ValueError("Audio duration disagrees with the 40 Hz H3 latent grid.")
        elif self.audio is not None:
            raise ValueError("Use a paired video reference to store a soundtrack.")
        if self.kind in ("video", "video_audio"):
            _tensor(self.timestamps, (len(self.frames),), "Vision timestamps")
            if self.timestamps[0] < 0 or self.timestamps[-1] >= self.duration:
                raise ValueError("Vision timestamps fall outside the paired clip.")
            if len(self.timestamps) > 1 and not torch.all(self.timestamps[1:] > self.timestamps[:-1]):
                raise ValueError("Vision timestamps must be strictly increasing.")
            tv = self.visual.shape[2]
            # Current H3: 5 pixels -> 2 latents, each additional 17 pixels -> 5.
            if tv < 2 or (tv - 2) % 5:
                raise ValueError("Unsupported H3 paired-video temporal geometry.")
            pixel_frames = 5 + ((tv - 2) // 5) * 17
            if abs(pixel_frames / 24.0 - self.duration) > 1 / 32000:
                raise ValueError("Video and soundtrack durations are not aligned.")
        elif self.timestamps is not None:
            raise ValueError("Only video references have presentation timestamps.")

    @property
    def token_count(self):
        visual = 0 if self.visual is None else (
            self.visual.shape[2] * (self.visual.shape[3] // 2) * (self.visual.shape[4] // 2))
        return visual + (0 if self.audio is None else 2 * self.audio.shape[-1])

    def vision_pixels(self):
        """v1 byte compatibility; v2 preserves prepared pixels without quantizing."""
        if self.frames is None:
            return None
        if self.frames.dtype == torch.uint8:
            return self.frames.float() / 255
        return self.frames.clone()

    def block(self):
        # Clone: downstream wrappers may mutate their payload, never the asset.
        block = {"kind": self.kind}
        if self.kind == "refmod_visual":
            # Original RefMod stacks may be arbitrary T, not a real video clock.
            block["kind"] = "video" if self.visual.shape[2] > 1 else "image"
        if self.visual is not None:
            block.update(latent=self.visual.clone(), latent_h=self.visual.shape[3],
                         latent_w=self.visual.shape[4])
        if block["kind"] in ("video", "video_audio"):
            block["latent_t"] = self.visual.shape[2]
            block.update(ref_audio_t=0, audio_latent=None)
        if self.audio is not None:
            block.update(audio_latent=self.audio.clone(), ref_audio_t=self.audio.shape[-1])
        return block


@dataclass
class H3CharacterMod:
    name: str
    references: list[CharacterReference]
    description: str = ""
    character_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provenance: dict = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Character name is missing.")
        if not isinstance(self.description, str) or not isinstance(self.provenance, dict):
            raise ValueError("Invalid character metadata.")
        if not isinstance(self.character_id, str) or not self.character_id:
            raise ValueError("Character identifier is missing.")
        if not 1 <= len(self.references) <= 64:
            raise ValueError("A character must contain 1-64 stored references.")
        for ref in self.references:
            ref.validate()
        if not any(r.visual is not None for r in self.references):
            raise ValueError("A character needs at least one visual reference.")
        if not any(r.audio is not None for r in self.references):
            raise ValueError("A character needs at least one voice reference.")

    @property
    def token_count(self):
        return sum(r.token_count for r in self.references)

    def summary(self):
        kinds = ", ".join(r.kind for r in self.references)
        seconds = sum(r.duration for r in self.references if r.audio is not None)
        return f"{self.name}: {kinds}; {seconds:.2f}s audio; {self.token_count:,} reference tokens"

    def save(self, path, overwrite=False, refmod_metadata=None):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Character already exists: {path.name}. Enable overwrite or use another name.")
        tensors, entries = {}, []
        for i, ref in enumerate(self.references):
            entry = {"kind": ref.kind, "duration": ref.duration}
            for field_name in ("visual", "audio", "frames", "timestamps"):
                tensor = getattr(ref, field_name)
                if tensor is not None:
                    key = f"refs.{i:03d}.{field_name}"
                    tensors[key] = tensor.detach().cpu().contiguous().clone()
                    entry[field_name] = key
            entries.append(entry)
        meta = {"format_version": FORMAT_VERSION, "model_family": MODEL_FAMILY,
                "name": self.name, "description": self.description,
                "character_id": self.character_id, "references": entries,
                "provenance": self.provenance}
        fd, temporary = tempfile.mkstemp(prefix=".character-", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            header = {META_KEY: json.dumps(meta, allow_nan=False)}
            if refmod_metadata is not None:
                header["refmod_meta"] = json.dumps(refmod_metadata, allow_nan=False)
            save_file(tensors, temporary, metadata=header)
            if overwrite:
                os.replace(temporary, path)
            else:
                # Atomic no-clobber publication, including concurrent extractions.
                os.link(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)

    @classmethod
    def load(cls, path):
        meta = read_character_metadata(path)
        tensors = load_file(str(path), device="cpu")
        refs = []
        try:
            for entry in meta["references"]:
                fields = {key: tensors[entry[key]].clone() for key in
                          ("visual", "audio", "frames", "timestamps") if key in entry}
                refs.append(CharacterReference(kind=entry["kind"], duration=float(entry["duration"]), **fields))
            mod = cls(meta["name"], refs, meta.get("description", ""), meta["character_id"],
                      meta.get("provenance", {}))
            mod.validate()
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid character file {Path(path).name}: {exc}") from exc
        return mod


def read_character_metadata(path):
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw = (handle.metadata() or {}).get(META_KEY)
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors file: {Path(path).name}") from exc
    if raw is None:
        raise ValueError("This is not an H3 Character file. Use Load H3 RefMods for older visual mods.")
    meta = json.loads(raw)
    if not isinstance(meta, dict) or meta.get("format_version") not in (1, 2, FORMAT_VERSION):
        raise ValueError("Unsupported H3 Character format version.")
    if meta.get("model_family") != MODEL_FAMILY:
        raise ValueError("This character is not compatible with MiniMax H3 Ref2VA.")
    if not isinstance(meta.get("references"), list) or not 1 <= len(meta["references"]) <= 64:
        raise ValueError("Invalid character reference list.")
    return meta
