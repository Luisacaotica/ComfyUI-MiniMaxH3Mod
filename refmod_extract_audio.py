"""Audio-capable extraction behind the existing Extract H3 RefMod node."""
import math

import torch

from .character_core import CharacterReference, H3CharacterMod
from .character_media import FPS, aligned_frames, load_character_clip, normalize_audio, resize_visual
from .character_nodes import _encode_audio, _encode_visual, _validate_vae, media_path
from .core import aspect_grid, optimize_latent, pool_latent
from .refmod_profile import ProfileRefMod


def extract_profile(name, ordered, vae, audio_vae, audio=None, refs_audio=None,
                    video_file="", audio_start_seconds=0.0, audio_duration_seconds=5.0,
                    ref_resolution=768, mode="encode", pool_h=16, pool_w=16,
                    identity=0, description="", merge=False, motion_only=False,
                    multiplier=1, mask=None):
    _validate_vae(vae, "video")
    _validate_vae(audio_vae, "audio")
    if merge or motion_only or multiplier != 1 or mask is not None:
        raise ValueError("Combined appearance/voice extraction keeps separate reference views and timing. "
                         "Set merge=False, motion_only=False, multiplier=1 and disconnect mask.")
    if not math.isfinite(audio_start_seconds) or audio_start_seconds < 0:
        raise ValueError("Audio/video start must be non-negative.")
    if not math.isfinite(audio_duration_seconds) or not 0.25 <= audio_duration_seconds <= 15:
        raise ValueError("Select 0.25-15 seconds from each source recording.")
    refs = []

    def add_visual(pixels, kind, duration=0.0, waveform=None):
        prepared = resize_visual(pixels, ref_resolution)
        z = _encode_visual(vae, prepared)
        if mode == "training":
            gh, gw = aspect_grid(pool_h, pool_w, prepared.shape[1] / prepared.shape[2])
            small = pool_latent(z, z.shape[2], gh, gw)
            z = optimize_latent(small, z, steps=identity) if identity else small
        indices = torch.arange(0, len(prepared), FPS // 2) if kind != "image" else torch.tensor([0])
        refs.append(CharacterReference(kind, visual=z, frames=prepared[indices].cpu().float().contiguous(),
                    audio=_encode_audio(audio_vae, waveform) if waveform is not None else None,
                    timestamps=indices.double() / FPS if kind != "image" else None, duration=duration))

    for pixels, is_video in ordered:
        if not is_video:
            add_visual(pixels[:1], "image")
            continue
        start = round(audio_start_seconds * FPS)
        count = aligned_frames(min(max(0, len(pixels) - start), math.floor(audio_duration_seconds * FPS)))
        add_visual(pixels[start:start + count], "video", duration=count / FPS)
    if str(video_file).strip():
        clip = load_character_clip(media_path(video_file), audio_start_seconds, audio_duration_seconds,
                                   max_edge=min(4096, ref_resolution * 2))
        waveform, duration = normalize_audio(clip.audio, 0, clip.duration, min_duration_seconds=5 / FPS)
        add_visual(clip.frames, "video_audio", duration=duration, waveform=waveform)
    sources = ([] if audio is None else [audio]) + [v for _, v in sorted(
        (refs_audio or {}).items(), key=lambda pair: int(pair[0].rsplit("_", 1)[-1])) if v is not None]
    for source in sources:
        waveform, duration = normalize_audio(source, audio_start_seconds, audio_duration_seconds)
        refs.append(CharacterReference("audio", audio=_encode_audio(audio_vae, waveform), duration=duration))
    profile = H3CharacterMod(name, refs, description, provenance={
        "producer": "Extract H3 RefMod", "mode": mode, "reference_short_edge": ref_resolution,
        "audio_start_seconds": audio_start_seconds, "audio_duration_seconds": audio_duration_seconds,
        "video_file": str(video_file), "video_fps": FPS})
    profile.validate()
    return ProfileRefMod.from_profile(profile)
