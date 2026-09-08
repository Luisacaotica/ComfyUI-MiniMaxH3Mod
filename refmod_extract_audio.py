"""Audio-capable extraction behind the existing Extract H3 RefMod node."""
import math

import torch

from .character_core import CharacterReference, H3CharacterMod
from .character_media import FPS, load_character_clip, normalize_audio, resize_visual
from .character_nodes import _encode_audio, _encode_visual, _validate_vae, media_path
from .refmod_profile import ProfileRefMod


def extract_with_original_visual(extractor, name, ordered, vae, audio_vae, *, audio=None,
                                 refs_audio=None, video_file="", audio_start_seconds=0.,
                                 audio_duration_seconds=5., visual_options=None):
    """Extend the original extractor, retaining its complete visual computation.

    A speaking file also contributes a genuine paired performance. The original
    identity stack and the timed performance have different roles: never assign
    physical audio timestamps to the author's sampled/pooled identity stack.
    """
    _validate_vae(vae, "video")
    _validate_vae(audio_vae, "audio")
    if not math.isfinite(audio_start_seconds) or audio_start_seconds < 0:
        raise ValueError("Audio/video start must be non-negative.")
    if not math.isfinite(audio_duration_seconds) or not 0.25 <= audio_duration_seconds <= 15:
        raise ValueError("Select 0.25-15 seconds from each source recording.")
    options = dict(visual_options or {})
    sources = list(ordered)
    paired_clip = None
    if str(video_file).strip():
        paired_clip = load_character_clip(media_path(video_file), audio_start_seconds,
            audio_duration_seconds, max_edge=2048)
        sources.append((paired_clip.frames.float() / 255, True))
    if not sources:
        raise ValueError("Connect a character image/video as well as voice audio, or provide a speaking video_file.")

    class CaptureVAE:
        def __init__(self):
            self.preview = None

        def __getattr__(self, key):
            return getattr(vae, key)

        def encode(self, pixels):
            if self.preview is None:
                # One first-view presentation per original identity stack.
                # All views remain encoded in the original visual latent.
                self.preview = pixels[:1].detach().cpu().float().clone()
            return vae.encode(pixels)

    capture = CaptureVAE()
    # Preserve mixed folder order too; regrouping by modality would change
    # the author's first-source canvas, mask indices and stack ordering.
    bundle = [pixels if is_video else pixels[:1] for pixels, is_video in sources]
    result = extractor(name=name, vae=capture, refs_bundle=bundle,
                       save=False, **options)
    original = result[0][0][0]
    refs = [CharacterReference("refmod_visual", visual=original.latent.detach().cpu().clone(),
                               frames=capture.preview)]
    if paired_clip is not None:
        # Separate encoding is intentional: the original identity extractor
        # samples frames for likeness; this extra reference preserves lip timing.
        prepared = resize_visual(paired_clip.frames, options.get("ref_resolution", 768))
        indices = torch.arange(0, len(prepared), FPS // 2)
        waveform, duration = normalize_audio(paired_clip.audio, 0, paired_clip.duration,
                                             min_duration_seconds=5 / FPS)
        refs.append(CharacterReference("video_audio", visual=_encode_visual(vae, prepared),
            audio=_encode_audio(audio_vae, waveform), frames=prepared[indices].cpu().float(),
            timestamps=indices.double() / FPS, duration=duration))
    recordings = ([] if audio is None else [audio]) + [value for _, value in sorted(
        (refs_audio or {}).items(), key=lambda row: int(row[0].rsplit("_", 1)[-1])) if value is not None]
    for source in recordings:
        waveform, duration = normalize_audio(source, audio_start_seconds, audio_duration_seconds)
        refs.append(CharacterReference("audio", audio=_encode_audio(audio_vae, waveform), duration=duration))
    provenance = {"producer": "Extract H3 RefMod", "visual_path": "original_extractor",
        "visual_metadata": {key: getattr(original, key) for key in
            ("mode", "source", "source_shape", "pool", "optimize_steps", "tags", "concept_type")},
        "video_file": str(video_file), "paired_performance": paired_clip is not None,
        "audio_start_seconds": audio_start_seconds, "audio_duration_seconds": audio_duration_seconds}
    profile = H3CharacterMod(name, refs, original.description, provenance=provenance)
    mod = ProfileRefMod.from_profile(profile)
    mod.concept_type = original.concept_type
    return mod
