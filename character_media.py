"""Bounded, timestamp-aware decoding and preprocessing for character assets."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

FPS = 24
SAMPLE_RATE = 32000


def _av():
    try:
        import av
        return av
    except ImportError as exc:
        raise RuntimeError("Install this pack's requirements.txt in ComfyUI's Python (missing PyAV: av).") from exc


def aligned_frames(count):
    if count < 5:
        raise ValueError("A speaking clip needs at least 5 frames at 24 fps.")
    return 5 + ((count - 5) // 17) * 17


@dataclass
class CharacterClip:
    frames: torch.Tensor  # uint8 RGB [T,H,W,3], constant 24 fps
    audio: dict          # ComfyUI AUDIO, cropped to the identical interval
    start_seconds: float
    duration: float
    source_name: str = ""


def _time(frame):
    if frame.pts is None or frame.time_base is None:
        raise ValueError("Media has no usable timestamps; re-encode it before creating a paired character.")
    return float(frame.pts * frame.time_base)


def load_character_clip(path, start_seconds=0.0, duration_seconds=5.0, max_edge=768):
    """Use one video time origin for BOTH streams, including audio stream offsets.

    Decode only the selected interval into memory. Video is sampled at real
    1/24-second intervals; it is never squeezed from a long clip into 16 frames.
    PyAV bundles FFmpeg libraries; no external ffmpeg executable is required.
    """
    from pathlib import Path
    av = _av()
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Clip start must be a non-negative number.")
    if not math.isfinite(duration_seconds) or not 0.25 <= duration_seconds <= 15:
        raise ValueError("Select a clip duration between 0.25 and 15 seconds.")
    if not 64 <= max_edge <= 2048:
        raise ValueError("max_edge must be between 64 and 2048.")
    selected = []
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError("Select a video containing both a picture stream and a soundtrack.")
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        start = origin + start_seconds
        end = start + duration_seconds
        rate = float(stream.average_rate or FPS)
        scale = min(1.0, max_edge / max(stream.width, stream.height))
        width, height = max(1, round(stream.width * scale)), max(1, round(stream.height * scale))
        container.seek(max(0, int((start - 1) * av.time_base)), backward=True)
        previous, last_end, next_time = None, None, start
        for frame in container.decode(stream):
            timestamp = _time(frame)
            if previous is not None:
                while next_time < min(timestamp, end) - 1e-7:
                    selected.append(previous)
                    next_time = start + len(selected) / FPS
            if next_time >= end - 1e-7:
                break
            if timestamp >= end:
                break
            previous = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            frame_ticks = getattr(frame, "duration", 0)
            frame_duration = float(frame_ticks * frame.time_base) if frame_ticks else 1 / rate
            last_end = timestamp + frame_duration
        if previous is not None and last_end is not None:
            while next_time < min(end, last_end) - 1e-7:
                selected.append(previous)
                next_time = start + len(selected) / FPS
    count = aligned_frames(min(len(selected), math.floor(duration_seconds * FPS + 1e-7)))
    duration = count / FPS
    frames = torch.from_numpy(np.stack(selected[:count]).copy())
    sample_count = round(duration * SAMPLE_RATE)
    waveform = np.zeros((2, sample_count), dtype=np.float32)
    covered = 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
        container.seek(max(0, int((start - 1) * av.time_base)), backward=True)

        def copy_audio(frame):
            nonlocal covered
            offset = round((_time(frame) - start) * SAMPLE_RATE)
            samples = frame.to_ndarray()
            left, right = max(0, offset), min(sample_count, offset + samples.shape[-1])
            if right > left:
                waveform[:, left:right] = samples[:, left - offset:right - offset]
                covered += right - left

        for frame in container.decode(stream):
            if _time(frame) > start + duration + 0.1:
                break
            for converted in resampler.resample(frame):
                copy_audio(converted)
        for converted in resampler.resample(None):
            copy_audio(converted)
    if covered == 0:
        raise ValueError("The soundtrack does not overlap the selected video interval.")
    audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    return CharacterClip(frames, audio, start_seconds, duration, Path(path).name)


def normalize_audio(audio, start_seconds=0.0, duration_seconds=5.0, min_duration_seconds=0.25):
    """Comfy AUDIO -> stereo fp32 at 32 kHz. Keep utterances separate."""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Connect a ComfyUI AUDIO input.")
    z = audio["waveform"]
    if not isinstance(z, torch.Tensor) or z.ndim != 3 or z.shape[0] != 1 or z.shape[1] not in (1, 2):
        raise ValueError("Voice audio must be one mono or stereo recording [1,1/2,samples].")
    sr = int(audio["sample_rate"])
    if sr <= 0 or not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Invalid audio sample rate or start time.")
    if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 15:
        raise ValueError("Audio duration must be positive and no more than 15 seconds.")
    left = round(start_seconds * sr)
    right = min(z.shape[-1], left + round(duration_seconds * sr))
    if right <= left:
        raise ValueError("The selected voice interval is empty.")
    z = z[:, :, left:right].detach().cpu().float()
    if not torch.isfinite(z).all():
        raise ValueError("Voice recording contains non-finite samples.")
    if z.shape[1] == 1:
        z = z.repeat(1, 2, 1)
    z = z.clamp(-1, 1)
    if sr != SAMPLE_RATE:
        # Match the native H3 reference node's waveform resampling.
        import torchaudio
        z = torchaudio.functional.resample(z, sr, SAMPLE_RATE)
    if z.shape[-1] < round(SAMPLE_RATE * min_duration_seconds):
        raise ValueError(f"Use at least {min_duration_seconds:.3f} seconds of voice audio.")
    return z.contiguous(), z.shape[-1] / SAMPLE_RATE


def resize_visual(frames, short_edge=512):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Visual input must be RGB [frames,height,width,3].")
    if min(frames.shape) <= 0:
        raise ValueError("Visual input is empty.")
    frames = frames.detach().cpu()
    pixels = frames.float() / 255 if frames.dtype == torch.uint8 else frames.float()
    if not torch.isfinite(pixels).all():
        raise ValueError("Visual input contains non-finite pixels.")
    h, w = pixels.shape[1:3]
    # Preserve aspect and honor the H3 tiler's minimum short edge.
    scale = min(1.0, short_edge / min(h, w))
    scale = max(scale, 320 / min(h, w))
    th, tw = max(320, round(h * scale / 32) * 32), max(320, round(w * scale / 32) * 32)
    if max(th, tw) > 4096:
        raise ValueError("Reference aspect ratio is too extreme; crop it before extraction.")
    # Same PIL Lanczos operation as ComfyUI's native H3 _resize, including
    # its byte conversion even when dimensions are unchanged. Saving the
    # resulting float pixels adds no further transformation at reload time.
    from PIL import Image
    prepared = []
    for image in pixels:
        array = np.clip(255 * image.numpy(), 0, 255).astype(np.uint8)
        resized = Image.fromarray(array).resize((tw, th), Image.Resampling.LANCZOS)
        prepared.append(torch.from_numpy(np.array(resized).astype(np.float32) / 255))
    return torch.stack(prepared)


def vision_bytes(pixels):
    return (pixels.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
