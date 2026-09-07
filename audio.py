"""H3 audio encoding, adapted from the user's ComfyUI-H3AudioMod.

Uses its tested normalized [1,32,2,T] representation and bounded 10-second
encode path. Curves, storage and loaders are shared with visual RefMods.
"""

import math
import torch
import comfy.model_management as model_management
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from .core import H3RefMod


def encode_audio(vae, audio, max_seconds=30.0, chunk_seconds=10.0):
    waveform, sample_rate = audio["waveform"], int(audio["sample_rate"])
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("Audio must have one batch of mono or stereo samples [1,C,L].")
    if sample_rate <= 0 or waveform.shape[-1] < 1:
        raise ValueError("Audio is empty or has an invalid sample rate.")
    if not math.isfinite(max_seconds) or not math.isfinite(chunk_seconds) or max_seconds <= 0 or chunk_seconds <= 0:
        raise ValueError("Audio duration and chunk length must be positive.")
    model = vae.first_stage_model
    if not isinstance(model, MiniMaxH3AudioVAE):
        raise ValueError("Connect the MiniMax H3 audio VAE, not the video VAE or another audio codec.")
    waveform = waveform[..., :max(1, round(max_seconds * sample_rate))]
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if sample_rate != 32000:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)
    model_management.load_models_gpu([vae.patcher], memory_required=0, force_full_load=True)
    chunk = max(800, round(chunk_seconds * 40) * 800)
    latents = []
    with model_management.cuda_device_context(vae.device):
        for start in range(0, waveform.shape[-1], chunk):
            piece = waveform[..., start:start + chunk].to(device=vae.device, dtype=vae.vae_dtype)
            z = model.encode(piece).detach().cpu()
            if z.ndim != 4 or tuple(z.shape[:3]) != (1, 32, 2):
                raise ValueError(f"H3 audio VAE returned an invalid latent: {tuple(z.shape)}")
            latents.append(z)
    return torch.cat(latents, dim=-1)


def make_audio_mod(vae, audio, name, max_seconds=30.0, max_tokens=5120,
                   budget_policy="error", description="", concept_type="voice"):
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("Audio token budget must be a non-negative integer.")
    if budget_policy not in ("error", "truncate"):
        raise ValueError("Audio budget policy must be error or truncate.")
    latent = encode_audio(vae, audio, max_seconds)
    if max_tokens and latent.shape[-1] * 2 > max_tokens:
        if max_tokens < 2 or budget_policy == "error":
            raise ValueError(f"Audio requires {latent.shape[-1] * 2} tokens; budget is {max_tokens}. "
                             "Lower max_seconds or choose truncate.")
        if budget_policy != "truncate":
            raise ValueError("Audio budget policy must be error or truncate.")
        latent = latent[..., :max_tokens // 2].clone()
    return H3RefMod(name=name, kind="audio", latent=latent, mode="encode", source="audio",
                    description=description, concept_type=concept_type, sample_rate=32000)
