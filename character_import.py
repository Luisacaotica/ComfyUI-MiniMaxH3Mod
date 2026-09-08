"""Convert supported upstream image/audio RefMods without re-encoding latents."""
from __future__ import annotations

import torch

from .core import H3RefMod
from .character_core import CharacterReference, H3CharacterMod, safe_name


def import_refmods(mods, source_images, name, description="", resize_image=None):
    if not mods:
        raise ValueError("Connect a Master or Load H3 RefMods bundle for ONE character.")
    visual, audio, seen = [], [], set()
    for mod, strength in mods:
        if not isinstance(mod, H3RefMod):
            raise ValueError("Expected upstream H3 RefMods from Master or Load H3 RefMods.")
        if isinstance(strength, bool) or strength != 1.0:
            raise ValueError("Import requires strength=1.0 and copies=1; use the original unmodified references.")
        identity = mod.path or id(mod)
        if identity in seen:
            raise ValueError("Duplicate RefMod in character import. Set copies=1 and select each file once.")
        seen.add(identity)
        if mod.kind not in ("image", "audio"):
            raise ValueError("Video/stacked RefMods do not retain enough source timing for safe pairing. "
                             "Re-extract the original speaking video with Load H3 Character Clip.")
        if mod.kind == "image" and mod.mode not in ("encode", "full"):
            raise ValueError("Import requires encode-mode RefMods. Re-extract pooled/training visuals in encode mode.")
        (visual if mod.kind == "image" else audio).append(mod)
    if not visual or not audio:
        raise ValueError("Select at least one image RefMod AND one audio RefMod for the same character.")
    if len(visual) > 9 or len(audio) > 3:
        raise ValueError("A character import supports at most 9 images and 3 audio clips.")
    images = []
    for batch in source_images:
        if (not isinstance(batch, torch.Tensor) or batch.ndim != 4 or batch.shape[-1] != 3
                or min(batch.shape) < 1):
            raise ValueError("Matching source images must be nonempty RGB IMAGE batches.")
        images.extend(batch.split(1))
    if len(images) != len(visual):
        raise ValueError(f"Provide {len(visual)} matching original image(s), in the visual RefMods' loader order. "
                         f"Received {len(images)}. Upstream files do not store vision-encoder pixels.")
    refs = []
    for mod, image in zip(visual, images):
        pixels = image.detach().cpu().float()
        if image.dtype == torch.uint8:
            pixels = pixels / 255
        z = mod.latent.detach().cpu().clone()
        if z.ndim != 5 or z.shape[:3] != (1, 24, 1):
            raise ValueError("An image RefMod must contain one full H3 image latent [1,24,1,H,W].")
        target = (z.shape[3] * 16, z.shape[4] * 16)
        if resize_image is not None:
            pixels = resize_image(pixels, target)
        if tuple(pixels.shape[1:3]) != target:
            raise ValueError(f"Source image does not match the saved encode canvas {target}. "
                             "Use the original image without changing its crop/aspect ratio.")
        refs.append(CharacterReference("image", visual=z, frames=pixels.contiguous().clone()))
    for mod in audio:
        if mod.sample_rate != 32000:
            raise ValueError("Audio RefMods must use the native H3 32 kHz codec.")
        refs.append(CharacterReference("audio", audio=mod.latent.detach().cpu().clone(),
                                       duration=mod.latent.shape[-1] / 40))
    if sum(ref.duration for ref in refs) > 15.0001:
        raise ValueError("Imported voice clips exceed H3's 15-second total reference-audio budget. "
                         "Extract shorter recordings or select fewer files.")
    result = H3CharacterMod(safe_name(name), refs, description.strip(), provenance={
        "producer": "ComfyUI-MiniMaxH3Mod/0.3.0", "import": "upstream-image-audio",
        "source_mods": [{"name": mod.name, "kind": mod.kind, "mode": mod.mode} for mod, _ in mods],
        "vision_source": "user-supplied original images; source identity cannot be verified from old metadata",
        "paired": False,
    })
    result.validate()
    return result
