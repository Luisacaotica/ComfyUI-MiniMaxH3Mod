"""One reference traversal for dialogue, full prompts, and controlled comparisons."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from .character_core import H3CharacterMod
from .character_binding import OWNER_KEY, ID_KEY


def tensor_digest(tensor):
    if tensor is None:
        return None
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass
class ReferenceAssembly:
    items: list
    blocks: list
    labels: list
    pairing: list
    tokens: int
    audio_seconds: float
    logical_refs: list
    av_layout: str

    def audit(self, characters, prompt, width, height, frame_count, turns=(), presentation="native"):
        sources = []
        for owner, ref in self.logical_refs:
            sources.append({"character": owner + 1, "kind": ref.kind, "duration": ref.duration,
                            "tensors": {k: tensor_digest(getattr(ref, k)) for k in
                                        ("visual", "audio", "frames", "timestamps")}})
        inputs = {"characters": [{"id": mod.character_id, "name": mod.name} for mod in characters],
                  "sources": sources, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "width": width, "height": height, "frame_count": frame_count,
                  "turns": [{"character": t.owner, "start": t.start, "end": t.end} for t in turns]}
        rows = []
        for index, block in enumerate(self.blocks, 1):
            rows.append({"block": index, "character": block[OWNER_KEY], "kind": block["kind"],
                         "visual_shape": list(block["latent"].shape) if "latent" in block else None,
                         "audio_shape": list(block["audio_latent"].shape) if block.get("audio_latent") is not None else None})
        return {"schema": "h3-character-trial-v1", "case_fingerprint": json_digest(inputs),
                "presentation": presentation, "av_layout": self.av_layout,
                "reference_tokens": self.tokens, "audio_seconds": self.audio_seconds,
                "labels": self.labels, "blocks": rows, **inputs}


def assemble_references(characters, max_tokens=0, av_layout="paired"):
    if av_layout not in ("paired", "separate"):
        raise ValueError("AV layout must be paired or separate.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("Reference token cap must be a non-negative integer.")
    if not 1 <= len(characters) <= 3:
        raise ValueError("Connect between one and three characters.")
    refs = []
    for owner, mod in enumerate(characters):
        if not isinstance(mod, H3CharacterMod):
            raise ValueError("Use Extract/Load H3 Character or Import H3 RefMods as Character.")
        mod.validate()
        refs.extend((owner, ref) for ref in mod.references)
    order = {"image": 0, "video_audio": 1, "audio": 2}
    refs.sort(key=lambda pair: order[pair[1].kind])
    n_images = sum(ref.kind == "image" for _, ref in refs)
    n_videos = sum(ref.kind == "video_audio" for _, ref in refs)
    n_audio = sum(ref.audio is not None for _, ref in refs)
    duration = sum(ref.duration for _, ref in refs if ref.audio is not None)
    if n_images > 9 or n_videos > 3 or n_audio > 3 or duration > 15.0001:
        raise ValueError(f"H3 reference limit exceeded: {n_images} images, {n_videos} videos, "
                         f"{n_audio} voice clips, {duration:.2f}s audio. Use at most 9 images, "
                         "3 videos, 3 voice clips and 15s total audio. Select 'first' voice "
                         "in Load H3 Character or extract shorter clips for a multi-character cast.")
    tokens = sum(ref.token_count for _, ref in refs)
    if max_tokens and tokens > max_tokens:
        raise ValueError(f"Characters need {tokens:,} reference tokens, above the {max_tokens:,} budget. "
                         "Re-extract at lower resolution/shorter duration, or raise the budget.")
    labels = [{"visual": [], "audio": []} for _ in characters]
    counters = {"image": 0, "video": 0, "audio": 0}
    items, blocks, pairing = [], [], []
    for owner, ref in refs:
        if ref.kind == "image":
            counters["image"] += 1
            labels[owner]["visual"].append(f"<Picture {counters['image']}>")
            items.append({"type": "image", "data": ref.vision_pixels()})
        elif ref.kind == "video_audio":
            counters["audio"] += 1
            counters["video"] += 1
            audio, video = f"<Audio {counters['audio']}>", f"<Video {counters['video']}>"
            labels[owner]["visual"].append(video)
            labels[owner]["audio"].append(audio)
            pairing.append(f"{audio} is the synchronized soundtrack of {video}.")
            items.extend([{"type": "audio"}, {"type": "video", "data": ref.vision_pixels(),
                                                "timestamps": ref.timestamps.tolist()}])
        else:
            counters["audio"] += 1
            labels[owner]["audio"].append(f"<Audio {counters['audio']}>")
            items.append({"type": "audio"})
        block = ref.block()
        ownership = {OWNER_KEY: owner + 1, ID_KEY: characters[owner].character_id}
        block.update(ownership)
        if ref.kind == "video_audio" and av_layout == "separate":
            # Diagnostic ablation: keep labels, tensors, and order; remove only
            # the shared reference timeline. Never write this back to the asset.
            blocks.append({"kind": "audio", "audio_latent": block.pop("audio_latent"),
                           "ref_audio_t": block["ref_audio_t"], **ownership})
            block.update(kind="video", ref_audio_t=0, audio_latent=None)
        blocks.append(block)
    return ReferenceAssembly(items, blocks, labels, pairing, tokens, duration, refs, av_layout)
