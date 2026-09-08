"""Present saved RefMod references to H3's native text/vision encoder."""

import math
import inspect

from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


class MiniMaxH3RefModTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "mods": ("H3_REF_MODS",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "reference_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                "tooltip": "Playback FPS of the reconstructed video, used for Qwen timestamps. Saved compressed or stacked refs do not preserve original timing."}),
            "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
        }, "optional": {
            "vae": ("VAE", {"tooltip": "H3 video VAE: reconstructs visual refs for Qwen. Not needed for audio-only bundles."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "reference_map")
    FUNCTION = "encode"
    CATEGORY = "MiniMax-H3/mod"
    DESCRIPTION = "Encode the prompt together with numbered RefMod references. Already attaches refs; connect directly to the sampler, without applying the same mods again."

    def encode(self, clip, mods, prompt, reference_fps=24.0, max_total_tokens=0, vae=None):
        from .nodes import _check_token_budget

        native = isinstance(clip.tokenizer, MiniMaxH3Tokenizer)
        # Projected encoders retain their smaller model's tokenizer, but expose
        # H3 reference presentation on the CLIP wrapper itself (e.g. ClipProj).
        if not native and "minimax_ref_items" not in inspect.signature(clip.tokenize).parameters:
            raise ValueError("Connect an H3 CLIP or a projected CLIP supporting minimax_ref_items (such as current ClipProj).")
        if not math.isfinite(reference_fps) or not 1 <= reference_fps <= 120:
            raise ValueError("reference_fps must be between 1 and 120.")
        # Keep zero-strength slots out of both the presentation and DiT payload.
        active = []
        for mod, strength in mods:
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("RefMod strength must be between 0 and 1.")
            if strength > 0:
                active.append((mod, strength))
        _check_token_budget(active, max_total_tokens)
        if any(mod.kind != "audio" for mod, _ in active) and vae is None:
            raise ValueError("Connect the H3 video VAE to present visual RefMods to CLIP.")
        if any(mod.kind != "audio" for mod, _ in active) and not isinstance(vae.first_stage_model, MiniMaxH3VideoVAE):
            raise ValueError("Visual RefMods require the MiniMax H3 video VAE.")

        items, blocks, mapping = [], [], []
        counters = {"image": 0, "video": 0, "audio": 0}
        labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
        for mod, strength in active:
            block = mod.ref_block(strength)
            block["refmod"] = True
            kind = block["kind"]
            counters[kind] += 1
            label = f"<{labels[kind]} {counters[kind]}>"
            mapping.append(f"{label} = {mod.name}")
            item = {"type": kind}
            if kind != "audio":
                # Decode the same weakened latent that the DiT receives. ComfyUI
                # owns device placement and its decode OOM/tiled fallback.
                pixels = vae.decode(block["latent"])
                # ComfyUI video VAEs return BTHWC; older wrappers may already
                # expose THWC. Each RefMod is one video, never a batch of videos.
                if pixels.ndim == 5 and pixels.shape[0] == 1:
                    pixels = pixels[0]
                if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.shape[0] < 1:
                    raise ValueError(f"H3 video VAE must decode to [1, frames, height, width, 3] or [frames, height, width, 3]; got {tuple(pixels.shape)}.")
                if kind == "image":
                    item["data"] = pixels[:1].cpu().clone()
                else:
                    # Native H3 presents video at 2 fps. Index by timestamps so
                    # non-integer frame rates do not accumulate rounding drift.
                    times = [i / 2 for i in range(math.ceil(pixels.shape[0] * 2 / reference_fps))]
                    indices = [min(round(t * reference_fps), pixels.shape[0] - 1) for t in times]
                    item["data"] = pixels[indices].cpu()
                    item["timestamps"] = times
                del pixels
            items.append(item)
            blocks.append(block)

        tokens = clip.tokenize(prompt, minimax_ref_items=items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        out = []
        for embedding, metadata in conditioning:
            if "minimax_token_tags" not in metadata:
                raise ValueError("The CLIP encoder did not return H3 minimax_token_tags. Use an H3-compatible encoder/projection.")
            metadata = dict(metadata)
            if blocks:
                metadata["minimax_refs"] = list(blocks)
            out.append([embedding, metadata])
        return out, "\n".join(mapping) or "No active RefMods."
