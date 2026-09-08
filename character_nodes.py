"""Character appearance + voice nodes using standard ComfyUI H3 models."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import folder_paths

from .character_core import H3CharacterMod, CharacterReference, read_character_metadata, safe_name
from .character_conditioning import build_character_conditioning, select_voice
from .character_binding import BINDING_KEY, H3CharacterVoiceBinding, make_binding, parse_turns
from .character_references import assemble_references
from .character_media import (
    CharacterClip, FPS, SAMPLE_RATE, load_character_clip, normalize_audio,
    resize_visual,
)

CATEGORY = "MiniMax-H3/character"


def _character_roots():
    try:
        registered = folder_paths.get_folder_paths("refmods")
    except (AttributeError, KeyError):
        registered = []
    roots = [Path(p) / "characters" for p in registered]
    fallback = Path(folder_paths.models_dir) / "refmods" / "characters"
    return list(dict.fromkeys([*roots, fallback]))


def characters_dir():
    path = _character_roots()[0]
    path.mkdir(parents=True, exist_ok=True)
    return path


def character_files():
    names = set()
    for root in _character_roots():
        for path in root.rglob("*.safetensors"):
            try:
                if not path.resolve().is_relative_to(root.resolve()):
                    continue
                read_character_metadata(path)
                names.add(path.relative_to(root).as_posix())
            except (ValueError, OSError):
                continue
    return sorted(names)


def character_path(filename):
    relative = Path(str(filename).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise ValueError("Select a character file inside a registered refmods/characters/ folder.")
    candidates = []
    for directory in _character_roots():
        root = directory.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Character path leaves its registered storage folder.")
        if path.is_file():
            return path
        candidates.append(path)
    return candidates[0]


def media_path(filename):
    path = Path(str(filename).strip().strip('"'))
    if not path.is_absolute():
        path = Path(folder_paths.get_input_directory()) / path
    if not path.is_file():
        raise ValueError(f"Video file not found: {path}")
    return path


def _encoder_name(vae):
    model = getattr(vae, "first_stage_model", None)
    return f"{type(model).__module__}.{type(model).__name__}" if model is not None else type(vae).__name__


def _encode_audio(vae, waveform):
    sample_rate = getattr(vae, "audio_sample_rate", SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Connect the MiniMax H3 32 kHz audio VAE.")
    with torch.no_grad():
        z = vae.encode(waveform.movedim(1, -1))
    if z.ndim != 4 or tuple(z.shape[:3]) != (1, 32, 2):
        raise ValueError(f"Expected H3 audio latent [1,32,2,T], got {tuple(z.shape)}. Check audio_vae.")
    return z.detach().cpu().float().contiguous()


def _encode_visual(vae, pixels):
    with torch.no_grad():
        z = vae.encode(pixels)
    if z.ndim != 5 or tuple(z.shape[:2]) != (1, 24):
        raise ValueError(f"Expected H3 visual latent [1,24,T,H,W], got {tuple(z.shape)}. Check video_vae.")
    return z.detach().cpu().contiguous()


class LoadH3CharacterClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_file": ("STRING", {"default": "character.mp4", "tooltip": "Path relative to ComfyUI/input, or an absolute path. Must include audio."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1}),
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 15.0, "step": 0.1}),
            "max_edge": ("INT", {"default": 768, "min": 320, "max": 2048, "step": 32}),
        }}

    RETURN_TYPES = ("H3_CHARACTER_CLIP", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("character_clip", "preview_frame", "audio", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, video_file, **kwargs):
        st = media_path(video_file).stat()
        return f"{st.st_mtime_ns}:{st.st_size}:{st.st_ctime_ns}"

    def load(self, video_file, start_seconds=0.0, duration_seconds=5.0, max_edge=768):
        clip = load_character_clip(media_path(video_file), start_seconds, duration_seconds, max_edge)
        info = f"{clip.source_name}: {len(clip.frames)} frames at 24 fps, {clip.duration:.3f}s synchronized audio"
        print(f"[H3 Character] {info}")
        return clip, clip.frames[:1].float() / 255, clip.audio, info


class ExtractH3Character:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "my_character"}),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "description": ("STRING", {"default": "", "multiline": True,
                "tooltip": "Character appearance/name. Do not put the reference recording's transcript here."}),
            "reference_short_edge": ("INT", {"default": 512, "min": 320, "max": 1024, "step": 32}),
            "audio_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1}),
            "audio_duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 15.0, "step": 0.1,
                "tooltip": "Applied separately to each standalone voice recording. Paired clips use their saved interval."}),
            "save": ("BOOLEAN", {"default": True}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "character_clip": ("H3_CHARACTER_CLIP",),
            "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
            "audio_1": ("AUDIO",), "audio_2": ("AUDIO",),
        }}

    RETURN_TYPES = ("H3_CHARACTER", "STRING")
    RETURN_NAMES = ("character", "info")
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def extract(self, name, video_vae, audio_vae, description="", reference_short_edge=512,
                audio_start_seconds=0.0, audio_duration_seconds=5.0, save=True, overwrite=False,
                character_clip=None, image_1=None, image_2=None, image_3=None, audio_1=None, audio_2=None):
        name = safe_name(name)
        path = characters_dir() / f"{name}.safetensors"
        if save and path.exists() and not overwrite:
            raise FileExistsError(f"{path.name} exists. Enable overwrite or change the character name.")
        images = [x for x in (image_1, image_2, image_3) if x is not None]
        audios = [x for x in (audio_1, audio_2) if x is not None]
        refs = []
        provenance = {"producer": "ComfyUI-MiniMaxH3Mod/0.3.0", "preprocessing": "paired-av-v2-float-vision",
                      "video_encoder": _encoder_name(video_vae), "audio_encoder": _encoder_name(audio_vae),
                      "sample_rate": SAMPLE_RATE, "video_fps": FPS,
                      "reference_short_edge": reference_short_edge}
        if character_clip is not None:
            if images or audios:
                raise ValueError("Use a speaking video OR images plus voice clips in one extraction, not both.")
            if not isinstance(character_clip, CharacterClip):
                raise ValueError("Connect Load H3 Character Clip.")
            clip = character_clip
            pixels = resize_visual(clip.frames, reference_short_edge)
            waveform, duration = normalize_audio(clip.audio, 0, clip.duration, min_duration_seconds=5 / FPS)
            if abs(duration - len(pixels) / FPS) > 1 / SAMPLE_RATE:
                raise ValueError("The clip's video and soundtrack have different durations.")
            print(f"[H3 Character] Encoding paired clip ({len(pixels)} frames, {duration:.3f}s).")
            visual = _encode_visual(video_vae, pixels)
            audio = _encode_audio(audio_vae, waveform)
            indices = torch.arange(0, len(pixels), FPS // 2)
            refs.append(CharacterReference("video_audio", visual=visual, audio=audio,
                frames=pixels[indices].detach().cpu().float().contiguous().clone(), timestamps=indices.double() / FPS,
                duration=len(pixels) / FPS))
            provenance.update(source_name=clip.source_name, start_seconds=clip.start_seconds,
                              duration=clip.duration)
        else:
            if not images or not audios:
                raise ValueError("Connect at least one image and one voice AUDIO, or a speaking character_clip.")
            if sum(x.shape[0] if x.ndim == 4 else 0 for x in images) > 9:
                raise ValueError("Use at most 9 still images; use character_clip for a speaking video.")
            for batch in images:
                pixels = resize_visual(batch, reference_short_edge)
                for image in pixels.split(1):
                    refs.append(CharacterReference("image", visual=_encode_visual(video_vae, image),
                                                   frames=image.detach().cpu().float().contiguous().clone()))
            for audio_input in audios:
                waveform, duration = normalize_audio(audio_input, audio_start_seconds, audio_duration_seconds)
                refs.append(CharacterReference("audio", audio=_encode_audio(audio_vae, waveform), duration=duration))
            provenance.update(start_seconds=audio_start_seconds, audio_duration_limit=audio_duration_seconds)
        mod = H3CharacterMod(name, refs, description.strip(), provenance=provenance)
        mod.validate()
        info = mod.summary()
        if save:
            info += "\nSaved: " + mod.save(path, overwrite=overwrite)
        print(f"[H3 Character] {info}")
        return mod, info


class LoadH3Character:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_file": (character_files() or ["(extract a character first)"],),
            "voice_reference": (["all", "first", "second"], {"default": "all",
                "tooltip": "Keep all saved voices or one example. Choose first for a 3-character cast."}),
        }}

    RETURN_TYPES = ("H3_CHARACTER", "STRING")
    RETURN_NAMES = ("character", "info")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, character_file, **kwargs):
        st = character_path(character_file).stat()
        return f"{st.st_mtime_ns}:{st.st_size}:{st.st_ctime_ns}"

    def load(self, character_file, voice_reference="all"):
        mod = select_voice(H3CharacterMod.load(character_path(character_file)), voice_reference)
        return mod, mod.summary()


def _frame_count(width, height, length):
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (width, height, length)):
        raise ValueError("Output dimensions and length must be integers.")
    if width % 32 or height % 32 or not 320 <= width <= 2048 or not 320 <= height <= 2048:
        raise ValueError("Output width and height must be multiples of 32 between 320 and 2048.")
    if not 5 <= length <= 362:
        raise ValueError("Choose an output length between 5 and 362 frames.")
    return 5 + ((length - 5 + 16) // 17) * 17


def _encode_conditioning(clip, prompt, items, blocks, turns, width, height, frame_count):
    import comfy.model_management
    import comfy.nested_tensor
    binding = make_binding(blocks, turns, frame_count)
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is not None and not callable(getattr(tokenizer, "tokenize_with_weights", None)):
        raise ValueError("Connect the MiniMax H3 CLIP/text encoder.")
    tokens = clip.tokenize(prompt, minimax_ref_items=items)
    positive = clip.encode_from_tokens_scheduled(tokens)
    # H3's CLIP emits modality tags. Fail rather than silently use generic CLIP.
    if not positive or any("minimax_token_tags" not in entry[1] for entry in positive):
        raise ValueError("The connected CLIP is not an H3-compatible text encoder (missing minimax_token_tags).")
    positive = [[entry[0], {**entry[1], "minimax_refs": blocks, BINDING_KEY: binding}] for entry in positive]
    video_t = 2 + ((frame_count - 5) // 17) * 5
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, video_t, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, round(frame_count / FPS * 40)), device=device)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    return positive, latent


class H3CharacterDialogueConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",), "character": ("H3_CHARACTER",),
            "scene": ("STRING", {"default": "The character faces the camera in a quiet room.", "multiline": True}),
            "dialogue": ("STRING", {"default": "Hello. It is good to see you again.", "multiline": True}),
            "emotion": ("STRING", {"default": "calm"}),
            "delivery": ("STRING", {"default": "natural conversational speech"}),
            "language": ("STRING", {"default": "English"}),
            "width": ("INT", {"default": 768, "min": 320, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 512, "min": 320, "max": 2048, "step": 32}),
            "length": ("INT", {"default": 124, "min": 5, "max": 362, "step": 17}),
            "max_reference_tokens": ("INT", {"default": 0, "min": 0, "max": 200000, "step": 256,
                "tooltip": "0 = no cap. An exceeded cap gives an error; it never silently drops character data."}),
        }
        optional = {}
        for i in (2, 3):
            optional[f"character_{i}"] = ("H3_CHARACTER",)
            optional[f"dialogue_{i}"] = ("STRING", {"default": "", "multiline": True})
            optional[f"emotion_{i}"] = ("STRING", {"default": "neutral"})
            optional[f"delivery_{i}"] = ("STRING", {"default": "natural conversational speech"})
        optional["turn_schedule"] = ("STRING", {"default": "", "multiline": True,
            "tooltip": "Optional experimental timing. One line per speaking character: character,start_seconds,end_seconds. "
                       "For example: 1,0.3,3.0 then 2,3.3,6.0. Leave blank for untimed generation."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "resolved_prompt", "reference_info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, character, scene, dialogue, emotion="calm", delivery="natural",
               language="English", width=768, height=512, length=124, max_reference_tokens=0, **kwargs):
        frame_count = _frame_count(width, height, length)
        cast = [(character, dialogue, emotion, delivery)]
        if kwargs.get("character_3") is not None and kwargs.get("character_2") is None:
            raise ValueError("Connect character_2 before character_3 so character slot numbers stay stable.")
        for i in (2, 3):
            other = kwargs.get(f"character_{i}")
            if other is not None:
                cast.append((other, kwargs.get(f"dialogue_{i}", ""), kwargs.get(f"emotion_{i}", "neutral"),
                             kwargs.get(f"delivery_{i}", "natural")))
            elif str(kwargs.get(f"dialogue_{i}", "")).strip():
                raise ValueError(f"Dialogue {i} has no connected character_{i}.")
        turns = parse_turns(kwargs.get("turn_schedule", ""), cast, frame_count / FPS)
        prompt, items, blocks, info = build_character_conditioning(cast, scene, language, max_reference_tokens, turns)
        positive, latent = _encode_conditioning(clip, prompt, items, blocks, turns, width, height, frame_count)
        print(f"[H3 Character]\n{info}")
        return positive, latent, prompt, info


class ImportH3RefModsAsCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mods": ("H3_REF_MODS",),
            "image_1": ("IMAGE", {"tooltip": "Matching original image(s), in the visual RefMods' loader order."}),
            "name": ("STRING", {"default": "imported_character"}),
            "description": ("STRING", {"default": "", "multiline": True}),
            "save": ("BOOLEAN", {"default": True}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }, "optional": {"image_2": ("IMAGE",), "image_3": ("IMAGE",)}}

    RETURN_TYPES = ("H3_CHARACTER", "STRING")
    RETURN_NAMES = ("character", "info")
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def convert(self, mods, image_1, name="imported_character", description="", save=True,
                overwrite=False, image_2=None, image_3=None):
        from .character_import import import_refmods
        from .common import resize_ref

        def prepare(image, target):
            # A single upstream encode uses this aspect-preserving preparation.
            # Do not center-crop a different image until it happens to fit.
            return resize_ref(image, min(target))

        path = characters_dir() / f"{safe_name(name)}.safetensors"
        if save and path.exists() and not overwrite:
            raise FileExistsError(f"{path.name} exists. Enable overwrite or change the character name.")
        images = [x for x in (image_1, image_2, image_3) if x is not None]
        mod = import_refmods(mods, images, name, description, resize_image=prepare)
        info = mod.summary() + "\nImported unchanged latents; source-image identity is supplied by the user."
        if save:
            info += "\nSaved: " + mod.save(path, overwrite=overwrite)
        print(f"[H3 Character] {info}")
        return mod, info


class H3CharacterReferenceConditioning:
    """Use a full prompt without rewriting it; optional controlled ablations."""
    @classmethod
    def INPUT_TYPES(cls):
        base = H3CharacterDialogueConditioning.INPUT_TYPES()["required"]
        required = {name: base[name] for name in
                    ("clip", "character", "width", "height", "length", "max_reference_tokens")}
        required["prompt"] = ("STRING", {"default": "", "multiline": True,
            "tooltip": "Your complete H3 reference prompt. Subject numbers follow character input slots; check reference_info for asset labels."})
        return {"required": required, "optional": {
            "character_2": ("H3_CHARACTER",), "character_3": ("H3_CHARACTER",),
            "turn_schedule": ("STRING", {"default": "", "multiline": True,
                "tooltip": "Optional attention timing: character,start_seconds,end_seconds. Match the speaking windows in your prompt; the prompt is not rewritten."}),
            "reference_presentation": (["native", "latent_only"], {"default": "native", "advanced": True,
                "tooltip": "Diagnostic only: latent_only omits the native reference presentation but keeps stored latents and your prompt."}),
            "av_layout": (["paired", "separate"], {"default": "paired", "advanced": True,
                "tooltip": "Diagnostic only: separate splits speaking clips into independent reference blocks. The source tensors and prompt stay unchanged."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "resolved_prompt", "reference_info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, character, prompt, width=768, height=512, length=124,
               max_reference_tokens=0, character_2=None, character_3=None, turn_schedule="",
               reference_presentation="native", av_layout="paired"):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter your complete H3 prompt.")
        if reference_presentation not in ("native", "latent_only"):
            raise ValueError("Reference presentation must be native or latent_only.")
        if character_3 is not None and character_2 is None:
            raise ValueError("Connect character_2 before character_3 so character slot numbers stay stable.")
        characters = [mod for mod in (character, character_2, character_3) if mod is not None]
        frame_count = _frame_count(width, height, length)
        assembled = assemble_references(characters, max_reference_tokens, av_layout)
        # Full prompts can contain silent characters. Validate the explicit
        # windows without pretending to parse natural-language dialogue.
        turns = parse_turns(turn_schedule, [(c, "", "", "") for c in characters], frame_count / FPS,
                            require_dialogue_coverage=False)
        items = assembled.items if reference_presentation == "native" else []
        report = assembled.audit(characters, prompt, width, height, frame_count, turns, reference_presentation)
        positive, latent = _encode_conditioning(clip, prompt, items, assembled.blocks, turns,
                                               width, height, frame_count)
        for entry in positive:
            entry[1]["h3_character_audit"] = report
        info = json.dumps(report, indent=2, allow_nan=False)
        mapping = "; ".join(f"{mod.name} -> <Subject {i}>: " + ", ".join(labels["visual"] + labels["audio"])
                            for i, (mod, labels) in enumerate(zip(characters, assembled.labels), 1))
        print(f"[H3 Character] {reference_presentation}/{av_layout}; {mapping}\n"
              f"[H3 Character] case_fingerprint={report['case_fingerprint']}")
        return positive, latent, prompt, info


NODE_CLASS_MAPPINGS = {
    "LoadH3CharacterClip": LoadH3CharacterClip,
    "ExtractH3Character": ExtractH3Character,
    "LoadH3Character": LoadH3Character,
    "H3CharacterDialogueConditioning": H3CharacterDialogueConditioning,
    "H3CharacterVoiceBinding": H3CharacterVoiceBinding,
    "ImportH3RefModsAsCharacter": ImportH3RefModsAsCharacter,
    "H3CharacterReferenceConditioning": H3CharacterReferenceConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadH3CharacterClip": "Load H3 Character Clip",
    "ExtractH3Character": "Extract H3 Character",
    "LoadH3Character": "Load H3 Character",
    "H3CharacterDialogueConditioning": "H3 Character Dialogue Conditioning",
    "H3CharacterVoiceBinding": "H3 Character Voice Binding (Experimental)",
    "ImportH3RefModsAsCharacter": "Import H3 RefMods as Character",
    "H3CharacterReferenceConditioning": "H3 Character Reference Conditioning",
}
