"""Version 5 containers; members remain ordinary independent H3 references."""

import json
import math
import os
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

from .core import H3RefMod, META_KEY, read_refmod_meta


def members(meta):
    if not isinstance(meta, dict) or meta.get("_format_version") != 5 or meta.get("kind") != "bundle":
        raise ValueError("Unsupported RefMod bundle format.")
    refs = meta.get("members")
    if not isinstance(refs, list) or not refs or len(refs) > 256:
        raise ValueError("A RefMod bundle must contain 1–256 references.")
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("kind") not in ("image", "video", "audio"):
            raise ValueError("Invalid RefMod bundle member.")
    return refs


def write_bundle(path, metadata, tensors, extra_metadata=None):
    members(metadata)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".refmod-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        header = dict(extra_metadata or {})
        header[META_KEY] = json.dumps(metadata)
        save_file(tensors, temporary, metadata=header)
        os.replace(temporary, path + ".safetensors")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path + ".safetensors"


def save_bundle(path, name, mods):
    # Copies affect runtime, not stored content; strengths are not baked in.
    unique = list({id(mod): mod for mod, _ in mods}.values())
    metadata = {"_format_version": 5, "kind": "bundle", "name": name,
                "members": [mod.metadata() for mod in unique]}
    tensors = {f"ref_{i}": mod.latent.detach().cpu().contiguous().clone()
               for i, mod in enumerate(unique)}
    return write_bundle(path, metadata, tensors)


def load_bundle(path, selection="All", visual_strength=1.0, audio_strength=1.0):
    if selection not in ("All", "Visual", "Audio"):
        raise ValueError("Components must be All, Visual or Audio.")
    if any(not math.isfinite(weight) or not 0 <= weight <= 1
           for weight in (visual_strength, audio_strength)):
        raise ValueError("Component strengths must be finite numbers between 0 and 1.")
    metadata = read_refmod_meta(path)
    refs = members(metadata)
    loaded = []
    with safe_open(path + ".safetensors", framework="pt", device="cpu") as file:
        for i, meta in enumerate(refs):
            audio = meta["kind"] == "audio"
            strength = audio_strength if audio else visual_strength
            if strength <= 0 or (selection == "Visual" and audio) or (selection == "Audio" and not audio):
                continue
            latent = file.get_tensor(f"ref_{i}").clone()
            if audio:
                valid = (latent.ndim == 4 and tuple(latent.shape[:3]) == (1, 32, 2)
                         and latent.shape[-1] > 0)
            else:
                valid = (latent.ndim == 5 and tuple(latent.shape[:2]) == (1, 24)
                         and all(n > 0 for n in latent.shape[2:])
                         and all(n % 2 == 0 for n in latent.shape[-2:]))
                valid = valid and (meta["kind"] != "image" or latent.shape[2] == 1)
            if not valid:
                raise ValueError(f"Invalid tensor layout for bundle member {i}.")
            mod = H3RefMod.from_metadata(meta, latent, path, bundle_index=i)
            if not audio:
                mod.latent_t, mod.latent_h, mod.latent_w = latent.shape[2:]
            loaded.append((mod, strength))
    return loaded


def update_member(mod):
    # Read the current container: updating only Visual must retain Audio, and
    # unknown tensors/metadata must survive a metadata-only Config operation.
    with safe_open(mod.path + ".safetensors", framework="pt", device="cpu") as file:
        header = file.metadata()
        metadata = json.loads(header[META_KEY])
        refs = members(metadata)
        if not 0 <= mod.bundle_index < len(refs) or refs[mod.bundle_index].get("name") != mod.name:
            raise ValueError("Bundle changed since loading. Reload before updating its config.")
        tensors = {key: file.get_tensor(key).clone() for key in file.keys()}
    refs[mod.bundle_index] = {**refs[mod.bundle_index], **mod.metadata()}
    tensors[f"ref_{mod.bundle_index}"] = mod.latent.detach().cpu().contiguous().clone()
    return write_bundle(mod.path, metadata, tensors, header)
