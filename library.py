"""Metadata-only library endpoint; no tensor allocation or VAE loading."""

from aiohttp import web


def library_entries(names, find_path, read_meta):
    entries = []
    for name in names():
        try:
            path = find_path(name)
            meta = read_meta(path)
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        kind = meta.get("kind")
        try:
            t = int(meta.get("latent_t", 0))
            h, w = int(meta.get("latent_h", 0)), int(meta.get("latent_w", 0))
        except (TypeError, ValueError):
            continue
        tokens = t * 2 if kind == "audio" else t * (h // 2) * (w // 2)
        entries.append({"name": name, "kind": kind, "path": path + ".safetensors",
                        "concept": meta.get("concept_type", "generic"),
                        "description": meta.get("description", ""),
                        "tokens": tokens or None, "shape": [t, h, w],
                        "config": meta.get("refmod_config", "")})
    return entries


def register_routes(server, names, find_path, read_meta):
    @server.routes.get("/refmods/library")
    async def library(request):
        return web.json_response(library_entries(names, find_path, read_meta))
