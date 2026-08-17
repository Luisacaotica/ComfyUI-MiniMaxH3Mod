"""
debug_grid.py — 1024x1024 curve-graph visualization for Apply H3 RefMod.

Renders the per-frame strength envelope (direction/shape/value) across the
full canvas, with the "concept zone" shaded where the concept shows up.  It
is both the Apply's optional debug IMAGE output and the saved graph-preset
PNG (graph embedded in the image's tEXt metadata).

Pure PIL; ``pil_to_tensor`` converts the result to a torch IMAGE tensor.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from .core import CURVE_DIRECTIONS, CURVE_SHAPES, curve_strengths

SIZE = 1024

BG = (18, 20, 24)
TITLE = (232, 230, 227)
TEXT = (207, 203, 196)
DIM = (107, 111, 118)
GRID_LINE = (42, 46, 54)
ACCENT = (255, 158, 61)
ACCENT_FILL = (58, 34, 15)
ACCENT_DIM = (122, 69, 21)

_CURVE_SAMPLES = 160

# where the concept shows up, per direction (mirror of the envelope)
_ZONES = {
    "concept_at_start": ((0.0, 1.0 / 3.0),),
    "concept_at_middle": ((1.0 / 3.0, 2.0 / 3.0),),
    "concept_at_end": ((2.0 / 3.0, 1.0),),
    "concept_at_ends": ((0.0, 1.0 / 3.0), (2.0 / 3.0, 1.0)),
}


# ── fonts ────────────────────────────────────────────────────────────────────

_FONT_CACHE: Dict[Tuple[int, bool, bool], ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False, mono: bool = False):
    key = (size, bold, mono)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = []
    if os.name == "nt":
        base = "C:/Windows/Fonts/"
        name = "consola.ttf" if mono else ("arialbd.ttf" if bold else "arial.ttf")
        candidates = [base + name]
    else:
        if mono:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/System/Library/Fonts/Menlo.ttc",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                f = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = f
                return f
            except OSError:
                pass
    try:
        f = ImageFont.load_default(size=size)
    except Exception:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    l, _t, r, _b = draw.textbbox((0, 0), text, font=font)
    return r - l


# ── the curve graph ──────────────────────────────────────────────────────────

def render_debug_grid(spec, preset_name: str = "") -> Image.Image:
    """Render the 1024x1024 curve graph for an Apply configuration.

    ``spec`` is the resolved (direction, shape, value) curve tuple and
    ``preset_name`` the active graph preset (shown when one is loaded).
    """
    img = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)

    draw.text((32, 26), "curve graph", font=_font(28, bold=True), fill=TITLE)
    if preset_name:
        draw.text((SIZE - 32, 34), f"preset: {preset_name}",
                  font=_font(20), fill=DIM, anchor="rs")

    x0, y0, x1, y1 = 90, 120, SIZE - 10, 940
    direction = spec[0]

    # concept zone shading (where the concept shows up in the output)
    for start, end in _ZONES.get(direction, ()):
        zx0 = x0 + start * (x1 - x0)
        zx1 = x0 + end * (x1 - x0)
        draw.rectangle([zx0, y0, zx1, y1], fill=ACCENT_FILL)
        draw.text((zx0 + 8, y1 + 8), "concept zone", font=_font(20), fill=ACCENT_DIM)

    # grid + axis labels
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y1 - f * (y1 - y0)
        draw.line([x0, gy, x1, gy], fill=GRID_LINE, width=2)
        draw.text((16, gy - 12), f"{1.0 - f:.2f}", font=_font(20), fill=DIM)
    for fx, label in ((0.0, "0"), (0.25, "0.25"), (0.5, "0.5"),
                      (0.75, "0.75"), (1.0, "1")):
        gx = x0 + fx * (x1 - x0)
        draw.line([gx, y0, gx, y1], fill=GRID_LINE, width=2)
        draw.text((gx, y1 + 8), label, font=_font(20), fill=DIM, anchor="ma")
    draw.text((x1 - 6, y1 + 8), "video →", font=_font(20), fill=DIM, anchor="ra")

    # the envelope
    strengths = curve_strengths(spec, _CURVE_SAMPLES)
    if strengths is None:
        strengths = [1.0] * _CURVE_SAMPLES
    pts = [(x0 + i * (x1 - x0) / (_CURVE_SAMPLES - 1),
            y1 - s * (y1 - y0)) for i, s in enumerate(strengths)]
    draw.polygon([(x0, y1)] + pts + [(x1, y1)], fill=ACCENT_FILL)
    draw.line(pts, fill=ACCENT, width=6)

    # peak marker
    peak = max(range(_CURVE_SAMPLES), key=lambda i: strengths[i])
    px, py = pts[peak]
    draw.line([px, y0, px, y1], fill=ACCENT_DIM, width=2)
    draw.ellipse([px - 9, py - 9, px + 9, py + 9], fill=ACCENT)

    # spec readout on a dark backdrop
    lines = [f"direction: {direction}", f"shape:     {spec[1]}",
             f"value:     {float(spec[2]):.2f}"]
    font = _font(24, mono=True)
    tw = max(_text_w(draw, ln, font) for ln in lines)
    draw.rectangle([x0 + 18, y0 + 18, x0 + 18 + tw + 24, y0 + 18 + 3 * 34 + 12],
                   fill=(14, 17, 22))
    for i, ln in enumerate(lines):
        draw.text((x0 + 30, y0 + 28 + i * 34), ln, font=font, fill=TEXT)

    return img


# ── graph presets (PNG metadata) ────────────────────────────────────────────

# The graph spec is embedded in the preset PNG's tEXt chunk, so a saved curve
# graph IS the preset: share the image, and the curve comes with it.
GRAPH_META_KEY = "graph"


def graph_pnginfo(spec):
    """PIL pnginfo carrying the (direction, shape, value) spec as tEXt."""
    info = PngImagePlugin.PngInfo()
    info.add_text(GRAPH_META_KEY, json.dumps({
        "direction": spec[0], "shape": spec[1], "value": float(spec[2])}))
    return info


def read_graph_meta(path: str) -> Optional[tuple]:
    """Read a (direction, shape, value) spec from a PNG's tEXt metadata.

    Returns None for a missing file, a missing/empty chunk, or an invalid
    spec (unknown direction/shape, bad value) so callers can fall back.
    """
    try:
        with Image.open(path) as img:
            raw = img.text.get(GRAPH_META_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    direction, shape = data.get("direction"), data.get("shape")
    if direction not in CURVE_DIRECTIONS or shape not in CURVE_SHAPES:
        return None
    try:
        value = float(data.get("value", 1.0))
    except (TypeError, ValueError):
        return None
    return (direction, shape, value)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a rendered PIL graph to a [1, 1024, 1024, 3] float32 tensor."""
    arr = np.asarray(img).copy()
    return torch.from_numpy(arr).float().unsqueeze(0) / 255.0  # [1, 1024, 1024, 3]
