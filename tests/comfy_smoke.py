"""No weights required: python tests/comfy_smoke.py /path/to/ComfyUI

Run with ComfyUI's Python/dependencies. Uses real node registration, H3
presentation tokenizer, packed layout, model conditioning and AV latent code.
Only the expensive text-encoder forward is replaced by a small test double.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(COMFY))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
import nodes
from comfy_extras import nodes_minimax_h3 as h3, nodes_audio, nodes_video, nodes_custom_sampler
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from comfy.ldm.minimax.model import PackedLayout
from comfy.model_base import MiniMaxH3

spec = importlib.util.spec_from_file_location("h3_smoke_pack", ROOT / "__init__.py",
                                             submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
from h3_smoke_pack.character_core import CharacterReference, H3CharacterMod
from h3_smoke_pack.character_nodes import H3CharacterDialogueConditioning, H3CharacterReferenceConditioning
from h3_smoke_pack.character_binding import H3CharacterVoiceBinding, BINDING_KEY, WRAPPER_KEY


class TextTokenizer:
    def __init__(self):
        self.segments = []

    def tokenize_with_weights(self, text, **kwargs):
        self.segments.append(text)
        return [[(100, 1.0)]]


class Clip:
    def __init__(self):
        self.tokenizer = MiniMaxH3Tokenizer.__new__(MiniMaxH3Tokenizer)
        self.tokenizer.qwen3vl_32b = TextTokenizer()

    def tokenize(self, prompt, **kwargs):
        self.last_tokens = self.tokenizer.tokenize_with_weights(prompt, **kwargs)
        return self.last_tokens

    def encode_from_tokens_scheduled(self, tokens):
        assert "qwen3vl_32b" in tokens
        return [[torch.zeros(1, 8, 16), {"minimax_token_tags": torch.zeros(8, dtype=torch.long)}]]


class RealTokenizerClip(Clip):
    def __init__(self):
        # Uses the Qwen vocabulary shipped with ComfyUI. Only the expensive
        # text-model forward remains a double; no network/model download.
        self.tokenizer = MiniMaxH3Tokenizer()


def check_conditioning():
    image = CharacterReference("image", visual=torch.ones(1, 24, 1, 20, 20),
                               frames=torch.zeros(1, 320, 320, 3, dtype=torch.uint8))
    audio = CharacterReference("audio", audio=torch.ones(1, 32, 2, 40), duration=1)
    pair = CharacterReference("video_audio", visual=torch.ones(1, 24, 7, 20, 20),
        audio=torch.ones(1, 32, 2, 37), frames=torch.zeros(2, 320, 320, 3, dtype=torch.uint8),
        timestamps=torch.tensor([0., 0.5], dtype=torch.float64), duration=22 / 24)
    separate, paired = H3CharacterMod("A", [image, audio]), H3CharacterMod("B", [pair])
    # Exercise a disk round trip before invoking the actual Comfy node.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "B.safetensors"
        paired.save(path)
        paired = H3CharacterMod.load(path)
    clip = Clip()
    positive, latent, prompt, info = H3CharacterDialogueConditioning().encode(
        clip, paired, "Two people take turns speaking.", "Hello, A.", width=640, height=320,
        length=125, character_2=separate, dialogue_2="Hello, B.")
    native, _ = h3._empty_av_latent(640, 320, 125)
    for ours, theirs in zip(latent["samples"].tensors, native["samples"].tensors):
        assert torch.equal(ours, theirs)
    blocks = positive[0][1]["minimax_refs"]
    assert [b["kind"] for b in blocks] == ["image", "video_audio", "audio"]
    text = clip.tokenizer.qwen3vl_32b.segments
    assert text[:4] == ["<Picture 1>: ", "<Audio 1>: ", "<Video 1>: ", "<0.2 seconds>"]
    assert text[4] == "<Audio 2>: "
    assert "B -> <Subject 1>: <Video 1>; voice <Audio 1>" in info
    video, sound = latent["samples"].tensors
    layout = PackedLayout(8, video.shape[2], video.shape[3], video.shape[4], sound.shape[-1], refs=blocks)
    rows = sum(end - start for start, end, kind in layout.segments if kind.startswith("ref_"))
    assert rows == separate.token_count + paired.token_count
    # The real model wrapper must also accept the metadata and produce its payload.
    model = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.concat_keys = ()
    model.latent_shapes = None
    model.get_dtype_inference = lambda: torch.float32
    model.diffusion_model = torch.nn.Identity()
    model.diffusion_model.preprocess_text_embeds = lambda x: x
    extra = model.extra_conds(device=torch.device("cpu"), noise=latent["samples"],
                              cross_attn=positive[0][0], **positive[0][1])
    assert "minimax_payload" in extra
    # Reusing CLIP must not accumulate references or stale assignments.
    again, _, _, info = H3CharacterDialogueConditioning().encode(
        clip, separate, "One person.", "Different words.", width=640, height=320)
    assert len(again[0][1]["minimax_refs"]) == 2
    assert "A -> <Subject 1>: <Picture 1>; voice <Audio 1>" in info
    print("PASS: saved assets -> native tokenizer -> conditioning -> packed layout; AV geometry; reuse")


def check_workflows():
    registry = dict(nodes.NODE_CLASS_MAPPINGS)
    registry.update(pack.NODE_CLASS_MAPPINGS)
    for module in (h3, nodes_audio, nodes_video, nodes_custom_sampler):
        for name in dir(module):
            cls = getattr(module, name)
            if isinstance(cls, type) and cls.__module__ == module.__name__ and hasattr(cls, "define_schema"):
                registry[cls.define_schema().node_id] = cls
    for name in ("LoadH3CharacterClip", "ExtractH3Character", "LoadH3Character", "H3CharacterDialogueConditioning"):
        assert registry[name].INPUT_TYPES()
    count = 0
    for path in sorted((ROOT / "examples" / "characters").glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        assert len(by_id) == len(graph["nodes"])
        for node in graph["nodes"]:
            if node["type"] == "Note":
                continue
            cls = registry[node["type"]]
            schema = cls.INPUT_TYPES()
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            for port in node["inputs"]:
                assert port["name"] in inputs, (path, node["type"], port)
                assert port["type"] == inputs[port["name"]][0], (path, port)
            assert [p["type"] for p in node["outputs"]] == list(cls.RETURN_TYPES), (path, node["type"])
            supplied = {p["name"] for p in node["inputs"] if p["link"] is not None}
            supplied.update(node["properties"].get("test_widget_names", []))
            assert set(schema.get("required", {})) <= supplied, (path, node["type"], supplied)
            # Widget names/order are included for repeatable schema checks.
            names = node["properties"].get("test_widget_names", [])
            assert len(names) == len(node["widgets_values"]), (path, node["type"])
            for name, value in zip(names, node["widgets_values"]):
                if name in ("control_after_generate", "format.codec"):
                    continue
                assert name in inputs, (path, name)
                entry = inputs[name]
                options = entry[0] if isinstance(entry[0], list) else entry[1].get("options", []) if len(entry) > 1 else []
                # Files are selected on the user's installation.
                if options and name not in ("image", "audio", "vae_name", "clip_name", "unet_name", "character_file"):
                    allowed = [x["key"] if isinstance(x, dict) else x for x in options]
                    assert value in allowed, (path, name, value, allowed)
        for link_id, source, out_slot, target, in_slot, kind in graph["links"]:
            output, input_ = by_id[source]["outputs"][out_slot], by_id[target]["inputs"][in_slot]
            assert output["type"] == input_["type"] == kind
            assert input_["link"] == link_id and link_id in output["links"]
        for node in graph["nodes"]:
            for port in node.get("inputs", []):
                assert port["link"] is None or port["link"] in links
        count += 1
        print(f"PASS: node schemas and graph links: {path.name}")
    assert count >= 4, "Missing character example workflows"


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b, (a, b)


def check_native_reference_parity():
    """Compare actual native node/tokenizer/layout against a saved mixed cast.

    VAE doubles depend on their inputs; this checks preprocessing/routing, not
    codec quality. Both paths use exactly the same prepared source intervals.
    """
    import folder_paths
    import torchaudio
    from h3_smoke_pack.character_nodes import ExtractH3Character, ImportH3RefModsAsCharacter
    from h3_smoke_pack.character_media import resize_visual, normalize_audio, CharacterClip
    from h3_smoke_pack.core import H3RefMod

    class VideoVAE:
        def encode(self, pixels):
            t = 1 if len(pixels) == 1 else 2 + (len(pixels) - 5) // 17 * 5
            return pixels.contiguous().mean().expand(1, 24, t, pixels.shape[1] // 16, pixels.shape[2] // 16).clone()

    class AudioVAE:
        audio_sample_rate = 32000
        def encode(self, pixels):
            assert pixels.shape[-1] == 2
            return pixels.mean().expand(1, 32, 2, (pixels.shape[1] + 799) // 800).clone()

    torch.manual_seed(127)
    raw_image = torch.rand(1, 480, 640, 3)
    prepared_image = resize_visual(raw_image, 320)
    assert torch.equal(prepared_image, h3._resize(raw_image, prepared_image.shape[2], 320, "disabled"))
    raw_audio = {"waveform": torch.rand(1, 2, 44100) * 0.2, "sample_rate": 44100}
    wave, _ = normalize_audio(raw_audio, 0, 1)
    assert torch.equal(wave, torchaudio.functional.resample(raw_audio["waveform"], 44100, 32000))
    audio = {"waveform": wave, "sample_rate": 32000}
    frames = torch.randint(0, 256, (22, 320, 320, 3), dtype=torch.uint8)
    soundtrack = {"waveform": torch.rand(1, 2, round(22 / 24 * 32000)) * 0.2, "sample_rate": 32000}
    clip_source = CharacterClip(frames, soundtrack, 0, 22 / 24, "test.mkv")
    prompt = "Full prompt preserved exactly. <Audio 1> guides Subject 2; <Audio 2> guides Subject 1."
    vvae, avae = VideoVAE(), AudioVAE()
    with tempfile.TemporaryDirectory() as directory, patch.object(folder_paths, "models_dir", directory), \
            patch.object(folder_paths, "get_folder_paths", return_value=[]):
        a, _ = ExtractH3Character().extract("A", vvae, avae, reference_short_edge=320,
            audio_duration_seconds=1, image_1=raw_image, audio_1=raw_audio, save=False)
        b, _ = ExtractH3Character().extract("B", vvae, avae, reference_short_edge=320,
                                           character_clip=clip_source, save=False)
        original = [a, b]
        loaded = []
        for mod in original:
            path = Path(directory) / (mod.name + ".safetensors")
            mod.save(path)
            loaded.append(H3CharacterMod.load(path))
        native_clip, cached_clip = RealTokenizerClip(), RealTokenizerClip()
        native, native_latent = h3.MiniMaxH3ReferenceToVideo.execute(
            native_clip, prompt, 640, 320, 124, ref_image_size="max", vae=vvae, audio_vae=avae,
            ref_images={"ref_image_1": prepared_image},
            ref_videos={"ref_video_1": resize_visual(frames, 320)},
            ref_video_audios={"ref_video_audio_1": soundtrack}, ref_audios={"ref_audio_1": audio})
        cached, cached_latent, resolved, info = H3CharacterReferenceConditioning().encode(
            cached_clip, loaded[0], prompt, character_2=loaded[1], width=640, height=320)
        assert resolved == prompt
        assert_nested_equal(native_clip.last_tokens, cached_clip.last_tokens)
        nrefs, crefs = native[0][1]["minimax_refs"], cached[0][1]["minimax_refs"]
        assert len(nrefs) == len(crefs)
        for nref, cref in zip(nrefs, crefs):
            assert_nested_equal(nref, {key: cref[key] for key in nref})
        assert_nested_equal(native_latent["samples"].tensors, cached_latent["samples"].tensors)
        nlayout = PackedLayout(8, 37, 20, 40, 207, refs=nrefs)
        clayout = PackedLayout(8, 37, 20, 40, 207, refs=crefs)
        assert nlayout.segments == clayout.segments
        assert torch.equal(nlayout.position_ids, clayout.position_ids)
        report = json.loads(info)
        # Same full prompt and source hashes in every diagnostic variant.
        for presentation, layout in (("latent_only", "paired"), ("native", "separate")):
            result, other_latent, text, other_info = H3CharacterReferenceConditioning().encode(
                Clip(), loaded[0], prompt, character_2=loaded[1], width=640, height=320,
                reference_presentation=presentation, av_layout=layout)
            assert text == prompt
            assert json.loads(other_info)["case_fingerprint"] == report["case_fingerprint"]
            assert_nested_equal(other_latent["samples"].tensors, cached_latent["samples"].tensors)
            assert [r["kind"] for r in result[0][1]["minimax_refs"]] == (
                ["image", "video_audio", "audio"] if layout == "paired" else ["image", "audio", "video", "audio"])
        # The new importer invokes upstream resizing and keeps actual saved latents.
        bundle = [(H3RefMod("image", "image", a.references[0].visual, latent_h=20, latent_w=26, mode="encode"), 1),
                  (H3RefMod("voice", "audio", a.references[1].audio, mode="encode"), 1)]
        imported, _ = ImportH3RefModsAsCharacter().convert(bundle, raw_image, save=False)
        assert torch.equal(imported.references[0].frames, prepared_image)
        assert torch.equal(imported.references[1].audio, a.references[1].audio)
    print("PASS: native versus saved mixed-reference presentation, tensors, positions and AV output; full prompt and diagnostic isolation; upstream import")


def check_native_binding_forward(cast_count):
    """Run the actual H3 transformer with tiny random weights; no quality claims."""
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP

    torch.manual_seed(37)
    tiny = MiniMaxH3Model(hidden_size=32, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64, text_dim=16,
        timestep_input_dim=8, time_embed_hidden_size=32, time_embed_dim=16,
        rope_inv_freq_len=2, dtype=torch.float32, device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for name, parameter in tiny.named_parameters():
            parameter.uniform_(-0.1, 0.1)
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
        tiny.rope.inv_freq.copy_(torch.tensor([1., 0.1]))
    tiny.eval().requires_grad_(False)
    base = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(base)
    base.diffusion_model = tiny
    patcher = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    characters = []
    for index in range(1, cast_count + 1):
        pair = CharacterReference("video_audio", visual=torch.full((1, 24, 2, 4, 4), index * 0.1),
            audio=torch.full((1, 32, 2, 9), index * 0.2),
            frames=torch.zeros(1, 320, 320, 3, dtype=torch.uint8),
            timestamps=torch.tensor([0.]), duration=5 / 24)
        characters.append(H3CharacterMod(str(index), [pair]))
    third = {"character_3": characters[2], "dialogue_3": "Good."} if cast_count == 3 else {}
    frame_count = 39 if cast_count == 3 else 22
    positive, _, _, _ = H3CharacterDialogueConditioning().encode(
        Clip(), characters[0], "Two people.", "Hi.", character_2=characters[1], dialogue_2="Hey.",
        width=320, height=320, length=frame_count,
        turn_schedule="1,0,0.4\n2,0.5,0.9" + ("\n3,1,1.4" if third else ""), **third)
    refs = positive[0][1]["minimax_refs"]
    payload = {"refs": refs, "cond_video_latents": [r["latent"] for r in refs],
               "cond_audio_latents": [r["audio_latent"] for r in refs], "seed": 123}
    x = [torch.randn(1, 24, 2 + (frame_count - 5) // 17 * 5, 4, 4),
         torch.randn(1, 32, 2, round(frame_count / 24 * 40))]
    context = torch.randn(1, 3, 32)
    before = {name: p.detach().clone() for name, p in tiny.named_parameters()}
    control = H3CharacterVoiceBinding()
    assert control.apply(patcher, positive, mode="off")[0] is patcher
    assert control.apply(patcher, positive, mode="scheduled", pair_bias=0, voice_bias=0)[0] is patcher
    for mode in ("pair_only", "scheduled"):
        patched, _ = control.apply(patcher, positive, mode=mode, pair_bias=0.5, voice_bias=0.8, query_chunk=7)
        wrappers = patched.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        assert len(wrappers) == 1 and not patcher.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        options = {"wrappers": {WrappersMP.DIFFUSION_MODEL: {WRAPPER_KEY: wrappers}}}
        with torch.inference_mode():
            for sigma in (800., 350.):
                baseline = tiny(x, torch.tensor([sigma]), context, minimax_payload=payload)
                changed = tiny(x, torch.tensor([sigma]), context, transformer_options=options, minimax_payload=payload)
                assert all(torch.isfinite(z).all() for z in changed)
                assert all(a.shape == b.shape for a, b in zip(baseline, changed))
                assert not torch.allclose(baseline[1], changed[1], atol=1e-8)
                repeated = tiny(x, torch.tensor([sigma]), context, minimax_payload=payload)
                assert all(torch.equal(a, b) for a, b in zip(baseline, repeated))
        assert "optimized_attention_override" not in options
        # Fail visibly if another patch owns attention, or if the model skips our hook.
        wrapper = wrappers[0]
        for extra in ({"optimized_attention_override": object()}, {"patches_replace": {"dit": {0: object()}}}, {}):
            try:
                wrapper(lambda *a, **k: x, x, torch.tensor([800.]), context, extra, minimax_payload=payload)
            except ValueError:
                pass
            else:
                raise AssertionError("A conflicting/bypassed attention hook must not silently succeed")
        try:
            control.apply(patched, positive)
        except ValueError:
            pass
        else:
            raise AssertionError("Chained binding nodes must be rejected")
    assert all(torch.equal(before[name], p) for name, p in tiny.named_parameters())
    print(f"PASS: {cast_count}-character tiny H3 forwards, both modes, two sigmas; exact off; no weight/global-option mutation; hook guards")


check_conditioning()
check_native_reference_parity()
check_workflows()
check_native_binding_forward(2)
check_native_binding_forward(3)
print(f"PASS: {len(pack.NODE_CLASS_MAPPINGS)} existing/new node registrations")
