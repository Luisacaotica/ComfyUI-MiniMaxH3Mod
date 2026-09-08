"""Combined RefMods through the real, unmodified official H3 conditioning node."""
import copy
import importlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
from comfy.ldm.minimax.model import PackedLayout
from comfy.model_base import MiniMaxH3

ROOT = Path(__file__).resolve().parents[1]
P = f"custom_nodes.{ROOT.name}"
N = importlib.import_module(P + ".nodes")
C = importlib.import_module(P + ".character_core")
R = importlib.import_module(P + ".refmod_native")
E = importlib.import_module(P + ".refmod_extract_audio")
M = importlib.import_module(P + ".character_media")


class VideoVAE:
    audio_sample_rate = 44100  # Generic ComfyUI wrapper field, also present on visual codecs.
    latent_channels, latent_dim, output_channels = 24, 3, 3

    def encode(self, pixels):
        t = 1 if len(pixels) == 1 else 2 + (len(pixels) - 5) // 17 * 5
        return pixels.mean().expand(1, 24, t, pixels.shape[1] // 16, pixels.shape[2] // 16).clone()


class AudioVAE:
    audio_sample_rate = 32000
    latent_channels, latent_dim, output_channels = 32, 2, 2

    def encode(self, samples):
        return samples.mean().expand(1, 32, 2, math.ceil(samples.shape[1] / 800)).clone()


class Clip:
    def __init__(self):
        self.tokenizer = MiniMaxH3Tokenizer()  # Real local Qwen vocabulary and H3 presentation.
        self.last_tokens = self.prompt = self.items = None

    def clone(self):
        return copy.copy(self)

    def tokenize(self, prompt, **kwargs):
        self.prompt, self.items = prompt, kwargs.get("minimax_ref_items")
        self.last_tokens = self.tokenizer.tokenize_with_weights(prompt, **kwargs)
        return self.last_tokens

    def encode_from_tokens_scheduled(self, tokens):
        # Only the large learned text-model forward is replaced.
        return [[torch.zeros(1, 8, 16), {"minimax_token_tags": torch.zeros(8, dtype=torch.long)}]]


def audio(seconds, value=0.1):
    return {"waveform": torch.full((1, 2, round(seconds * 32000)), value), "sample_rate": 32000}


def profile(name, durations=(4.05, 5.0), images=3):
    refs = [C.CharacterReference("image", visual=torch.full((1, 24, 1, 20, 20), 0.5),
                                frames=torch.full((1, 320, 320, 3), 0.5)) for _ in range(images)]
    refs += [C.CharacterReference("audio", audio=torch.full((1, 32, 2, math.ceil(t * 40)), 0.1 + i / 10),
                                  duration=t) for i, t in enumerate(durations)]
    return C.H3CharacterMod(name, refs)


def equal_values(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            equal_values(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for av, bv in zip(a, b):
            equal_values(test, av, bv)
    else:
        test.assertEqual(a, b)


class NativeRefModTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_clip = Clip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scope = patch.object(N, "_mod_search_dirs", return_value=[str(self.root)])
        scope.start()
        self.addCleanup(scope.stop)
        N._MOD_CACHE.clear()
        N._MOD_LIST_CACHE_KEY = None

    def load(self, **kwargs):
        return N.MiniMaxH3RefModsLoader().load(clip=self.base_clip, **kwargs)

    def generate_conditioning(self, mods, clip, prompt="Custom full Ref2VA prompt: <d>[English] New words.</d>", **kwargs):
        official = MiniMaxH3ReferenceToVideo.execute(clip, prompt, 320, 320, 124, **kwargs)
        applied = N.MiniMaxH3RefModApply.execute(official[0], mods, scramble_seed=-1)
        positive = applied[0]
        video, sound = official[1]["samples"].tensors
        layout = PackedLayout(8, video.shape[2], video.shape[3], video.shape[4], sound.shape[-1],
                              refs=positive[0][1]["minimax_refs"])
        expected = sum(m.token_count for m, s in mods if s > 0)
        actual = sum(end - start for start, end, kind in layout.segments if kind.startswith("ref_"))
        self.assertGreaterEqual(actual, expected)  # Additional ordinary refs are allowed.
        model = MiniMaxH3.__new__(MiniMaxH3)
        torch.nn.Module.__init__(model)
        model.concat_keys, model.latent_shapes = (), None
        model.get_dtype_inference = lambda: torch.float32
        model.diffusion_model = torch.nn.Identity()
        model.diffusion_model.preprocess_text_embeds = lambda x: x
        extra = model.extra_conds(device=torch.device("cpu"), noise=official[1]["samples"],
                                  cross_attn=positive[0][0], **positive[0][1])
        self.assertIn("minimax_payload", extra)
        return official, applied[0]

    def test_previous_character_files_and_two_cast_default(self):
        for name in ("A", "B"):
            profile(name).save(self.root / f"{name}.safetensors")
        self.assertEqual(N._list_mod_names(), ["A", "B"])
        mods, hint, clip = self.load(mod_1="A", mod_2="B")
        self.assertIn("<Picture 1>, <Picture 2>, <Picture 3>, <Audio 1>", hint)
        self.assertIn("<Picture 4>, <Picture 5>, <Picture 6>, <Audio 2>", hint)
        official, positive = self.generate_conditioning(mods, clip)
        refs = positive[0][1]["minimax_refs"]
        self.assertEqual([r["kind"] for r in refs], ["image"] * 6 + ["audio"] * 2)
        self.assertAlmostEqual(sum(r["ref_audio_t"] / 40 for r in refs if "ref_audio_t" in r), 8.1)
        self.assertIsNone(self.base_clip.last_tokens)
        self.assertNotIn("minimax_refs", official[0][0][1])
        self.assertEqual(clip.base.prompt, "Custom full Ref2VA prompt: <d>[English] New words.</d>")
        self.assertEqual(len(mods[0][0].profile.references), 5)

    def test_all_voices_is_explicit_and_not_a_15_second_architecture_error(self):
        for name in ("A", "B"):
            profile(name).save(self.root / f"{name}.safetensors")
        mods, _, clip = self.load(mod_1="A", mod_2="B", voice_reference_1=0, voice_reference_2=0)
        _, positive = self.generate_conditioning(mods, clip)
        voices = [r for r in positive[0][1]["minimax_refs"] if "audio_latent" in r]
        self.assertEqual(len(voices), 4)
        self.assertAlmostEqual(sum(r["ref_audio_t"] / 40 for r in voices), 18.1)

    def test_original_extractor_one_file_and_resave_keeps_every_recording(self):
        target = self.root / "A"
        with patch.object(N, "mod_output_path", return_value=str(target)):
            out = N.MiniMaxH3RefModExtract.execute("A", "encode", vae=VideoVAE(), audio_vae=AudioVAE(),
                refs_image={"ref_image_1": torch.full((1, 320, 320, 3), 0.5)},
                audio=audio(4), refs_audio={"ref_audio_1": audio(5, 0.2)}, ref_resolution=320)
        self.assertEqual([p.name for p in self.root.glob("*.safetensors")], ["A.safetensors"])
        mod = N.H3RefMod.load(str(target))
        self.assertEqual(len(mod.profile.references), 3)
        selected = copy.copy(mod)
        selected.voice_reference = 2
        selected.config = {"retention": 0.75}
        selected.save(str(self.root / "resaved"))
        restored = N.H3RefMod.load(str(self.root / "resaved"))
        self.assertEqual(len(restored.profile.references), 3)
        self.assertEqual(restored.config, {"retention": 0.75})
        self.assertGreater(restored.storage_bytes, restored.latent.numel() * restored.latent.element_size())

    def test_cached_and_direct_official_native_reference_parity(self):
        image = torch.full((1, 320, 320, 3), 0.5)
        wave = audio(1)
        mod = E.extract_profile("A", [(image, False)], VideoVAE(), AudioVAE(), audio=wave, ref_resolution=320)
        mod.save(str(self.root / "A"))
        mods, _, clip = self.load(mod_1="A")
        cached, positive = self.generate_conditioning(mods, clip, prompt="An unchanged custom prompt.")
        direct_clip = self.base_clip.clone()
        direct = MiniMaxH3ReferenceToVideo.execute(direct_clip, "An unchanged custom prompt.", 320, 320, 124,
            ref_image_size="max", vae=VideoVAE(), audio_vae=AudioVAE(),
            ref_images={"ref_image_1": image}, ref_audios={"ref_audio_1": wave})
        equal_values(self, clip.base.last_tokens, direct_clip.last_tokens)
        for actual, expected in zip(positive[0][1]["minimax_refs"], direct[0][0][1]["minimax_refs"]):
            equal_values(self, {k: actual[k] for k in expected}, expected)
        for actual, expected in zip(cached[1]["samples"].tensors, direct[1]["samples"].tensors):
            self.assertTrue(torch.equal(actual, expected))

    def test_speaking_video_retains_joint_timing_with_compression(self):
        frames = torch.full((22, 320, 320, 3), 128, dtype=torch.uint8)
        clip = M.CharacterClip(frames, audio(22 / 24), 0, 22 / 24, "talk.mp4")
        with patch.object(E, "media_path", return_value="talk.mp4"), patch.object(E, "load_character_clip", return_value=clip):
            mod = E.extract_profile("speaker", [], VideoVAE(), AudioVAE(), video_file="talk.mp4",
                                    ref_resolution=320, mode="training", pool_h=4, pool_w=4, identity=0)
        mod.save(str(self.root / "speaker"))
        mods, _, prepared = self.load(mod_1="speaker")
        _, positive = self.generate_conditioning(mods, prepared)
        ref = positive[0][1]["minimax_refs"][0]
        self.assertEqual(ref["kind"], "video_audio")
        self.assertEqual(ref["latent"].shape, (1, 24, 7, 4, 4))
        self.assertEqual(ref["ref_audio_t"], 37)
        self.assertEqual(prepared.base.items[1]["timestamps"], [0, 0.5])

    def test_mismatch_scramble_duplicate_and_missing_preparation_fail(self):
        profile("A").save(self.root / "A.safetensors")
        mods, _, clip = self.load(mod_1="A")
        official, positive = self.generate_conditioning(mods, clip)
        other, _, _ = self.load(mod_1="A", voice_reference_1=2)
        for conditioning, bundle, seed, message in (
            (official[0], other, -1, "differ"),
            (official[0], mods, 123, "Fixed"),
            (positive, mods, -1, "already"),
            ([[torch.zeros(1), {}]], mods, -1, "CLIP Loader"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                N.MiniMaxH3RefModApply.execute(conditioning, bundle, scramble_seed=seed)
        with self.assertRaisesRegex(ValueError, "retention > 0"):
            N.MiniMaxH3RefModApply.execute(official[0], mods, retention=0)
        with self.assertRaisesRegex(ValueError, "Continuum bridge"):
            N.MiniMaxH3RefModContinuumBridge().arm(None, mods)

    def test_native_raw_reference_after_profiles_and_clip_clone(self):
        profile("A", (1,), images=1).save(self.root / "A.safetensors")
        mods, _, clip = self.load(mod_1="A")
        clone = clip.clone()
        image = torch.zeros(1, 320, 320, 3)
        _, positive = self.generate_conditioning(mods, clone, vae=VideoVAE(), ref_images={"ref_image_1": image})
        self.assertEqual([r["kind"] for r in positive[0][1]["minimax_refs"]], ["image", "audio", "image"])
        self.assertEqual([i["type"] for i in clone.base.items], ["image", "audio", "image"])
        self.assertIsNone(clip.base.last_tokens)
        self.assertIsNone(self.base_clip.last_tokens)

    def test_many_stored_recordings_do_not_all_become_generation_inputs(self):
        mod = profile("archive", (6,) * 5 + (9,) * 3 + (4.5,) * 2)
        N.ProfileRefMod.from_profile(mod).save(str(self.root / "archive"))
        mods, _, clip = self.load(mod_1="archive")
        self.assertEqual(len(mods[0][0].profile.references), 13)
        _, positive = self.generate_conditioning(mods, clip)
        self.assertEqual(sum("audio_latent" in r for r in positive[0][1]["minimax_refs"]), 1)

    def test_actual_video_file_decoder_via_original_extractor(self):
        # Real container demux/resampling; only VAE model forwards are doubles.
        from test_characters import MediaTests
        path = self.root / "talk.mkv"
        MediaTests.make_clip(path)
        with patch.object(N, "mod_output_path", return_value=str(self.root / "speaker")):
            out = N.MiniMaxH3RefModExtract.execute("speaker", "encode", vae=VideoVAE(), audio_vae=AudioVAE(),
                video_file=str(path), audio_duration_seconds=2, ref_resolution=320)
        stored = N.H3RefMod.load(str(self.root / "speaker"))
        ref = stored.profile.references[0]
        self.assertEqual(ref.kind, "video_audio")
        self.assertAlmostEqual(ref.duration, 39 / 24)
        self.assertEqual(ref.visual.shape[2], 12)
        self.assertEqual(ref.audio.shape[-1], 65)

    def test_strength_copies_and_token_budget(self):
        profile("A", (1,), images=1).save(self.root / "A.safetensors")
        mods, hint, clip = self.load(mod_1="A", copies_1=2, strength_1=0.5)
        _, positive = self.generate_conditioning(mods, clip)
        self.assertEqual(len(positive[0][1]["minimax_refs"]), 4)
        self.assertIn("<Audio 2>", hint)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.load(mod_1="A", copies_1=2, max_total_tokens=359)
        refs_before = [r.block() for r in mods[0][0].profile.references]
        official = MiniMaxH3ReferenceToVideo.execute(clip, "Different words", 320, 320, 124)
        N.MiniMaxH3RefModApply.execute(official[0], mods, curve_direction="concept_at_end", curve_value=0.5)
        equal_values(self, refs_before, [r.block() for r in mods[0][0].profile.references])


if __name__ == "__main__":
    unittest.main()
