"""CPU tests: python -m unittest discover -s tests -v

Tiny encoder doubles exercise the node contracts without distributing H3
weights. Tensor math, serialization, and media decoding use real libraries.
"""
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("h3_character_testpack")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
core = importlib.import_module(package.__name__ + ".character_core")
media = importlib.import_module(package.__name__ + ".character_media")
conditioning = importlib.import_module(package.__name__ + ".character_conditioning")


def image_ref(value=1.0):
    return core.CharacterReference("image", visual=torch.full((1, 24, 1, 20, 20), value),
                                   frames=torch.zeros((1, 320, 320, 3), dtype=torch.uint8))


def audio_ref(value=1.0, duration=1.0):
    return core.CharacterReference("audio", audio=torch.full((1, 32, 2, math.ceil(duration * 40)), float(value)),
                                   duration=duration)


def paired_ref():
    return core.CharacterReference("video_audio", visual=torch.ones(1, 24, 7, 20, 20),
        audio=torch.ones(1, 32, 2, 37), frames=torch.zeros((2, 320, 320, 3), dtype=torch.uint8),
        timestamps=torch.tensor([0.0, 0.5], dtype=torch.float64), duration=22 / 24)


def character(name="A", two_voices=False):
    return core.H3CharacterMod(name, [image_ref(), audio_ref()] + ([audio_ref(2)] if two_voices else []))


class FormatTests(unittest.TestCase):
    def test_images_and_two_voices_roundtrip(self):
        mod = core.H3CharacterMod("A", [image_ref(i) for i in (1., 2., 3.)] + [audio_ref(), audio_ref(2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A.safetensors"
            mod.save(path)
            loaded = core.H3CharacterMod.load(path)
            self.assertEqual(loaded.character_id, mod.character_id)
            self.assertEqual(len(loaded.references), 5)
            self.assertEqual(loaded.token_count, 460)
            for old, new in zip(mod.references, loaded.references):
                for field in ("visual", "audio", "frames"):
                    if getattr(old, field) is not None:
                        self.assertTrue(torch.equal(getattr(old, field), getattr(new, field)))
            with self.assertRaises(FileExistsError):
                mod.save(path)
            replace(mod, name="Changed").save(path, overwrite=True)
            self.assertEqual(core.H3CharacterMod.load(path).name, "Changed")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_paired_roundtrip_and_payload(self):
        mod = core.H3CharacterMod("Speaking", [paired_ref()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaking.safetensors"
            mod.save(path)
            ref = core.H3CharacterMod.load(path).references[0]
            block = ref.block()
            self.assertEqual(block["kind"], "video_audio")
            self.assertEqual(block["ref_audio_t"], 37)
            self.assertEqual(block["latent_t"], 7)
            block["audio_latent"].zero_()
            self.assertGreater(ref.audio.sum(), 0)

    def test_invalid_pair_timing_is_rejected(self):
        ref = paired_ref()
        ref.audio = torch.ones(1, 32, 2, 100)
        with self.assertRaisesRegex(ValueError, "duration"):
            ref.validate()

    def test_invalid_vae_shape_and_nan_are_rejected(self):
        ref = image_ref()
        ref.visual = torch.zeros(1, 16, 1, 20, 20)
        with self.assertRaises(ValueError):
            ref.validate()
        ref = audio_ref()
        ref.audio[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            ref.validate()

    def test_future_and_non_character_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.safetensors"
            save_file({"x": torch.zeros(1)}, str(path), metadata={core.META_KEY: json.dumps({"format_version": 99})})
            with self.assertRaisesRegex(ValueError, "version"):
                core.H3CharacterMod.load(path)
            with self.assertRaisesRegex(ValueError, "older visual"):
                core.H3CharacterMod.load(ROOT / "mods" / "vanellope_example.safetensors")

    def test_legacy_mod_still_loads(self):
        old = importlib.import_module(package.__name__ + ".core")
        mod = old.H3RefMod.load(str(ROOT / "mods" / "vanellope_example"))
        self.assertEqual(mod.token_count, 2816)

    def test_filename_sanitization(self):
        self.assertEqual(core.safe_name("../my:voice.safetensors"), "_my_voice")
        for name in ("..", "CON", "NUL.txt", "   "):
            with self.assertRaises(ValueError):
                core.safe_name(name)


class ConditioningTests(unittest.TestCase):
    def test_two_voices_remain_distinct(self):
        mod = character(two_voices=True)
        prompt, items, blocks, info = conditioning.build_character_conditioning(
            [(mod, "New words", "happy", "soft")], "A room")
        self.assertEqual([x["type"] for x in items], ["image", "audio", "audio"])
        self.assertEqual([x["kind"] for x in blocks], ["image", "audio", "audio"])
        self.assertNotIn("latent", blocks[1])
        self.assertEqual(blocks[1]["audio_latent"].mean(), 1)
        self.assertEqual(blocks[2]["audio_latent"].mean(), 2)
        self.assertIn("<Audio 1>, <Audio 2>", prompt)
        self.assertIn("New words", prompt)
        self.assertIn("happy", prompt)

    def test_mixed_cast_has_stable_native_order_and_owner_labels(self):
        cast = [(core.H3CharacterMod("A", [paired_ref()]), "A line", "calm", "slow"),
                (character("B"), "B line", "happy", "quick")]
        prompt, items, blocks, info = conditioning.build_character_conditioning(cast, "Room")
        self.assertEqual([x["type"] for x in items], ["image", "audio", "video", "audio"])
        self.assertEqual([x["kind"] for x in blocks], ["image", "video_audio", "audio"])
        self.assertIn("A -> <Subject 1>: <Video 1>; voice <Audio 1>", info)
        self.assertIn("B -> <Subject 2>: <Picture 1>; voice <Audio 2>", info)
        self.assertIn("synchronized soundtrack", prompt)

    def test_source_transcript_never_injected(self):
        mod = character()
        mod.provenance["transcript"] = "Source secret sentence"
        prompt, _, _, _ = conditioning.build_character_conditioning([(mod, "Target", "neutral", "slow")], "Room")
        self.assertNotIn("Source secret sentence", prompt)

    def test_selections_budget_and_voice_limit(self):
        mod = character(two_voices=True)
        first = conditioning.select_voice(mod, "first")
        second = conditioning.select_voice(mod, "second")
        self.assertEqual(first.references[-1].audio.mean(), 1)
        self.assertEqual(second.references[-1].audio.mean(), 2)
        self.assertEqual(len(mod.references), 3)
        with self.assertRaisesRegex(ValueError, "budget"):
            conditioning.build_character_conditioning([(mod, "A", "", "")], "Room", max_tokens=1)
        with self.assertRaisesRegex(ValueError, "reference limit"):
            conditioning.build_character_conditioning([(mod, "A", "", "")] * 2, "Room")


class MediaTests(unittest.TestCase):
    def test_mono_resample_and_crop(self):
        sr = 44100
        waveform = torch.full((1, 1, sr * 2), 0.2)
        z, seconds = media.normalize_audio({"waveform": waveform, "sample_rate": sr}, 0.5, 1)
        self.assertEqual(tuple(z.shape), (1, 2, 32000))
        self.assertAlmostEqual(seconds, 1)
        self.assertTrue(torch.allclose(z[:, 0], z[:, 1]))
        self.assertAlmostEqual(float(z.mean()), 0.2, places=3)

    def test_empty_and_short_audio(self):
        audio = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 32000}
        with self.assertRaises(ValueError):
            media.normalize_audio(audio)
        with self.assertRaises(ValueError):
            media.normalize_audio(audio, 2, 1)

    @staticmethod
    def make_clip(path, audio=True, offset=0.5):
        with av.open(str(path), "w") as container:
            video = container.add_stream("ffv1", rate=24)
            video.width = video.height = 64
            video.pix_fmt = "bgr0"
            sound = container.add_stream("pcm_s16le", rate=32000) if audio else None
            if sound is not None:
                sound.layout = "stereo"
            for i in range(48):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 4, np.uint8), format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, 24)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            if sound is not None:
                for i in range(0, 32000, 800):
                    frame = av.AudioFrame.from_ndarray(np.full((1, 1600), 10000, np.int16), format="s16", layout="stereo")
                    frame.sample_rate, frame.pts, frame.time_base = 32000, round(offset * 32000) + i, Fraction(1, 32000)
                    for packet in sound.encode(frame):
                        container.mux(packet)
                for packet in sound.encode():
                    container.mux(packet)

    def test_real_video_preserves_delayed_soundtrack(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "talk.mkv"
            self.make_clip(path)
            clip = media.load_character_clip(path, 0, 2, 320)
            self.assertEqual(len(clip.frames), 39)
            self.assertAlmostEqual(clip.duration, 39 / 24)
            z = clip.audio["waveform"]
            self.assertEqual(float(z[..., :14000].abs().max()), 0)
            self.assertGreater(float(z[..., 18000:30000].abs().mean()), 0.2)
            cropped = media.load_character_clip(path, 0.5, 1, 320)
            self.assertEqual(int(cropped.frames[0, 0, 0, 0]), 48)
            self.assertGreater(float(cropped.audio["waveform"][..., 100:10000].abs().mean()), 0.2)

    def test_missing_soundtrack_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silent.mkv"
            self.make_clip(path, audio=False)
            with self.assertRaisesRegex(ValueError, "soundtrack"):
                media.load_character_clip(path)

    def test_fractional_duration_does_not_extend_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "talk.mkv"
            self.make_clip(path, offset=0)
            clip = media.load_character_clip(path, 0, 0.91, 320)
            self.assertEqual(len(clip.frames), 5)  # 22 frames would exceed 0.91s.
            self.assertLessEqual(clip.duration, 0.91)
            wave, duration = media.normalize_audio(clip.audio, 0, clip.duration, min_duration_seconds=5 / 24)
            self.assertEqual(wave.shape[-1], round(5 / 24 * 32000))


class VideoVAE:
    audio_sample_rate = 44100  # Present even on native ComfyUI visual VAEs.
    latent_channels = 24
    latent_dim = 3
    output_channels = 3

    def encode(self, pixels):
        t = 1 if len(pixels) == 1 else 2 + ((len(pixels) - 5) // 17) * 5
        return torch.ones(1, 24, t, pixels.shape[1] // 16, pixels.shape[2] // 16)


class AudioVAE:
    audio_sample_rate = 32000
    latent_channels = 32
    latent_dim = 2
    output_channels = 2

    def encode(self, pixels):
        assert pixels.shape[-1] == 2  # Comfy VAE wrapper consumes channel-last.
        return torch.ones(1, 32, 2, math.ceil(pixels.shape[1] / 800))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folders = types.ModuleType("folder_paths")
        folders.models_dir = self.directory.name
        folders.get_input_directory = lambda: self.directory.name
        self.context = patch.dict(sys.modules, {"folder_paths": folders})
        self.context.start()
        self.addCleanup(self.context.stop)
        self.nodes = importlib.import_module(package.__name__ + ".character_nodes")
        self.nodes.folder_paths = folders

    def test_extract_three_images_two_audio_then_reload(self):
        image = torch.zeros(1, 320, 320, 3)
        audio = {"waveform": torch.ones(1, 1, 32000) * 0.1, "sample_rate": 32000}
        mod, info = self.nodes.ExtractH3Character().extract("A", VideoVAE(), AudioVAE(),
            image_1=image, image_2=image, image_3=image, audio_1=audio, audio_2=audio)
        self.assertEqual(len(mod.references), 5)
        self.assertIn("A.safetensors", self.nodes.character_files())
        loaded, _ = self.nodes.LoadH3Character().load("A.safetensors", "first")
        self.assertEqual(len(loaded.references), 4)
        before = self.nodes.LoadH3Character.IS_CHANGED("A.safetensors")
        mod.description = "Changed"
        mod.save(self.nodes.character_path("A.safetensors"), overwrite=True)
        self.assertNotEqual(before, self.nodes.LoadH3Character.IS_CHANGED("A.safetensors"))
        self.assertEqual(self.nodes.LoadH3Character().load("A.safetensors")[0].description, "Changed")

    def test_extract_video_only_with_embedded_audio(self):
        path = Path(self.directory.name) / "talk.mkv"
        MediaTests.make_clip(path)
        clip, _, _, _ = self.nodes.LoadH3CharacterClip().load(str(path), duration_seconds=2, max_edge=320)
        mod, _ = self.nodes.ExtractH3Character().extract("video", VideoVAE(), AudioVAE(), character_clip=clip)
        self.assertEqual([r.kind for r in mod.references], ["video_audio"])
        mod.validate()

    def test_wrong_vae_selection_fails_before_either_encode(self):
        cases = ((AudioVAE(), AudioVAE(), "video_vae"),
                 (VideoVAE(), VideoVAE(), "audio_vae"),
                 (AudioVAE(), VideoVAE(), "video_vae"))
        for video, audio, socket in cases:
            with self.subTest(socket=socket, video=type(video), audio=type(audio)):
                with patch.object(video, "encode") as video_encode, patch.object(audio, "encode") as audio_encode:
                    with self.assertRaisesRegex(ValueError, f"Incorrect VAE at {socket}.*vae_name"):
                        self.nodes.ExtractH3Character().extract("wrong", video, audio,
                            image_1=torch.zeros(1, 320, 320, 3),
                            audio_1={"waveform": torch.zeros(1, 1, 32000), "sample_rate": 32000})
                    video_encode.assert_not_called()
                    audio_encode.assert_not_called()
                self.assertFalse((self.nodes.characters_dir() / "wrong.safetensors").exists())

    def test_wrong_audio_rate_fails_before_visual_encode(self):
        video, audio = VideoVAE(), AudioVAE()
        audio.audio_sample_rate = 44100
        with patch.object(video, "encode") as video_encode, patch.object(audio, "encode") as audio_encode:
            with self.assertRaisesRegex(ValueError, "Incorrect VAE at audio_vae.*32 kHz"):
                self.nodes.ExtractH3Character().extract("wrong_rate", video, audio, save=False)
            video_encode.assert_not_called()
            audio_encode.assert_not_called()

    def test_invalid_mixed_inputs_and_path(self):
        with self.assertRaises(ValueError):
            self.nodes.character_path("../outside.safetensors")
        with self.assertRaisesRegex(ValueError, "not both"):
            self.nodes.ExtractH3Character().extract("A", VideoVAE(), AudioVAE(), save=False,
                character_clip=object(), image_1=torch.zeros(1, 320, 320, 3))

    def test_corrupt_file_does_not_break_dropdown(self):
        (self.nodes.characters_dir() / "broken.safetensors").write_bytes(b"broken file")
        self.assertEqual(self.nodes.character_files(), [])

    def test_registered_roots_and_legacy_character_discovery(self):
        legacy_path = self.nodes.characters_dir() / "legacy.safetensors"
        character("legacy").save(legacy_path)
        registered = Path(self.directory.name) / "external"
        self.nodes.folder_paths.get_folder_paths = lambda name: [str(registered)]
        self.assertEqual(self.nodes.characters_dir(), registered / "characters")
        path = registered / "characters" / "cast" / "A.safetensors"
        character().save(path)
        self.assertEqual(self.nodes.character_files(), ["cast/A.safetensors", "legacy.safetensors"])
        self.assertEqual(self.nodes.character_path("cast\\A.safetensors"), path.resolve())
        self.assertEqual(self.nodes.character_path("legacy.safetensors"), legacy_path.resolve())


if __name__ == "__main__":
    unittest.main()
