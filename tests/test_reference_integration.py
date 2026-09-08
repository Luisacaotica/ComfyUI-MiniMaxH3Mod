"""Cached payload parity, import failures, and diagnostic isolation on CPU."""
import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from test_characters import core, package, image_ref, audio_ref, paired_ref, character

refs = importlib.import_module(package.__name__ + ".character_references")
legacy = importlib.import_module(package.__name__ + ".core")
imports = importlib.import_module(package.__name__ + ".character_import")


class ReferenceIntegrationTests(unittest.TestCase):
    def test_float_pixels_roundtrip_exact_and_v1_still_loads(self):
        mod = character()
        mod.references[0].frames = torch.rand(1, 320, 320, 3)
        before = mod.references[0].frames.clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.safetensors"
            mod.save(path)
            loaded = core.H3CharacterMod.load(path)
            self.assertTrue(torch.equal(before, loaded.references[0].vision_pixels()))
            self.assertEqual(core.read_character_metadata(path)["format_version"], 2)
            # Actual v1 header and byte frames, rather than merely defaulting a version.
            old = character()
            old.save(path, overwrite=True)
            with safe_open(str(path), framework="pt") as handle:
                tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
                metadata = dict(handle.metadata())
            header = json.loads(metadata[core.META_KEY])
            header["format_version"] = 1
            metadata[core.META_KEY] = json.dumps(header)
            save_file(tensors, str(path), metadata=metadata)
            loaded = core.H3CharacterMod.load(path)
            self.assertEqual(loaded.references[0].frames.dtype, torch.uint8)
            self.assertTrue(torch.equal(loaded.references[0].vision_pixels(), old.references[0].frames.float() / 255))

    def test_invalid_float_pixels_are_rejected(self):
        for value in (float("nan"), -0.1, 1.1):
            ref = image_ref()
            ref.frames = torch.full((1, 320, 320, 3), value)
            with self.assertRaisesRegex(ValueError, "Vision frames"):
                ref.validate()

    def test_ablation_changes_only_pair_packing(self):
        a, b = character(), core.H3CharacterMod("B", [paired_ref()])
        paired = refs.assemble_references([a, b])
        separate = refs.assemble_references([a, b], av_layout="separate")
        self.assertEqual(paired.labels, separate.labels)
        self.assertEqual(paired.tokens, separate.tokens)
        self.assertEqual([x["kind"] for x in separate.blocks], ["image", "audio", "video", "audio"])
        self.assertTrue(torch.equal(paired.blocks[1]["latent"], separate.blocks[2]["latent"]))
        self.assertTrue(torch.equal(paired.blocks[1]["audio_latent"], separate.blocks[1]["audio_latent"]))
        self.assertEqual(separate.blocks[1]["h3_character_owner"], 2)
        self.assertEqual(separate.blocks[2]["h3_character_owner"], 2)
        def audit(assembly, prompt="new words"):
            return assembly.audit([a, b], prompt, 640, 320, 124)
        self.assertEqual(audit(paired)["case_fingerprint"], audit(separate)["case_fingerprint"])
        self.assertNotEqual(audit(paired)["case_fingerprint"], audit(paired, "other words")["case_fingerprint"])
        b.references[0].audio[0, 0, 0, 0] += 1
        changed = refs.assemble_references([a, b])
        self.assertNotEqual(refs.tensor_digest(paired.blocks[1]["audio_latent"]),
                            refs.tensor_digest(changed.blocks[1]["audio_latent"]))

    def test_cached_assembly_has_identical_fingerprint_and_no_shared_pixels(self):
        mod = core.H3CharacterMod("B", [paired_ref()])
        mod.references[0].frames = torch.rand(2, 320, 320, 3)
        before = refs.assemble_references([mod])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "B.safetensors"
            mod.save(path)
            loaded = core.H3CharacterMod.load(path)
        after = refs.assemble_references([loaded])
        self.assertEqual(before.audit([mod], "words", 640, 320, 124),
                         after.audit([loaded], "words", 640, 320, 124))
        original = loaded.references[0].frames.clone()
        after.items[1]["data"].zero_()
        self.assertTrue(torch.equal(loaded.references[0].frames, original))


class UpstreamImportTests(unittest.TestCase):
    def bundle(self):
        return [(legacy.H3RefMod("A_visual", "image", torch.rand(1, 24, 1, 20, 20),
                                latent_h=20, latent_w=20, mode="encode"), 1.0),
                (legacy.H3RefMod("A_audio", "audio", torch.rand(1, 32, 2, 40), mode="encode"), 1.0)]

    def test_import_uses_exact_latents_and_original_pixels(self):
        bundle = self.bundle()
        image = torch.rand(1, 320, 320, 3)
        mod = imports.import_refmods(bundle, [image], "A")
        self.assertTrue(torch.equal(mod.references[0].visual, bundle[0][0].latent))
        self.assertTrue(torch.equal(mod.references[1].audio, bundle[1][0].latent))
        self.assertTrue(torch.equal(mod.references[0].vision_pixels(), image))
        bundle[1][0].latent.zero_()
        self.assertGreater(mod.references[1].audio.sum(), 0)
        self.assertFalse(mod.provenance["paired"])

    def test_import_rejects_ambiguous_or_damaged_sources(self):
        image = torch.zeros(1, 320, 320, 3)
        bundle = self.bundle()
        cases = [(bundle + [bundle[0]], [image], "Duplicate"),
                 ([(bundle[0][0], 0.5), bundle[1]], [image], "strength"),
                 (bundle, [], "matching original"),
                 (bundle, [torch.zeros(1, 320, 640, 3)], "canvas"),
                 ([(replace(bundle[0][0], mode="training"), 1), bundle[1]], [image], "encode-mode"),
                 ([(replace(bundle[0][0], kind="video"), 1), bundle[1]], [image], "source timing")]
        for mods, images, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                imports.import_refmods(mods, images, "A")

    def test_import_rejects_excess_audio_without_truncating(self):
        bundle = self.bundle()
        bundle[1][0].latent = torch.zeros(1, 32, 2, 640)
        with self.assertRaisesRegex(ValueError, "15-second"):
            imports.import_refmods(bundle, [torch.zeros(1, 320, 320, 3)], "A")

    def test_legacy_audio_metadata_without_visual_mode(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.safetensors"
            save_file({"latent": bundle[1][0].latent}, str(path), metadata={
                "audio_refmod_meta": json.dumps({"kind": "audio", "name": "old", "_format_version": 1})})
            audio = legacy.H3RefMod.load(str(path.with_suffix("")))
            mod = imports.import_refmods([bundle[0], (audio, 1)], [torch.zeros(1, 320, 320, 3)], "A")
            self.assertTrue(torch.equal(mod.references[1].audio, bundle[1][0].latent))


if __name__ == "__main__":
    unittest.main()
