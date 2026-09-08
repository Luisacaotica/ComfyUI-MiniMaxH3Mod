"""Loader regressions using real safetensors and ComfyUI's queue validator.

Run with ComfyUI's Python: python tests/test_loaders.py
No models or server are started; all mod files live in temporary folders.
"""

import asyncio
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from comfy.cli_args import args

args.cpu = True

import folder_paths
import torch
from safetensors.torch import save_file

with tempfile.TemporaryDirectory() as import_dir:
    with patch.object(folder_paths, "models_dir", import_dir):
        N = importlib.import_module(f"custom_nodes.{ROOT.name}.nodes")

import execution
import nodes as comfy_nodes


class UpstreamString:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}

    def run(self, text):
        raise AssertionError("Queue validation must not execute upstream nodes")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "refmods"
        self.legacy = Path(self.temp.name) / "mods"
        self.root.mkdir()
        self.legacy.mkdir()
        search = patch.object(N, "_mod_search_dirs", return_value=[str(self.root), str(self.legacy)])
        search.start()
        self.addCleanup(search.stop)
        N._MOD_CACHE.clear()
        N._MOD_LIST_CACHE_KEY = N._MOD_LIST_CACHE_VAL = None
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def save_mod(self, name, kind="image", root=None, value=1.0, metadata=None):
        path = (root or self.root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        mod = N.H3RefMod(name=path.name, kind=kind,
                        latent=torch.full((1, 24, 1, 4, 4), value),
                        latent_h=4, latent_w=4)
        if metadata is None:
            mod.save(str(path))
        else:
            save_file({"latent": mod.latent}, str(path) + ".safetensors",
                      metadata={"refmod_meta": json.dumps(metadata)})
        return path

    def test_empty_names_validate_and_execute_in_every_slot(self):
        for cls in (N.MiniMaxH3RefModsLoader, N.MiniMaxH3RefModsAxis):
            for field in cls.INPUT_TYPES()["required"]:
                if not field.startswith("mod_"):
                    continue
                for empty in (None, "", "None", "(none)"):
                    with self.subTest(node=cls.__name__, field=field, empty=empty):
                        inputs = {field: empty}
                        if "_a_" in field or "_b_" in field:
                            inputs["value_" + field.rsplit("_", 1)[1]] = -1 if "_a_" in field else 1
                        self.assertIs(cls.VALIDATE_INPUTS(**inputs), True)
                        result = cls().load(**inputs)
                        self.assertEqual(result[:2], ([], ""))
                        if cls is N.MiniMaxH3RefModsLoader:
                            self.assertIsNone(result[2])

    def test_recursive_listing_filters_metadata_and_presets(self):
        self.save_mod("root")
        self.save_mod("celebs/person", kind="video")
        self.save_mod("oc/deep/person", root=self.legacy)
        self.save_mod("celebs/audio", metadata={"kind": "other"})
        self.save_mod("celebs/bad_json_type", metadata=["image"])
        self.save_mod("graph_presets/hidden")
        self.save_mod("oc/graph_presets/hidden")
        (self.root / "curve.png").touch()
        (self.root / "junk.safetensors").write_bytes(b"invalid")
        expected = ["celebs/person", "oc/deep/person", "root"]
        self.assertEqual(N._list_mod_names(), expected)
        for cls in (N.MiniMaxH3RefModsLoader, N.MiniMaxH3RefModsAxis):
            for field, schema in cls.INPUT_TYPES()["required"].items():
                if field.startswith("mod_"):
                    self.assertEqual(schema[0], ["(none)"] + expected)

    def test_listing_cache_reuses_metadata_reads(self):
        self.save_mod("deep/a")
        with patch.object(N, "read_refmod_meta", wraps=N.read_refmod_meta) as read:
            self.assertEqual(N._list_mod_names(), ["deep/a"])
            self.assertEqual(N._list_mod_names(), ["deep/a"])
            self.assertEqual(read.call_count, 1)

    def test_nested_add_rename_delete_refresh_listing(self):
        self.assertEqual(N._list_mod_names(), [])
        path = self.save_mod("new/deep/a")
        self.assertEqual(N._list_mod_names(), ["new/deep/a"])
        file = Path(str(path) + ".safetensors")
        renamed = file.with_name("b.safetensors")
        file.rename(renamed)
        self.assertEqual(N._list_mod_names(), ["new/deep/b"])
        renamed.unlink()
        self.assertEqual(N._list_mod_names(), [])

    def test_subsecond_metadata_change_refreshes_listing(self):
        path = self.save_mod("deep/a", metadata={"kind": "image"})
        file = Path(str(path) + ".safetensors")
        stamp = 1_700_000_000_100_000_000
        os.utime(file, ns=(stamp, stamp))
        self.assertEqual(N._list_mod_names(), ["deep/a"])
        self.save_mod("deep/a", metadata={"kind": "other"})
        os.utime(file, ns=(stamp + 1_000_000, stamp + 1_000_000))
        self.assertEqual(N._list_mod_names(), [])

    def test_legacy_sidecar_add_change_delete_refresh_listing(self):
        path = self.root / "deep" / "legacy"
        path.parent.mkdir()
        save_file({"latent": torch.zeros(1, 24, 1, 4, 4)}, str(path) + ".safetensors")
        self.assertEqual(N._list_mod_names(), [])
        sidecar = Path(str(path) + ".json")
        sidecar.write_text('{"kind": "image"}')
        self.assertEqual(N._list_mod_names(), ["deep/legacy"])
        self.assertEqual(N._load_mod("deep/legacy").kind, "image")
        sidecar.write_text('{"kind": "unsupported"}')
        self.assertEqual(N._list_mod_names(), [])
        sidecar.unlink()
        self.assertEqual(N._list_mod_names(), [])

    def test_search_root_identity_and_precedence(self):
        self.save_mod("a", value=1)
        self.save_mod("a", root=self.legacy, value=2)
        self.assertEqual(N._list_mod_names(), ["a"])
        self.assertEqual(N._load_mod("a").latent.flatten()[0].item(), 1)
        with patch.object(N, "_mod_search_dirs", return_value=[str(self.legacy)]):
            self.assertEqual(N._load_mod("a").latent.flatten()[0].item(), 2)
        self.save_mod("a", metadata={"kind": "other"})
        self.assertEqual(N._list_mod_names(), [])

    def test_nested_runtime_both_loaders_and_windows_separators(self):
        self.save_mod("celebs/person", value=3)
        self.assertIs(N.MiniMaxH3RefModsLoader.VALIDATE_INPUTS(mod_1="celebs\\person"), True)
        bundle, _, _ = N.MiniMaxH3RefModsLoader().load(mod_1="celebs\\person", strength_1=0.6, copies_1=2)
        self.assertEqual(len(bundle), 2)
        self.assertEqual(bundle[0][1], 0.6)
        self.assertEqual(bundle[0][0].latent.flatten()[0].item(), 3)
        for side, value in (("a", -0.4), ("b", 0.4)):
            bundle, _ = N.MiniMaxH3RefModsAxis().load(**{f"mod_{side}_1": "celebs/person", "value_1": value})
            self.assertEqual(bundle[0][1], 0.4)
            self.assertIs(bundle[0][0], N._load_mod("celebs\\person"))

    def test_same_metadata_name_config_does_not_poison_root_cache(self):
        self.save_mod("person", value=1)
        nested_path = self.save_mod("celebs/person", value=2)
        root_mod = N._load_mod("person")
        nested_mod = N._load_mod("celebs/person")
        N.MiniMaxH3RefModConfig().fix([(nested_mod, 1.0)], retention=0.4)
        self.assertIs(N._load_mod("person"), root_mod)
        self.assertEqual(root_mod.config, {})
        self.assertEqual(N.H3RefMod.load(str(nested_path)).config["retention"], 0.4)

    def test_fixed_missing_names_rejected_and_linked_missing_fail_at_load(self):
        for cls, field in ((N.MiniMaxH3RefModsLoader, "mod_1"),
                           (N.MiniMaxH3RefModsAxis, "mod_a_1"),
                           (N.MiniMaxH3RefModsAxis, "mod_b_1")):
            self.assertIn("not found", cls.VALIDATE_INPUTS(**{field: "missing"}))
            self.assertIs(cls.VALIDATE_INPUTS(input_types={field: "STRING"}, **{field: None}), True)
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            N.MiniMaxH3RefModsLoader().load(mod_1="missing")

    def test_path_boundary_is_checked_at_execution(self):
        for name in ("../outside", "a/../../outside", "/absolute", "C:/absolute", "C:relative",
                     "\\\\server\\share\\mod", "graph_presets/a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                N._load_mod(name)

    def test_actual_comfy_queue_with_unexecuted_upstream(self):
        for cls, field in ((N.MiniMaxH3RefModsLoader, "mod_1"),
                           (N.MiniMaxH3RefModsAxis, "mod_a_1"),
                           (N.MiniMaxH3RefModsAxis, "mod_b_1")):
            inputs = {}
            for name, (kind, options) in cls.INPUT_TYPES()["required"].items():
                inputs[name] = kind[0] if isinstance(kind, list) else options["default"]
            inputs[field] = ["source", 0]
            prompt = {"source": {"class_type": "TestRefModString", "inputs": {"text": "future/person"}},
                      "loader": {"class_type": "TestRefModLoader", "inputs": inputs}}
            with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, TestRefModString=UpstreamString, TestRefModLoader=cls):
                for output_type in ("STRING", "COMBO", ["future/person"]):
                    with self.subTest(node=cls.__name__, field=field, output_type=output_type):
                        with patch.object(UpstreamString, "RETURN_TYPES", (output_type,)):
                            result = asyncio.run(execution.validate_inputs("test-refmods", prompt, "loader", {}))
                            self.assertTrue(result[0], result)
                with patch.object(UpstreamString, "RETURN_TYPES", ("IMAGE",)):
                    result = asyncio.run(execution.validate_inputs("test-refmods", prompt, "loader", {}))
                    self.assertFalse(result[0], result)

    def test_connected_non_mod_inputs_keep_type_checks(self):
        for cls, field in ((N.MiniMaxH3RefModsLoader, "strength_1"),
                           (N.MiniMaxH3RefModsLoader, "copies_1"),
                           (N.MiniMaxH3RefModsLoader, "show_info"),
                           (N.MiniMaxH3RefModsAxis, "value_1")):
            with self.subTest(field=field):
                self.assertIn("expected", cls.VALIDATE_INPUTS(input_types={field: "IMAGE"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
