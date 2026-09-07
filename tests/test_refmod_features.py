import copy
import importlib
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import test_loaders as loaders
N, torch = loaders.N, loaders.torch
from safetensors.torch import save_file

CORE = importlib.import_module(N.__package__ + ".core")
COMMON = importlib.import_module(N.__package__ + ".common")
AUDIO = importlib.import_module(N.__package__ + ".audio")


class FeatureTests(unittest.TestCase):
    setUp = loaders.LoaderTests.setUp
    save_mod = loaders.LoaderTests.save_mod

    def test_cache_overwrite_and_execution_fingerprint(self):
        path = self.save_mod("a", value=1)
        original = N.MiniMaxH3RefModsLoader.IS_CHANGED(mod_1="a")
        N._load_mod("a")
        self.save_mod("a", value=2)
        self.assertNotEqual(original, N.MiniMaxH3RefModsLoader.IS_CHANGED(mod_1="a"))
        self.assertEqual(N._load_mod("a").latent.flatten()[0], 2)
        Path(str(path) + ".safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            N._load_mod("a")

    def test_cache_byte_budget(self):
        self.save_mod("a")
        with patch.object(N, "_MOD_CACHE_BYTES", 1):
            self.assertEqual(N._load_mod("a").name, "a")
            self.assertEqual(N._MOD_CACHE, {})

    def test_numeric_limits_static_and_runtime(self):
        self.save_mod("a")
        for value in (-1, 0, 11, 1.5, float("nan"), float("inf")):
            self.assertIsNot(N.MiniMaxH3RefModsLoader.VALIDATE_INPUTS(copies_1=value), True)
            with self.assertRaises(ValueError):
                N.MiniMaxH3RefModsLoader().load(mod_1="a", copies_1=value)
        for value in (float("nan"), float("inf"), -1, 1.1):
            with self.assertRaises(ValueError):
                N.MiniMaxH3RefModsLoader().load(mod_1="a", strength_1=value)

    def test_global_budget_after_copies(self):
        self.save_mod("a")
        with self.assertRaisesRegex(ValueError, "after copies"):
            N.MiniMaxH3RefModsLoader().load(mod_1="a", copies_1=2, max_total_tokens=7)
        self.assertEqual(len(N.MiniMaxH3RefModsLoader().load(mod_1="a", copies_1=2, max_total_tokens=8)[0]), 2)
        with self.assertRaisesRegex(ValueError, "one frame"):
            CORE.fit_token_budget(torch.zeros(1,24,1,8,8), 1, "large")

    def test_step_curve_empty_keyframes_audio_and_no_mutation(self):
        image, audio = torch.randn(1,24,1,8,8), torch.randn(1,32,2,20)
        native = torch.randn_like(image)
        payload = {"keyframes": [{"latent": native}],
                   "refs": [{"latent": image, "refmod": True}, {"audio_latent": audio, "refmod": True}],
                   "cond_video_latents": [native, image], "cond_audio_latents": [audio]}
        wrapper = N._make_step_wrapper(("constant", "linear", .5))
        def forward(*args, **kwargs):
            return kwargs.get("minimax_payload")
        result = wrapper(forward, None, torch.tensor([1000.]), None, {}, minimax_payload=payload)
        self.assertIs(result["cond_video_latents"][0], native)
        torch.testing.assert_close(result["cond_video_latents"][1], .5 * image + .5 * CORE._blur_latent(image))
        torch.testing.assert_close(result["cond_audio_latents"][0], .5 * audio + .5 * CORE._blur_latent(audio))
        self.assertIs(payload["cond_video_latents"][1], image)
        for empty in ({}, {"refs": [], "cond_video_latents": []}, None):
            self.assertIs(wrapper(forward, None, torch.tensor([1000.]), None, {}, minimax_payload=empty), empty)
        again = wrapper(forward, None, torch.tensor([1000.]), None, {}, minimax_payload=payload)
        torch.testing.assert_close(again["cond_video_latents"][1], result["cond_video_latents"][1])

    def test_chained_step_curves_and_cfg(self):
        z = torch.randn(1,24,2,8,8)
        a = N._make_step_wrapper(("constant", "linear", .5))
        b = N._make_step_wrapper(("constant", "linear", .8))
        def forward(*args, **kwargs): return kwargs["minimax_payload"]
        for value in (z, z * 2, z):
            payload = {"refs": [{"latent": value, "refmod": True}], "cond_video_latents": [value]}
            def next_wrapper(*args, **kwargs): return b(forward, *args, **kwargs)
            result = a(next_wrapper, None, torch.tensor([500.]), None, {}, minimax_payload=payload)
            first = value * .5 + CORE._blur_latent(value) * .5
            torch.testing.assert_close(result["cond_video_latents"][0], .8 * first + .2 * CORE._blur_latent(first))

    def test_bridge_scoped_and_cleanup_on_exception(self):
        from comfy.model_patcher import ModelPatcher
        from comfy.patcher_extension import WrappersMP
        model = ModelPatcher(torch.nn.Linear(1,1), torch.device("cpu"), torch.device("cpu"))
        z = torch.ones(1,24,1,4,4)
        patched = N.continuum_bridge.attach(model, [{"latent":z, "kind":"image", "refmod":True}])
        self.assertFalse(model.get_wrappers(WrappersMP.OUTER_SAMPLE, N.continuum_bridge.BRIDGE_KEY))
        wrappers = patched.get_wrappers(WrappersMP.OUTER_SAMPLE, N.continuum_bridge.BRIDGE_KEY)
        wrapper = wrappers[0]
        original = {"positive": [{"minimax_refs": [{"native":True}]}], "negative": [{}]}
        guider = types.SimpleNamespace(conds=original)
        observed = []
        class Executor:
            class_obj = guider
            def __call__(self):
                observed.append(guider.conds)
                raise RuntimeError("cancelled")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                wrapper(Executor())
            self.assertIs(guider.conds, original)
        self.assertEqual(len(observed[0]["positive"][0]["minimax_refs"]), 2)
        self.assertEqual(len(original["positive"][0]["minimax_refs"]), 1)
        disabled = N.continuum_bridge.attach(patched, [], False)
        self.assertFalse(disabled.get_wrappers(WrappersMP.OUTER_SAMPLE, N.continuum_bridge.BRIDGE_KEY))

    def test_mask_matches_center_crop(self):
        for h,w,canvas in ((64,128,(64,64)), (128,64,(64,64))):
            image = torch.zeros(1,h,w,3)
            image[:,:h//4,:w//4] = 1
            result = N._resize_ref(image,64,canvas)
            mask = N._resize_mask(image[...,0],64,64,"center")
            torch.testing.assert_close(result[...,0], mask)

    def test_video_unknown_length_bounded_and_ordered(self):
        frames = [np.full((2,2,3),i,dtype=np.uint8) for i in range(100)]
        original_stack = np.stack
        counts = []
        def stack(values):
            counts.append(len(values))
            return original_stack(values)
        with patch.object(np,"stack",side_effect=stack):
            result = COMMON._sample_video_frames(iter(frames),0,3,None)
        self.assertEqual(counts,[3])
        values = result[:,0,0,0].tolist()
        self.assertEqual(values,sorted(values))
        self.assertEqual(len(values),3)

    def test_audio_roundtrip_legacy_import_and_mixed_bundle(self):
        z = torch.randn(1,32,2,40)
        p = self.root / "voice"
        mod = N.H3RefMod(name="voice",kind="audio",latent=z,concept_type="voice")
        mod.save(str(p))
        loaded = N._load_mod("voice")
        self.assertEqual(loaded.token_count,80)
        torch.testing.assert_close(loaded.latent,z)
        save_file({"latent":z}, str(self.root / "legacy.safetensors"),
                  metadata={"audio_refmod_meta":json.dumps({"kind":"audio","name":"old","_format_version":1})})
        self.assertEqual(N._load_mod("legacy").kind,"audio")
        self.assertIn("legacy",N._list_mod_names())
        self.save_mod("face")
        blocks = N._ref_blocks([(loaded,.5),(N._load_mod("face"),1)],1,curve=("constant","linear",1))
        self.assertEqual([b["kind"] for b in blocks],["audio","image"])
        self.assertNotIn("latent",blocks[0])
        torch.testing.assert_close(blocks[0]["audio_latent"], .5*z+.5*CORE._blur_latent(z))

    def test_audio_budget_preserves_contiguous_time(self):
        z = torch.randn(1,32,2,20)
        with patch.object(AUDIO,"encode_audio",return_value=z):
            with self.assertRaises(ValueError):
                AUDIO.make_audio_mod(None,None,"a",max_tokens=10)
            m = AUDIO.make_audio_mod(None,None,"a",max_tokens=10,budget_policy="truncate")
            torch.testing.assert_close(m.latent,z[...,:5])

    def test_atomic_save_failure_preserves_previous_file(self):
        p = self.save_mod("a",value=1)
        m = N.H3RefMod.load(str(p)); m.latent.fill_(2)
        with patch.object(CORE,"save_file",side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError): m.save(str(p))
        self.assertEqual(N.H3RefMod.load(str(p)).latent.flatten()[0],1)
        self.assertEqual(list(self.root.glob("*.tmp")),[])

    def test_shuffle_keeps_all_and_subset_is_explicit(self):
        mods = [N.H3RefMod(name=str(i),kind="image",latent=torch.ones(1,24,1,4,4)*i) for i in range(5)]
        bundle = [(m,1) for m in mods]
        self.assertEqual(len(N._ref_blocks(bundle,1,seed=4,scramble_mode="shuffle")),5)
        self.assertEqual(len(N._ref_blocks(bundle,1,seed=4,scramble_mode="subset",scramble_keep=2)),2)

    def test_multi_ref_strategies_match_full_objective_with_mixed_shapes(self):
        torch.manual_seed(417)
        initial = torch.randn(1, 3, 2, 2, 2)
        targets = [torch.randn(1, 3, *shape) for shape in ((3,4,4),(3,4,4),(2,6,4))]
        param = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([param], lr=0.02)
        for _ in range(7):
            optimizer.zero_grad()
            losses = [torch.nn.functional.mse_loss(
                torch.nn.functional.interpolate(param, size=t.shape[2:], mode="trilinear", align_corners=False), t
            ) for t in targets]
            torch.stack(losses).mean().backward()
            optimizer.step()
        for strategy in ("resident", "stream", "grouped"):
            with self.subTest(strategy=strategy):
                result = CORE.optimize_latent_multi(initial, targets, steps=7, device="cpu", strategy=strategy)
                torch.testing.assert_close(result, param.detach(), rtol=1e-4, atol=1e-5)

    def test_inspector_reports_mixed_bundle_without_decoding(self):
        mod = N.H3RefMod(name="voice", kind="audio", latent=torch.ones(1,32,2,20))
        report, image, audio = N.MiniMaxH3RefModInspect().inspect([(mod,1), (mod,.5)])
        self.assertEqual(json.loads(report)["total_tokens"], 80)
        self.assertEqual(tuple(image.shape), (1,64,64,3))
        with self.assertRaisesRegex(ValueError, "audio VAE"):
            N.MiniMaxH3RefModInspect().inspect([(mod,1)], preview="stored", vae=types.SimpleNamespace(first_stage_model=object()))

    def test_library_endpoint_reads_metadata_without_loading_tensors(self):
        import asyncio
        from aiohttp import web
        library = importlib.import_module(N.__package__ + ".library")
        self.save_mod("people/person")
        server = types.SimpleNamespace(routes=web.RouteTableDef())
        library.register_routes(server, N._list_mod_names, N._find_mod_path, CORE.read_refmod_meta)
        with patch.object(CORE, "load_file", side_effect=AssertionError("Tensor load forbidden")):
            response = asyncio.run(next(iter(server.routes)).handler(None))
        self.assertEqual(json.loads(response.text)[0]["name"], "people/person")

    def test_master_combines_and_saves_loadable_modalities(self):
        image = N.H3RefMod(name="hero_visual", kind="image", latent=torch.ones(1,24,1,4,4))
        voice = N.H3RefMod(name="hero_audio", kind="audio", latent=torch.ones(1,32,2,20))
        with patch.object(N.MiniMaxH3RefModExtract, "execute", return_value=N.io.NodeOutput([(image,1)])) as visual, \
             patch.object(N.MiniMaxH3RefModAudioExtract, "extract", return_value=([(voice,1)],)) as audio, \
             patch.object(COMMON, "refmods_dir", return_value=str(self.root)):
            mods, details = N.MiniMaxH3RefModMasterExtract.execute(
                "hero", refs_image={"ref_image_0":torch.zeros(1,64,64,3)}, vae=object(),
                audio={}, audio_vae=object(), subfolder="characters")
        self.assertEqual([m.kind for m,_ in mods], ["image","audio"])
        self.assertEqual(json.loads(details)["total_tokens"], 44)
        self.assertFalse(visual.call_args.kwargs["save"])
        self.assertFalse(audio.call_args.kwargs["save"])
        self.assertEqual(N._load_mod("characters/hero_audio").kind, "audio")
        self.assertEqual(N._load_mod("characters/hero_visual").kind, "image")

    def test_master_audio_only_and_combined_budget_before_save(self):
        voice = N.H3RefMod(name="hero_audio", kind="audio", latent=torch.ones(1,32,2,20))
        with patch.object(N.MiniMaxH3RefModExtract, "execute", side_effect=AssertionError("No visual input")), \
             patch.object(N.MiniMaxH3RefModAudioExtract, "extract", return_value=([(voice,1)],)):
            mods, _ = N.MiniMaxH3RefModMasterExtract.execute("hero", audio={}, audio_vae=object(), save=False)
            self.assertEqual(len(mods),1)
            with self.assertRaises(ValueError):
                N.MiniMaxH3RefModMasterExtract.execute("hero", audio={}, audio_vae=object(), max_total_tokens=39)
        self.assertFalse((self.root / "hero_audio.safetensors").exists())
        with self.assertRaisesRegex(ValueError, "at least one"):
            N.MiniMaxH3RefModMasterExtract.execute("empty")
        with self.assertRaisesRegex(ValueError, "audio_vae"):
            N.MiniMaxH3RefModMasterExtract.execute("voice", audio={})

    def test_master_encode_overwrites_both_files_and_reports_destinations(self):
        import contextlib
        import io
        image = N.H3RefMod(name="hero_visual", kind="image", mode="encode", latent=torch.ones(1,24,1,4,4))
        voice = N.H3RefMod(name="hero_audio", kind="audio", latent=torch.ones(1,32,2,20))
        with patch.object(N.MiniMaxH3RefModExtract,"execute",return_value=N.io.NodeOutput([(image,1)])), \
             patch.object(N.MiniMaxH3RefModAudioExtract,"extract",return_value=([(voice,1)],)), \
             patch.object(COMMON,"refmods_dir",return_value=str(self.root)):
            inputs = dict(name="hero", mode="encode", refs_image={"ref_image_0":object()},
                          vae=object(), audio={}, audio_vae=object(), save=True)
            first = N.MiniMaxH3RefModMasterExtract.execute(**inputs)
            image.latent.fill_(2)
            voice.latent.fill_(3)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                second = N.MiniMaxH3RefModMasterExtract.execute(**inputs)
        paths = json.loads(second[1])["saved_paths"]
        self.assertEqual(paths,json.loads(first[1])["saved_paths"])
        self.assertEqual(log.getvalue().count("[RefMod Master] Replaced:"),2)
        for path,value in zip(paths,(2,3)):
            loaded = N.H3RefMod.load(path.removesuffix(".safetensors"))
            torch.testing.assert_close(loaded.latent,torch.full_like(loaded.latent,value))

    def test_master_schema_keeps_visual_schema_independent(self):
        master = N.MiniMaxH3RefModMasterExtract.define_schema()
        original = N.MiniMaxH3RefModExtract.define_schema()
        self.assertEqual(master.node_id,"MiniMaxH3RefModMasterExtract")
        self.assertEqual(original.node_id,"MiniMaxH3RefModExtract")
        self.assertEqual(len(master.outputs),2)
        schema = N.MiniMaxH3RefModMasterExtract.INPUT_TYPES()
        self.assertIn("audio",schema["optional"])
        self.assertIn("audio_vae",schema["optional"])

    def test_master_visual_only_preserves_extractor_controls(self):
        mod = N.H3RefMod(name="hero_visual", kind="image", latent=torch.ones(1,24,1,4,4))
        with patch.object(N.MiniMaxH3RefModExtract, "execute", return_value=N.io.NodeOutput([(mod,1)])) as visual, \
             patch.object(N.MiniMaxH3RefModAudioExtract, "extract", side_effect=AssertionError("No audio input")):
            mods, _ = N.MiniMaxH3RefModMasterExtract.execute(
                "hero", mode="encode", refs_video={"ref_video_0":torch.zeros(5,64,64,3)},
                vae=object(), save=False, pool_h=8, pool_w=8, identity=0)
        self.assertEqual(mods,[(mod,1)])
        self.assertEqual(visual.call_args.kwargs["mode"],"encode")
        self.assertEqual(visual.call_args.kwargs["pool_h"],8)

    def test_storage_uses_registered_root_and_does_not_create_on_lookup(self):
        external = self.root / "external_drive"
        fallback = self.root / "default_models"
        with patch.dict(loaders.folder_paths.folder_names_and_paths, {"refmods":([str(external),str(self.legacy)], {".safetensors"})}), \
             patch.object(loaders.folder_paths,"models_dir",str(fallback)):
            self.assertEqual(COMMON.refmods_dir(),str(external))
            self.assertFalse(external.exists())
            destination = COMMON.mod_output_path("voice","characters")
            N.H3RefMod(name="voice",kind="audio",latent=torch.ones(1,32,2,4)).save(destination)
            self.assertTrue((external / "characters/voice.safetensors").exists())
            self.assertFalse(fallback.exists())
        with patch.dict(loaders.folder_paths.folder_names_and_paths, {"refmods":([],set())}), \
             patch.object(loaders.folder_paths,"models_dir",str(fallback)):
            self.assertEqual(COMMON.refmods_dir(),str(fallback / "refmods"))
            self.assertFalse(fallback.exists())

    def test_legacy_encoder_resolves_native_video_vae_only(self):
        import comfy.sd
        path = self.root / "video.safetensors"
        path.touch()
        encoder = types.SimpleNamespace(video_path=str(path), audio_path="not-required.safetensors")
        loaded = types.SimpleNamespace(throw_exception_if_invalid=lambda:None)
        with patch.object(N.comfy.utils,"load_torch_file",return_value=({"video":"weights"},{"meta":"v"})) as read, \
             patch.object(comfy.sd,"VAE",return_value=loaded) as construct:
            self.assertIs(N._resolve_visual_vae(av_encoder=encoder),loaded)
            read.assert_called_once_with(str(path),return_metadata=True)
            construct.assert_called_once_with(sd={"video":"weights"},metadata={"meta":"v"})
            self.assertIs(N._resolve_visual_vae(vae=loaded,av_encoder=encoder),loaded)
            self.assertEqual(read.call_count,1)

    def test_save_output_mixed_bundle_deduplicates_copies_and_preserves_inputs(self):
        visual = N.H3RefMod(name="hero", kind="image", latent=torch.ones(1,24,1,4,4))
        audio = N.H3RefMod(name="hero", kind="audio", latent=torch.ones(1,32,2,4))
        with patch.object(COMMON,"refmods_dir",return_value=str(self.root)):
            result = N.MiniMaxH3RefModSave().save([(visual,.5),(visual,.8),(audio,1)],subfolder="characters")
        out, paths = result["result"]
        self.assertEqual(len(paths.splitlines()),2)
        self.assertIs(out[0][0],out[1][0])
        self.assertEqual([strength for _,strength in out],[.5,.8,1])
        self.assertEqual(visual.name,"hero")
        self.assertNotEqual(visual.path,out[0][0].path)
        self.assertEqual(N._load_mod("characters/hero").kind,"image")
        self.assertEqual(N._load_mod("characters/hero_2").kind,"audio")
        self.assertTrue(N.MiniMaxH3RefModSave.OUTPUT_NODE)

    def test_save_output_rejects_empty_and_invalid_destination(self):
        with self.assertRaises(ValueError):
            N.MiniMaxH3RefModSave().save([])
        mod = N.H3RefMod(name="hero",kind="image",latent=torch.ones(1,24,1,4,4))
        with patch.object(COMMON,"refmods_dir",return_value=str(self.root)), \
             patch.object(N.H3RefMod,"save",side_effect=AssertionError("Do not write")):
            with self.assertRaises(ValueError):
                N.MiniMaxH3RefModSave().save([(mod,1)],subfolder="../outside")

    def test_save_is_a_comfy_queue_output_without_downstream_consumers(self):
        import asyncio
        prompt = {"source": {"class_type":"TestRefModSource","inputs":{"text":"unused"}},
                  "save": {"class_type":"MiniMaxH3RefModSave","inputs":{
                      "mods":["source",0],"filename_prefix":"","subfolder":""}}}
        with patch.dict(loaders.comfy_nodes.NODE_CLASS_MAPPINGS,
                        TestRefModSource=loaders.UpstreamString, MiniMaxH3RefModSave=N.MiniMaxH3RefModSave), \
             patch.object(loaders.UpstreamString,"RETURN_TYPES",("H3_REF_MODS",)):
            result = asyncio.run(loaders.execution.validate_prompt("refmod-save",prompt,None))
        self.assertTrue(result[0],result)
        self.assertEqual(result[2],["save"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
