"""Create minimal combined extraction and official H3 generation workflows."""
import json
from pathlib import Path

from build_character_comparisons import node

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples/native_refmods"


def port(name, type_, link=None):
    return {"name": name, "type": type_, "link": link}


def write(name, graph):
    graph["last_node_id"] = max(n["id"] for n in graph["nodes"])
    graph["last_link_id"] = max((link[0] for link in graph["links"]), default=0)
    (OUT / name).write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def connect(graph, source, out_slot, target, in_name):
    by_id = {n["id"]: n for n in graph["nodes"]}
    number = max((link[0] for link in graph["links"]), default=0) + 1
    src = by_id[source]["outputs"][out_slot]
    index = next(i for i, p in enumerate(by_id[target]["inputs"]) if p["name"] == in_name)
    dst = by_id[target]["inputs"][index]
    assert src["type"] == dst["type"]
    src["links"].append(number)
    dst["link"] = number
    graph["links"].append([number, source, out_slot, target, index, src["type"]])


def extraction(video=False):
    nodes = [node(1, "VAELoader", "H3 VIDEO VAE (appearance)", 20, 130,
                  {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}, [("VAE", "VAE", [])]),
             node(2, "VAELoader", "H3 AUDIO VAE (voice)", 20, 340,
                  {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}, [("VAE", "VAE", [])])]
    for loader in nodes:
        loader["size"] = [420, 140]
    widgets = {"name": "character_a", "mode": "encode", "concept_type": "identity",
               "background_retention": 0.0, "ref_resolution": 768, "pool_h": 16, "pool_w": 16,
               "latent_frames": 16, "identity": 0, "merge": False, "motion_only": False,
               "multiplier": 1, "max_tokens": 0, "description": "", "save": True,
               "extraction_preset": "manual", "subfolder": "characters",
               "video_file": "speaking_character.mp4" if video else "",
               "audio_start_seconds": 0.0, "audio_duration_seconds": 5.0}
    inputs = [port("vae", "VAE"), port("audio_vae", "VAE")]
    if not video:
        inputs += [port(f"refs_image.ref_image_{i}", "IMAGE") for i in range(3)]
        inputs += [port("audio", "AUDIO"), port("refs_audio.ref_audio_0", "AUDIO")]
    nodes.append(node(3, "MiniMaxH3RefModExtract", "Extract H3 RefMod — one appearance + voice file", 940, 140,
                      widgets, [("mods", "H3_REF_MODS", [])], inputs))
    nodes[-1]["size"] = [480, 860]
    text = ("Select both matching H3 VAEs. " +
            ("Upload your speaking video to ComfyUI/input and enter its filename in video_file. Both frames and soundtrack are extracted from the same interval. "
             if video else "Load images and audio of ONE character. Unused image/audio loaders can be deleted. Audio duration applies to each recording separately. ") +
            "Set a name and Queue: this original extractor saves one models/refmods/characters/name.safetensors. "
            "Use 03 or 04 in this folder to generate through the official Reference to Video node. Existing files with this name are replaced.")
    nodes.append(node(9, "Note", "Start here", 20, -100, {"text": text}))
    nodes[-1]["size"] = [1380, 190]
    graph = {"nodes": nodes, "links": [], "groups": [], "config": {}, "version": 0.4,
             "extra": {"ds": {"scale": 0.65, "offset": [30, 150]}}}
    connect(graph, 1, 0, 3, "vae")
    connect(graph, 2, 0, 3, "audio_vae")
    if not video:
        for i in range(3):
            nodes.append(node(4 + i, "LoadImage", f"Character view {i + 1}", 460, 130 + i * 430,
                              {"image": f"character_{i + 1}.png"}, [("IMAGE", "IMAGE", []), ("MASK", "MASK", [])]))
            nodes[-1]["size"] = [420, 360]
            connect(graph, 4 + i, 0, 3, f"refs_image.ref_image_{i}")
        for i in range(2):
            nodes.append(node(7 + i, "LoadAudio", f"Voice example {i + 1}", 20, 590 + i * 260,
                              {"audio": f"voice_{i + 1}.wav"}, [("AUDIO", "AUDIO", [])]))
            nodes[-1]["size"] = [420, 200]
            connect(graph, 7 + i, 0, 3, "audio" if i == 0 else "refs_audio.ref_audio_0")
    write("02_extract_speaking_video.json" if video else "01_extract_images_and_audio.json", graph)


def generation(count):
    graph = json.loads((ROOT / "examples/characters/03_generate_one_character.json").read_text())
    by_id = {n["id"]: n for n in graph["nodes"]}
    graph["links"] = [link for link in graph["links"] if link[1] not in (2, 4) and link[3] not in (2, 4)]
    alive = {link[0] for link in graph["links"]}
    for n in graph["nodes"]:
        for p in n["inputs"]:
            if p["link"] not in alive:
                p["link"] = None
        for p in n["outputs"]:
            p["links"] = [link for link in p["links"] if link in alive]
    values = {"show_info": True}
    for i in range(1, 9):
        values.update({f"mod_{i}": f"characters/character_{chr(96+i)}" if i <= count else "(none)",
                       f"strength_{i}": 1.0, f"copies_{i}": 1})
    values["max_total_tokens"] = 0
    values.update({f"voice_reference_{i}": 1 for i in range(1, 9)})
    replacement = node(2, "MiniMaxH3RefModsLoader", "Load H3 RefMods — select your saved characters", 450, 50,
        values, [("mods", "H3_REF_MODS", []), ("prompt_hint", "STRING", []), ("clip", "CLIP", [])], [port("clip", "CLIP")])
    replacement["size"] = [490, 1080]
    by_id[2].clear(); by_id[2].update(replacement)
    definitions, retention, actions = [], [], []
    for i in range(1, count + 1):
        definitions.append(f"<Subject {i}> is character {chr(64+i)}.")
        retention.append(f"<Subject {i}> (appears in [Shot 1]): fully_preserved - retain this character's appearance and voice identity.")
        line = (["Ready.", "Let's go.", "Wait here.", "I see it.", "Stay close.", "Over there.", "Follow me.", "All clear."][i-1]
                if count == 8 else ["The old bridge is still standing.", "Then we can cross before sunset.", "I will bring the lantern."][i-1])
        timing = f"From {0.8 + (i-1)*1.6:.1f} to {2.1 + (i-1)*1.6:.1f} seconds, " if count == 8 else ""
        actions.append(timing + f"<Subject {i}> (S{i}) speaks with calm, natural delivery: <d>[English] {line}</d>, "
                       "then closes their mouth while listening to the next speaker.")
    prompt = ("subject_definitions:\n" + "\n".join(definitions) +
              "\n\nsummary:\n[reference generation + audio reference] The characters discuss their journey.\n\n"
              "retention_analysis:\n" + "\n".join(retention) +
              "\n\ndetailed_description:\n[Shot 1] A stationary eye-level medium shot shows the characters "
              "in a quiet room lit by soft daylight. Each character's face is clearly visible. They speak "
              "one at a time in the order below, with natural expressions and lip movements.\n" + "\n".join(actions) +
              "\n\noverall_soundscape:\nQuiet room ambience beneath the voices.\n\nnon_diegetic_music:\nN/A")
    replacement = node(4, "MiniMaxH3ReferenceToVideo", "MiniMax H3 Reference to Video — YOUR FULL PROMPT", 1050, 30,
        {"prompt": prompt, "width": 768, "height": 512, "length": {1:124, 2:243, 3:362, 8:362}[count], "ref_image_size": "max"},
        [("positive", "CONDITIONING", []), ("latent", "LATENT", [])], [port("clip", "CLIP")])
    replacement["size"] = [650, 850]
    by_id[4].clear(); by_id[4].update(replacement)
    apply = node(18, "MiniMaxH3RefModApply", "Apply H3 RefMod", 1780, 30,
        {"override": False, "retention": 1.0, "curve_direction": "constant",
         "scramble_seed": -1, "control_after_generate": "fixed", "curve_shape": "linear", "curve_value": 1.0,
         "graph_preset": "(none)", "scramble_mode": "shuffle", "scramble_keep": 1,
         "max_total_tokens": 0, "save_preset_as": ""},
        [("conditioning", "CONDITIONING", []), ("debug", "IMAGE", [])],
        [port("conditioning", "CONDITIONING"), port("mods", "H3_REF_MODS")])
    graph["nodes"].append(apply)
    apply["size"] = [430, 440]
    graph["nodes"].append(node(19, "PreviewAny", "Automatic character assignments (diagnostic)", 450, 1210, {},
                               [("STRING", "STRING", [])], [port("source", "STRING")]))
    connect(graph, 3, 0, 2, "clip")
    connect(graph, 2, 2, 4, "clip")
    connect(graph, 2, 0, 18, "mods")
    connect(graph, 2, 1, 19, "source")
    connect(graph, 4, 0, 18, "conditioning")
    connect(graph, 18, 0, 7, "conditioning")
    connect(graph, 4, 1, 11, "latent_image")
    by_id[1]["widgets_values"] = [
        "Select combined profiles in Load H3 RefMods; files from our earlier character extractor work too. "
        "Select your H3 checkpoint, minimax text encoder and decode VAEs. Keep copies=1 and scramble_seed=-1. "
        "The official node contains your complete Ref2VA prompt. mod_1 supplies Subject 1, mod_2 supplies Subject 2, etc. "
        "Write Subject tags and dialogue normally; visual/voice reference associations are inserted internally. "
        "voice_reference_N=1 selects the first recording in slot N; 2 selects the second; 0 explicitly uses all. "
        "Do not upload the same cached character refs again to the official node. Binding experiments are not required."]
    if count == 8:
        by_id[1]["widgets_values"][0] += " Eight-character experimental capacity test: routing is CPU-tested; voice accuracy is not established. Start with one image and one short clean voice sample per character."
    by_id[12]["widgets_values"] = ["minimax_h3_video_vae_fp16.safetensors"]
    by_id[13]["widgets_values"] = ["minimax_h3_audio_vae_fp32.safetensors"]
    by_id[17]["widgets_values"][0] = f"video/refmod_{count}_characters"
    # Keep the wider native prompt and new Apply node clear of the inherited
    # sampling/decode graph, with room for node titles and media previews.
    positions = {1: (20, -240), 3: (20, 70), 5: (2320, -230), 6: (2320, -20),
                 7: (2770, 50), 8: (2320, 210), 9: (2320, 430), 10: (2320, 620),
                 11: (2770, 290), 12: (3190, -170), 13: (3190, 540),
                 14: (3190, 30), 15: (3190, 290), 16: (3600, 30), 17: (3600, 340)}
    for ident, pos in positions.items():
        by_id[ident]["pos"] = list(pos)
    graph["extra"] = {"ds": {"scale": 0.4, "offset": [30, 270]}}
    write({1:"03_generate_one_character.json", 2:"04_generate_two_characters.json", 3:"05_generate_three_characters.json",
           8:"06_generate_eight_characters_experimental.json"}[count], graph)


def main():
    OUT.mkdir(exist_ok=True)
    extraction()
    extraction(video=True)
    for count in (1, 2, 3, 8):
        generation(count)
    # Old examples keep their connected output indices when the loader adds CLIP.
    path = ROOT / "examples/characters/07_import_upstream_refmods.json"
    graph = json.loads(path.read_text())
    for n in graph["nodes"]:
        if n["type"] == "MiniMaxH3RefModsLoader" and len(n["outputs"]) == 2:
            n["outputs"].append({"name": "clip", "type": "CLIP", "links": [], "slot_index": 2})
    path.write_text(json.dumps(graph, indent=2) + "\n")
    print(f"Wrote six workflows to {OUT}")


if __name__ == "__main__":
    main()
