"""Rebuild import/comparison examples from the existing complete sampling graph."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples/characters"


def write(name, graph):
    (OUT / name).write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def node(number, kind, title, x, y, widgets, outputs=(), inputs=()):
    return {"id": number, "type": kind, "pos": [x, y], "size": [420, 300], "flags": {},
            "order": number - 1, "mode": 0, "inputs": list(inputs),
            "outputs": [{"name": name, "type": type_, "links": list(links), "slot_index": i}
                        for i, (name, type_, links) in enumerate(outputs)], "title": title,
            "properties": {"Node name for S&R": kind, "test_widget_names": list(widgets)},
            "widgets_values": list(widgets.values())}


def main():
    source = json.loads((OUT / "06_binding_three_characters.json").read_text())
    for count in (1, 2, 3):
        graph = copy.deepcopy(source)
        removed = {4} if count == 2 else {3, 4} if count == 1 else set()
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] not in removed]
        graph["links"] = [link for link in graph["links"] if link[1] not in removed and link[3] not in removed]
        valid_links = {link[0] for link in graph["links"]}
        for n in graph["nodes"]:
            for port in n.get("inputs", []):
                if port["link"] not in valid_links:
                    port["link"] = None
            for port in n.get("outputs", []):
                port["links"] = [link for link in port["links"] if link in valid_links]
        by_id = {n["id"]: n for n in graph["nodes"]}
        windows = [(0.5, 2.5), (3.0, 5.0), (6.0, 8.5)][:count]
        lines = ["The old bridge is still standing.", "Then we can cross before sunset.", "I will bring the lantern."][:count]
        definitions, retention, actions = [], [], []
        for i, ((start, end), dialogue) in enumerate(zip(windows, lines), 1):
            definitions.extend([f"<Subject {i}> is character {i}, shown in <Video {i}>.",
                                f"<Audio {i}> is the voice-timbre reference for <Subject {i}> (S{i})."])
            retention.append(f"<Audio {i}>: reference - retain vocal identity while generating the new words.")
            actions.append(f"Between {start:.1f} and {end:.1f} seconds, <Subject {i}> (S{i}) says "
                           f"<d>[English] {dialogue}</d> with calm, natural delivery, then closes their mouth.")
        prompt = ("subject_definitions:\n" + "\n".join(definitions) +
                  "\n\nsummary:\n[reference generation + audio reference] A quiet conversation with new dialogue.\n\n"
                  "retention_analysis:\n" + "\n".join(retention) +
                  "\n\ndetailed_description:\n[Shot 1] A stationary eye-level camera frames the characters "
                  "side by side in a quiet room under soft daylight. Their individual reference appearances "
                  "remain consistent. They speak one at a time; the others listen with closed mouths.\n" +
                  "\n".join(actions) + "\n\noverall_soundscape:\nQuiet room ambience below the dialogue."
                  "\n\nnon_diegetic_music:\nN/A")
        cond = by_id[6]
        cond.update(type="H3CharacterReferenceConditioning", title="Full prompt + reference diagnostics", size=[660, 710])
        values = {"width": 768, "height": 512, "length": {1: 124, 2: 175, 3: 243}[count],
                  "max_reference_tokens": 0, "prompt": prompt,
                  "turn_schedule": "\n".join(f"{i},{a},{b}" for i, (a, b) in enumerate(windows, 1)),
                  "reference_presentation": "native", "av_layout": "paired"}
        cond["properties"] = {"Node name for S&R": cond["type"], "test_widget_names": list(values)}
        cond["widgets_values"] = list(values.values())
        by_id[11]["widgets_values"] = [101, "fixed"]
        by_id[19]["widgets_values"][0] = f"video/voice_test_{count}characters"
        by_id[1]["widgets_values"] = [
            f"{count}-character controlled test. Use one synchronized speaking-video profile per character, "
            "voice_reference=first. Select all model files. Baseline: native / paired / binding off. "
            "See VOICE_TESTING.md for the five variants and results sheet. Keep prompt, seed, timeline, "
            "profiles, models and sampling settings fixed. If using image+audio profiles, replace Video "
            "labels with the actual Picture labels in reference_info; paired versus separate then has no effect. "
            "Copy case_fingerprint from the console. Save Video embeds the workflow. This prompt is only "
            "a short diagnostic example; replace the dialogue before starting a matched test set."
        ]
        write(f"{7 + count:02d}_compare_{count}_characters.json", graph)
    widgets = {"show_info": True}
    for i in range(1, 9):
        widgets.update({f"mod_{i}": "(none)", f"strength_{i}": 1.0, f"copies_{i}": 1})
    widgets["max_total_tokens"] = 0
    graph = {"last_node_id": 4, "last_link_id": 2, "version": 0.4, "groups": [], "config": {}, "extra": {},
             "nodes": [
                 node(1, "Note", "Import existing upstream RefMods", 20, -250, {"text":
                     "Select one character's encode-mode image RefMod and audio RefMod in the loader. "
                     "Keep strength=1 and copies=1. Load the matching original image. Queue to save a "
                     "Character file, then use workflows 08-10. For multiple separate image RefMods, "
                     "connect matching image_2/image_3 or image batches in loader order. Video/pooled "
                     "stacks must be re-extracted from source; filenames cannot recover synchronization."}),
                 node(2, "MiniMaxH3RefModsLoader", "Select matching visual + voice RefMods", 20, 30, widgets,
                      [("mods", "H3_REF_MODS", [1]), ("prompt_hint", "STRING", []), ("clip", "CLIP", [])]),
                 node(3, "LoadImage", "Matching original source image", 480, 30, {"image": "character_a.png"},
                      [("IMAGE", "IMAGE", [2]), ("MASK", "MASK", [])]),
                 node(4, "ImportH3RefModsAsCharacter", "Save reusable character", 980, 30,
                      {"name": "character_a", "description": "", "save": True, "overwrite": False},
                      [("character", "H3_CHARACTER", []), ("info", "STRING", [])],
                      [{"name": "mods", "type": "H3_REF_MODS", "link": 1},
                       {"name": "image_1", "type": "IMAGE", "link": 2},
                       {"name": "image_2", "type": "IMAGE", "link": None},
                       {"name": "image_3", "type": "IMAGE", "link": None}])],
             "links": [[1, 2, 0, 4, 0, "H3_REF_MODS"], [2, 3, 0, 4, 1, "IMAGE"]]}
    write("07_import_upstream_refmods.json", graph)
    print("Wrote importer and one/two/three-character comparison workflows.")


if __name__ == "__main__":
    main()
