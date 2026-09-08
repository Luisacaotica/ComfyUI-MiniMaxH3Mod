"""Build reference presentation and payload in one stable, shared order."""
from __future__ import annotations

from dataclasses import replace

from .character_core import H3CharacterMod
from .character_references import assemble_references


def select_voice(mod: H3CharacterMod, selection="all"):
    mod.validate()
    if selection not in ("all", "first", "second"):
        raise ValueError("Unknown voice selection.")
    if selection == "all":
        return mod
    voices = [r for r in mod.references if r.audio is not None]
    index = 0 if selection == "first" else 1
    if index >= len(voices):
        raise ValueError(f"{mod.name} has no {selection} voice reference.")
    selected = voices[index]
    refs = [r for r in mod.references if r.audio is None or r is selected]
    out = replace(mod, references=refs)
    out.validate()
    return out


def build_character_conditioning(cast, scene, language="English", max_tokens=0, turns=()):
    """cast: [(H3CharacterMod, dialogue, emotion, delivery), ...].

    No tensor averaging, artificial speaker embeddings, or global model state.
    Native presentation order is images, paired videos, standalone audio.
    Every label and reference block is emitted by the same traversal.
    """
    assembled = assemble_references([row[0] for row in cast], max_tokens)
    labels, pairing = assembled.labels, assembled.pairing
    presentation, blocks = assembled.items, assembled.blocks
    token_count, duration = assembled.tokens, assembled.audio_seconds
    speaking_order = [t.owner for t in turns] if turns else [i for i, row in enumerate(cast, 1) if str(row[1]).strip()]
    speaker_ids = {owner: i for i, owner in enumerate(speaking_order, 1)}
    windows = {t.owner: t for t in turns}
    definitions, retention, actions, silent, debug = [], [], {}, [], []
    for i, ((mod, dialogue, emotion, delivery), mapping) in enumerate(zip(cast, labels), 1):
        subject = f"<Subject {i}>"
        visuals, voices = ", ".join(mapping["visual"]), ", ".join(mapping["audio"])
        appearance = mod.description.strip() or mod.name
        definitions.append(f"{subject} is {appearance}, shown in {visuals}.")
        retention.append(f"{subject} (appears in [Shot 1]): fully_preserved - preserve the character's visual identity.")
        if str(dialogue).strip():
            speaker = f"{subject} (S{speaker_ids[i]})"
            definitions.append(f"{voices} provide voice-timbre references for {speaker} only.")
            interval = windows.get(i)
            timing = f"Between {interval.start:.3f} and {interval.end:.3f} seconds, " if interval else ""
            words = str(dialogue).strip()
            if words[-1] not in ".?!":
                words += "."
            actions[i] = (f"{timing}{speaker}, {appearance}, speaks using the vocal identity from {voices}, "
                          f"with {str(emotion).strip() or 'neutral'} emotion and "
                          f"{str(delivery).strip() or 'natural'} delivery: <d>[{language}] {words}</d> "
                          "The other characters listen with their mouths at rest. The speaker closes their mouth after the line.")
        else:
            definitions.append(f"{voices} belong to {subject}, who has no spoken line in this scene.")
            silent.append(f"{subject} remains silent.")
        for audio_label in mapping["audio"]:
            retention.append(f"{audio_label}: reference - use only {subject}'s vocal identity; "
                             "generate the new words and requested emotion without reusing the recorded performance.")
        debug.append(f"{mod.name} -> {subject}: {visuals}; voice {voices}")
    if not any(str(row[1]).strip() for row in cast):
        raise ValueError("Enter new dialogue for at least one character.")
    prompt = (
        "subject_definitions:\n" + "\n".join(definitions + pairing) +
        "\n\nsummary:\n[reference generation + audio reference]\n" + str(scene).strip() +
        "\nEach character uses their assigned voice for the new dialogue.\n\nretention_analysis:\n" +
        "\n".join(retention) + "\n\ndetailed_description:\n[Shot 1]\n" + str(scene).strip() + "\n" +
        "\n".join([actions[i] for i in speaking_order] + silent) +
        "\nCharacters speak in the order listed. Only the current speaker moves their mouth to speak." +
        "\n\noverall_soundscape:\nEnvironmental sounds follow the scene and remain below the dialogue. "
        "Only the specified characters speak.\n\nnon_diegetic_music:\nN/A"
    )
    info = "\n".join(debug) + f"\n{token_count:,} reference tokens; {duration:.2f}s reference audio"
    if turns:
        info += "\nSpeaking windows: " + "; ".join(f"character {t.owner}: {t.start:.3f}-{t.end:.3f}s" for t in turns)
    return prompt, presentation, blocks, info
