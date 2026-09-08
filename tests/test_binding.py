"""Numerical/ownership tests for the experimental all-H3 attention control."""
import importlib
import math
import types
import unittest

import torch
import torch.nn.functional as F

from test_characters import core, conditioning, package

binding = importlib.import_module(package.__name__ + ".character_binding")


def make_case():
    image = core.CharacterReference("image", visual=torch.ones(1, 24, 1, 4, 4),
        frames=torch.zeros(1, 320, 320, 3, dtype=torch.uint8))
    audio = core.CharacterReference("audio", audio=torch.ones(1, 32, 2, 9), duration=5 / 24)
    pair = core.CharacterReference("video_audio", visual=torch.full((1, 24, 2, 4, 4), 2.),
        audio=torch.full((1, 32, 2, 9), 2.), frames=torch.zeros(1, 320, 320, 3, dtype=torch.uint8),
        timestamps=torch.tensor([0.]), duration=5 / 24)
    cast = [(core.H3CharacterMod("A", [image, audio]), "A line", "calm", "slow"),
            (core.H3CharacterMod("B", [pair]), "B line", "happy", "quick")]
    turns = binding.parse_turns("1,0,0.4\n2,0.5,0.9", cast, 22 / 24)
    _, _, refs, _ = conditioning.build_character_conditioning(cast, "A room", turns=turns)
    spec = binding.make_binding(refs, turns, 22)
    # Exact native packing for this mixed cast: images, paired AV, standalone A audio.
    layout = types.SimpleNamespace(signature=(3, 7, 4, 4, 37), seq_len=153, segments=[
        (0, 3, "text"), (3, 7, "ref_img"), (7, 25, "ref_audio"), (25, 33, "ref_img"),
        (33, 51, "ref_audio"), (51, 125, "audio"), (125, 153, "video")])
    return cast, refs, spec, layout


def attention(q, k, v, heads, **kwargs):
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def dense_bias(groups, length):
    result = torch.zeros(length, length)
    for group in groups:
        for a, b, value in group.keys:
            result[group.start:group.stop, a:b] += value
        if group.weights:
            result[group.start:group.stop] *= torch.tensor(group.weights)[:, None]
    return result


class TurnTests(unittest.TestCase):
    def test_schedule_validation(self):
        cast, *_ = make_case()
        self.assertEqual(binding.parse_turns("", cast, 1), ())
        for schedule in ("1,0,0.6\n2,0.5,0.9", "1,0,0.5", "3,0,0.5", "1,nan,0.5",
                         "1,0,2\n2,2,3", "1,0,0.4\n1,0.5,0.9", "not CSV"):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                binding.parse_turns(schedule, cast, 1)

    def test_actual_speaking_order_sets_speaker_ids(self):
        cast, *_ = make_case()
        turns = binding.parse_turns("1,0.5,0.9\n2,0,0.4", cast, 1)
        prompt, _, _, _ = conditioning.build_character_conditioning(cast, "Room", turns=turns)
        self.assertIn("voice-timbre references for <Subject 2> (S1)", prompt)
        self.assertIn("voice-timbre references for <Subject 1> (S2)", prompt)
        self.assertLess(prompt.index("<d>[English] B line."), prompt.index("<d>[English] A line."))
        for section in ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                        "overall_soundscape:", "non_diegetic_music:"):
            self.assertIn(section, prompt)

    def test_silent_character_does_not_consume_speaker_id(self):
        cast, *_ = make_case()
        cast[0] = (cast[0][0], "", "", "")
        prompt, _, _, _ = conditioning.build_character_conditioning(cast, "Room")
        self.assertIn("<Subject 2> (S1)", prompt)
        self.assertNotIn("(S2)", prompt)
        with self.assertRaises(ValueError):
            binding.parse_turns("1,0,0.4\n2,0.5,0.9", cast, 1)


class BindingTests(unittest.TestCase):
    def test_cross_modal_bias_uses_owners_after_native_reordering(self):
        _, refs, spec, layout = make_case()
        groups = binding.build_bias_groups(layout, refs, spec, 0.5, 0, 0)
        mask = dense_bias(groups, layout.seq_len)
        # Paired B audio reads B visual more strongly, A visual less strongly.
        self.assertTrue(torch.all(mask[7:25, 25:33] == 0.5))
        self.assertTrue(torch.all(mask[7:25, 3:7] == -0.5))
        # Standalone A audio is associated with A's image, not B's video.
        self.assertTrue(torch.all(mask[33:51, 3:7] == 0.5))
        self.assertTrue(torch.all(mask[33:51, 25:33] == -0.5))
        self.assertEqual(float(mask[51:].abs().sum()), 0)
        self.assertEqual(float(mask[:3].abs().sum()), 0)

    def test_stereo_windows_and_pauses(self):
        _, refs, spec, layout = make_case()
        groups = binding.build_bias_groups(layout, refs, spec, 0, 0.8, 0)
        mask = dense_bias(groups, layout.seq_len)
        left, right = mask[51:88], mask[88:125]
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.all(left[:16, 33:51] == 0.8))
        self.assertTrue(torch.all(left[:16, 7:25] == -0.8))
        self.assertEqual(float(left[16:20].abs().sum()), 0)  # 0.4-0.5s pause
        self.assertTrue(torch.all(left[20:36, 7:25] == 0.8))
        self.assertEqual(float(mask[125:].abs().sum()), 0)  # target video is not directly biased

    def test_window_fade(self):
        _, refs, spec, layout = make_case()
        groups = binding.build_bias_groups(layout, refs, spec, 0, 1, 0.1)
        first = groups[0]
        self.assertAlmostEqual(first.weights[0], 0.125)
        self.assertEqual(first.weights[5], 1)
        self.assertAlmostEqual(first.weights[-1], 0.125)

    def test_wrong_conditioning_or_layout_fails(self):
        _, refs, spec, layout = make_case()
        wrong = [dict(r) for r in refs]
        wrong[0][binding.OWNER_KEY] = 2
        with self.assertRaises(ValueError):
            binding.build_bias_groups(layout, wrong, spec, 1, 1, 0)
        wrong = [dict(r) for r in refs]
        binding.make_binding(wrong, spec.turns, 22)  # same shapes, different conditioning run
        with self.assertRaises(ValueError):
            spec.validate_refs(wrong)
        layout.segments[1] = (3, 8, "ref_img")
        with self.assertRaises(ValueError):
            binding.build_bias_groups(layout, refs, spec, 1, 1, 0)

    def test_chunked_attention_equals_dense_reference_and_preserves_inputs(self):
        _, refs, spec, layout = make_case()
        groups = binding.build_bias_groups(layout, refs, spec, 0.5, 0.8, 0.1)
        generator = torch.Generator().manual_seed(15)
        q, k, v = [torch.randn(1, 2, layout.seq_len, 8, generator=generator) for _ in range(3)]
        saved = [x.clone() for x in (q, k, v)]
        mask = dense_bias(groups, layout.seq_len)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(1, layout.seq_len, 16)
        baseline = attention(q, k, v, 2)
        for chunk in (1, 7, 64):
            out = binding.biased_attention(attention, q, k, v, 2, groups, chunk, skip_reshape=True)
            self.assertTrue(torch.allclose(out, expected, atol=2e-6))
            self.assertTrue(torch.equal(out[:, 125:], baseline[:, 125:]))
        for old, new in zip(saved, (q, k, v)):
            self.assertTrue(torch.equal(old, new))

    def test_bias_has_measurable_direction_and_zero_is_original(self):
        # With equal QK logits, +/- b changes the odds by exp(2*b).
        q = k = torch.zeros(1, 1, 2, 2)
        v = torch.tensor([[[[1., 0.], [0., 1.]]]])
        group = binding.BiasGroup(0, 1, ((0, 1, 0.5), (1, 2, -0.5)))
        out = binding.biased_attention(attention, q, k, v, 1, [group], skip_reshape=True)
        self.assertAlmostEqual(float(out[0, 0, 0] / out[0, 0, 1]), math.exp(1), places=5)
        expected = object()
        self.assertIs(binding.biased_attention(lambda *a, **k: expected, q, k, v, 1, []), expected)


if __name__ == "__main__":
    unittest.main()
