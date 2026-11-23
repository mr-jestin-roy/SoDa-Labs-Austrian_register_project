"""Unit tests for the structured (C3) field-aware constrained decoder.

Runs without model weights or a GPU: only a tokenizer and torch are needed.
Prefers the real Qwen tokenizer (cached or downloadable); falls back to a
minimal hand-built mock tokenizer when neither network nor cache is
available, so the JSON tracker and logits math stay testable everywhere.

    venv/bin/python -m unittest test_logit_constrained -v
"""
import unittest

import torch

import schema
from logit_constrained import (JSONFieldTracker, TokenTrie,
                               make_structured_processor,
                               scan_vocab_token_sets)

NEG_INF = float("-inf")


# ── tokenizer acquisition ────────────────────────────────────────────────────

class MockTokenizer:
    """Greedy longest-match tokenizer over a tiny hand-built vocab, mimicking
    the byte-level-BPE properties the processor relies on: value strings open
    with a ' "' token, closing quotes come merged ('",', '"}'), and ⟦/⟧ are
    single tokens."""

    PIECES = [
        '{"', '"}', '",', '":', ' "', '": "', '"', "⟦", "⟧",
        "surname", "given_name", "place", "role", "date", "notes",
        "Hub", "er", "Maier", " Maier", " Hub", "Sankt", " Sankt",
        " Veit", "Veit", " an", " der", " Glan", "an", "der", "Glan",
        "Braut", " ", "-", "?", "{", "}", ":", ",", "[", "]", "\n", "\\",
    ] + [chr(c) for c in range(32, 127)]

    def __init__(self):
        seen: dict[str, int] = {}
        for p in self.PIECES:
            if p not in seen:
                seen[p] = len(seen)
        self._vocab = seen
        self._inv = {i: p for p, i in seen.items()}
        self._sorted = sorted(seen, key=len, reverse=True)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            for p in self._sorted:
                if text.startswith(p, i):
                    ids.append(self._vocab[p])
                    i += len(p)
                    break
            else:
                raise ValueError(f"untokenizable char {text[i]!r}")
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._inv[i] for i in ids)

    def __len__(self) -> int:
        return len(self._vocab)


def get_tokenizer():
    for name in ("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
                 "Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name), name
        except Exception:
            continue
    return MockTokenizer(), "mock"


TOKENIZER, TOKENIZER_NAME = get_tokenizer()


# ── shared fixtures ──────────────────────────────────────────────────────────

def make_proc(hard: bool, bias: float = 4.0):
    tries = {
        "surnames": TokenTrie.from_lexicon(["Huber", "Maier"], TOKENIZER),
        "given_names": TokenTrie.from_lexicon(["Johann"], TOKENIZER),
        "towns": TokenTrie.from_lexicon(["Sankt Veit an der Glan"], TOKENIZER),
        "occupations": TokenTrie.from_lexicon(["Bauer"], TOKENIZER),
    }
    return make_structured_processor(tries, TOKENIZER, bias=bias, hard=hard)


def step_scores(proc, generated: list[int]) -> torch.Tensor:
    """Replay `generated` one token at a time (as .generate() would) and
    return the processed scores for the NEXT position. A leading dummy token
    stands in for the prompt, whose tokens are never observed."""
    ids = [0] + generated
    scores = None
    for i in range(1, len(ids) + 1):
        scores = proc(torch.tensor([ids[:i]]),
                      torch.zeros(1, len(TOKENIZER)))
    return scores[0]


def enc(text: str) -> list[int]:
    return TOKENIZER.encode(text, add_special_tokens=False)


class TrackerTest(unittest.TestCase):
    """(a) entering/leaving value spans, keyed on the last seen key."""

    def spans(self, doc: str) -> list[str | None]:
        """active_category before consuming each char of `doc`."""
        tr = JSONFieldTracker()
        out = []
        for ch in doc:
            out.append(tr.active_category)
            tr.feed(ch)
        return out

    def test_surname_value_span_detected(self):
        doc = '{"role": "Braut", "surname": "Huber", "notes": "x"}'
        active = self.spans(doc)
        for i, ch in enumerate(doc):
            expect = "surnames" if i in range(doc.index("Huber"),
                                              doc.index('Huber"') + 6) else None
            self.assertEqual(active[i], expect,
                             f"char {i} ({ch!r}): {active[i]} != {expect}")

    def test_unmapped_and_key_strings_unconstrained(self):
        doc = '{"date": "1. Mai 1873", "surname": "M'
        active = self.spans(doc)
        self.assertTrue(all(a is None for a in active[:-1]))
        self.assertEqual(active[-1], "surnames")  # before the final 'M'

    def test_place_maps_to_towns_and_span_id_resets(self):
        tr = JSONFieldTracker()
        tr.feed('{"surname": "Huber", "place": "')
        self.assertEqual(tr.active_category, "towns")
        self.assertEqual(tr.span_id, 2)  # one span per constrained value

    def test_escaped_quote_stays_inside_value(self):
        tr = JSONFieldTracker()
        tr.feed('{"surname": "a\\"b')
        self.assertEqual(tr.active_category, "surnames")

    def test_oov_markers_toggle_suspension(self):
        tr = JSONFieldTracker()
        tr.feed('{"surname": "⟦Hu?')
        self.assertTrue(tr.suspended)
        self.assertIsNone(tr.active_category)
        tr.feed("⟧")
        self.assertFalse(tr.suspended)
        self.assertEqual(tr.active_category, "surnames")


class SoftBiasTest(unittest.TestCase):
    """(b) soft bias applied only inside constrained spans."""

    def test_no_bias_outside_value_span(self):
        proc = make_proc(hard=False)
        scores = step_scores(proc, enc('{"surname"'))  # next token: '":'
        self.assertTrue(torch.all(scores == 0),
                        "logits must be untouched outside value strings")

    def test_bias_on_trie_continuation_inside_span(self):
        proc = make_proc(hard=False, bias=4.0)
        scores = step_scores(proc, enc('{"surname": "Hub'))
        er = enc("Huber")[len(enc("Hub")):][0]     # continuation token 'er'
        self.assertEqual(scores[er].item(), 4.0)
        # off-trie tokens keep their original score, quotes stay available
        self.assertEqual(scores[enc('",')[0]].item(), 0.0)
        self.assertEqual(scores[enc("Braut")[0]].item(), 0.0)

    def test_word_start_tokens_biased_at_span_open(self):
        proc = make_proc(hard=False, bias=4.0)
        scores = step_scores(proc, enc('{"surname": "'))
        hub = enc("Huber")[0]
        self.assertEqual(scores[hub].item(), 4.0)
        self.assertEqual(scores[enc("Braut")[0]].item(), 0.0)


class HardModeTest(unittest.TestCase):
    """(c) closing quote allowed at end-of-word, blocked mid-word."""

    def test_close_quote_allowed_at_end_of_word(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "') + enc("Huber"))
        for q in ('"', '",', '"}'):
            self.assertNotEqual(scores[enc(q)[0]].item(), NEG_INF,
                                f"{q!r} must survive at end-of-word")
        self.assertEqual(scores[enc("Braut")[0]].item(), NEG_INF)

    def test_close_quote_masked_mid_word(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "Hub'))
        er = enc("Huber")[len(enc("Hub")):][0]
        self.assertNotEqual(scores[er].item(), NEG_INF)
        self.assertEqual(scores[enc('",')[0]].item(), NEG_INF)
        self.assertEqual(scores[enc("Braut")[0]].item(), NEG_INF)

    def test_close_quote_allowed_before_any_word(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "'))
        self.assertNotEqual(scores[enc('",')[0]].item(), NEG_INF)

    def test_multiword_entry_follows_continuous_trie_path(self):
        # "Sankt Veit an der Glan" is ONE lexicon entry: its internal
        # spaces live inside the tokenised path (' Veit', ' an', …)
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"place": "') + enc("Sankt"))
        self.assertNotEqual(scores[enc(" Veit")[0]].item(), NEG_INF,
                            "mid-entry continuation must stay allowed")
        self.assertNotEqual(scores[enc(" ")[0]].item(), NEG_INF,
                            "separator must always stay available")
        self.assertEqual(scores[enc("Braut")[0]].item(), NEG_INF)

    def test_space_restarts_trie_for_double_surname(self):
        # "Huber Maier": two lexicon entries joined by a separator token,
        # after which the trie restarts from the root
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "')
                             + enc("Huber") + [enc(" ")[0]])
        self.assertNotEqual(scores[enc("Maier")[0]].item(), NEG_INF,
                            "trie must restart at the next word")
        self.assertEqual(scores[enc("Braut")[0]].item(), NEG_INF)

    def test_untouched_outside_span(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"notes": "free text'))
        self.assertTrue(torch.all(scores == 0),
                        "unmapped fields decode unconstrained even in --hard")


class OOVEscapeTest(unittest.TestCase):
    """(d) ⟦ escape hatch suspends constraints until after ⟧."""

    def test_oov_open_always_allowed_in_hard_mode(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "Hub'))
        self.assertNotEqual(scores[enc("⟦")[0]].item(), NEG_INF)

    def test_constraints_suspended_inside_oov_span(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "') + enc("⟦"))
        self.assertTrue(torch.all(scores == 0),
                        "logits must pass through untouched after ⟦")
        # …and stay suspended for arbitrary OOV content
        scores = step_scores(make_proc(hard=True),
                             enc('{"surname": "⟦Hu?'))
        self.assertTrue(torch.all(scores == 0))

    def test_constraints_resume_after_oov_close(self):
        proc = make_proc(hard=True)
        scores = step_scores(proc, enc('{"surname": "⟦Hu?er⟧'))
        # back at a word boundary: quote allowed, arbitrary content masked
        self.assertNotEqual(scores[enc('",')[0]].item(), NEG_INF)
        self.assertEqual(scores[enc("Braut")[0]].item(), NEG_INF)


class VocabScanTest(unittest.TestCase):
    def test_quote_and_separator_sets(self):
        quote_ids, sep_ids = scan_vocab_token_sets(TOKENIZER)
        self.assertIn(enc('"')[0], quote_ids)
        self.assertIn(enc('",')[0], quote_ids)
        self.assertIn(enc(" ")[0], sep_ids)
        self.assertIn(enc("-")[0], sep_ids)
        self.assertNotIn(enc(" ")[0], quote_ids)


if __name__ == "__main__":
    print(f"tokenizer under test: {TOKENIZER_NAME}")
    unittest.main(verbosity=2)
