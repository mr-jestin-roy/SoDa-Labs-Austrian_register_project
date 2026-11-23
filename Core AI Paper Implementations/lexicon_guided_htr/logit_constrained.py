"""Optional Stage C — lexicon-constrained decoding at the logit level.

Implements the core mechanism of Dictionary-Guided Scene Text Recognition
(CVPR 2021) on top of a locally loaded Qwen3-VL: a token-trie built from the
parish-scoped lexicon softly biases (or hard-masks) the decoder logits, so
ambiguous Kurrent strokes are *structurally* steered toward attested names
during generation instead of being fixed afterwards.

Two decoding modes:

  * flat (default)      — PASS1_PROMPT free-text transcription with ONE trie
                          merged over every lexicon category, biased at every
                          step (the original behaviour).
  * --structured (C3)   — schema.STRUCTURED_PROMPT JSON transcription with
                          one trie PER lexicon category. A lightweight
                          quote/brace/escape-aware JSON scanner tracks which
                          field's string value the decoder is currently
                          inside; only the string VALUES of fields listed in
                          schema.FIELD_LEXICON_MAP are constrained, and each
                          with the matching category trie (surname →
                          surnames, place → towns, …). Everything else —
                          JSON syntax, keys, unmapped fields like date/notes
                          — decodes untouched. The ⟦…⟧ uncertainty markers
                          of the frozen prompt double as an OOV escape
                          hatch: once ⟦ is emitted, constraints are
                          suspended for that value until after ⟧.

This requires local model weights (GPU box / Nectar HPC, or a Mac with
enough unified memory) — it cannot run against a remote vLLM endpoint
because logits processors execute inside the generation loop.

Usage (on a machine with the weights):
    venv/bin/python logit_constrained.py --image page.jpg \
        --parish "Althofen (Kärnten)" \
        --model Qwen/Qwen3-VL-8B-Instruct --bias 4.0 [--structured] [--hard]
"""
import argparse
from pathlib import Path

import config
import schema
from kg_retrieval import LexiconKG

# Opening / closing marker of the frozen prompt's uncertainty convention
# ("surname": "⟦Hu?er⟧") — reused in structured mode as the OOV escape hatch.
OOV_OPEN = "⟦"
OOV_CLOSE = "⟧"


class TokenTrie:
    """Prefix trie over tokenised lexicon entries (each entry tokenised with
    and without a leading space, since BPE merges the space into the first
    token)."""

    def __init__(self):
        self.root: dict = {}

    def insert(self, token_ids: list[int]):
        node = self.root
        for t in token_ids:
            node = node.setdefault(t, {})
        node[-1] = True  # end-of-word marker

    @classmethod
    def from_lexicon(cls, words: list[str], tokenizer) -> "TokenTrie":
        trie = cls()
        for w in words:
            for variant in (w, " " + w):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    trie.insert(ids)
        return trie


def make_processor(trie: TokenTrie, bias: float, hard: bool):
    """LogitsProcessor tracking all live trie states.

    At each step the allowed set is: continuations of any active trie path,
    plus every word-start token (so the model can always begin a new lexicon
    word or — in soft mode — emit unconstrained syntax tokens like spaces,
    digits and Markdown pipes at their original score)."""
    import torch
    from transformers import LogitsProcessor

    class LexiconTrieLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self.states: list[dict] = []
            self.prev_len: int | None = None

        def __call__(self, input_ids, scores):
            if self.prev_len is not None and input_ids.shape[1] > self.prev_len:
                last = input_ids[0, -1].item()
                nxt = [s[last] for s in self.states + [trie.root] if last in s]
                self.states = [s for s in nxt if any(k != -1 for k in s)]
            self.prev_len = input_ids.shape[1]

            boosted = set(trie.root.keys())
            for s in self.states:
                boosted.update(k for k in s if k != -1)
            boosted.discard(-1)
            if not boosted:
                return scores
            idx = torch.tensor(sorted(boosted), device=scores.device)
            if hard and self.states:
                masked = torch.full_like(scores, float("-inf"))
                masked[:, idx] = scores[:, idx]
                return masked
            scores[:, idx] += bias
            return scores

    return LexiconTrieLogitsProcessor()


# ── Structured (C3) mode ─────────────────────────────────────────────────────

class JSONFieldTracker:
    """Minimal incremental JSON scanner for the frozen PAGE_SCHEMA output.

    Not a full JSON parser: just enough quote/escape-aware state to know
    (1) whether we are inside a string, (2) whether that string is a key or
    a value (a string opened right after ':' is a value), (3) which key the
    current value belongs to, and (4) whether the value is inside a ⟦…⟧
    uncertainty span (constraints suspended). Torch-free and tokenizer-free
    so it can be unit-tested on plain character streams.
    """

    def __init__(self, field_map: dict[str, str] | None = None):
        self.field_map = dict(schema.FIELD_LEXICON_MAP
                              if field_map is None else field_map)
        self.in_string = False
        self.is_value = False          # current string is a value (vs a key)
        self.escape = False
        self.suspended = False         # inside ⟦…⟧ within the current value
        self.current_key: str | None = None
        self.span_id = 0               # bumps when a constrained value opens
        self._buf: list[str] = []      # content of the string being read
        self._last_sig = ""            # last non-whitespace char outside strings

    @property
    def active_category(self) -> str | None:
        """Lexicon category to constrain with right now, or None."""
        if (self.in_string and self.is_value and not self.suspended
                and self.current_key in self.field_map):
            return self.field_map[self.current_key]
        return None

    def feed(self, text: str) -> None:
        for ch in text:
            if self.in_string:
                if self.escape:
                    self.escape = False
                    self._buf.append(ch)
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
                    if not self.is_value:
                        self.current_key = "".join(self._buf)
                    self.suspended = False
                    self._last_sig = ch
                else:
                    if ch == OOV_OPEN:
                        self.suspended = True
                    elif ch == OOV_CLOSE:
                        self.suspended = False
                    self._buf.append(ch)
            elif ch == '"':
                self.in_string = True
                self.is_value = self._last_sig == ":"
                self.escape = False
                self._buf = []
                if self.is_value and self.current_key in self.field_map:
                    self.span_id += 1
            elif not ch.isspace():
                self._last_sig = ch


def scan_vocab_token_sets(tokenizer) -> tuple[set[int], set[int]]:
    """(closing-quote token ids, pure space/hyphen separator token ids).

    Byte-level BPE vocab strings keep printable ASCII verbatim (space is
    'Ġ', newline 'Ċ'), so a string scan over the vocab is cheap and exact
    for the structural characters we care about.
    """
    quote_ids: set[int] = set()
    sep_ids: set[int] = set()
    for tok_str, tid in tokenizer.get_vocab().items():
        s = tok_str.replace("Ġ", " ").replace("Ċ", "\n")
        if s.startswith('"'):
            quote_ids.add(tid)
        elif s and set(s) <= {" ", "-"}:
            sep_ids.add(tid)
    return quote_ids, sep_ids


def make_structured_processor(tries: dict[str, "TokenTrie"], tokenizer,
                              bias: float, hard: bool,
                              field_map: dict[str, str] | None = None):
    """Field-aware LogitsProcessor for schema.STRUCTURED_PROMPT output.

    Per generated token: the token is detokenised individually (cached; ASCII
    structural bytes survive byte-level per-token decoding, and the only
    multi-byte characters that matter structurally, ⟦ and ⟧, are single
    tokens in the Qwen vocabulary), fed to a JSONFieldTracker, and — while
    inside the string value of a FIELD_LEXICON_MAP field — matched against
    that field's category trie. Trie states reset at every new value span
    and restart per word after space/hyphen separators.

      soft (default) : +bias on trie-consistent continuations, nothing else
                       touched, so quotes/⟦/separators stay available at
                       their natural scores.
      --hard         : mask everything except trie continuations, ALWAYS
                       also allowing ⟦ (OOV escape), space/hyphen separators
                       (multi-word values), and — when at an end-of-word
                       trie node or before any word has started — tokens
                       beginning with the closing quote.

    Outside constrained value spans logits pass through untouched.
    Assumes batch size 1, matching the flat-mode processor.
    """
    import torch
    from transformers import LogitsProcessor

    fmap = dict(schema.FIELD_LEXICON_MAP if field_map is None else field_map)
    quote_ids, sep_ids = scan_vocab_token_sets(tokenizer)
    oov_ids = set(tokenizer.encode(OOV_OPEN, add_special_tokens=False))

    class FieldAwareTrieLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self.tracker = JSONFieldTracker(fmap)
            self.states: list[dict] = []   # live trie nodes for current word
            self.prev_len: int | None = None
            self._span_id = 0
            self._piece_cache: dict[int, str] = {}

        def _piece(self, tid: int) -> str:
            if tid not in self._piece_cache:
                self._piece_cache[tid] = tokenizer.decode([tid])
            return self._piece_cache[tid]

        def observe(self, tid: int) -> None:
            """Consume one generated token: update JSON context + trie states."""
            was = self.tracker.active_category
            self.tracker.feed(self._piece(tid))
            now = self.tracker.active_category
            if self.tracker.span_id != self._span_id:
                # a new constrained value just opened (any content merged
                # into the opening-quote token is deliberately not matched;
                # hard mode never produces such tokens mid-span)
                self._span_id = self.tracker.span_id
                self.states = []
            elif was and now == was:
                if tid in sep_ids:
                    self.states = []       # word boundary: restart the trie
                else:
                    node_list = self.states + [tries[now].root]
                    self.states = [s[tid] for s in node_list if tid in s]
            elif now is None:
                self.states = []           # span closed / ⟦ suspension began

        def constrained_token_sets(self) -> tuple[set[int], bool] | None:
            """(trie-consistent token ids, at-word-boundary) or None when the
            current position is unconstrained. Pure — no torch needed."""
            cat = self.tracker.active_category
            if cat is None:
                return None
            conts: set[int] = set()
            at_boundary = not self.states
            for s in self.states:
                conts.update(k for k in s if k != -1)
                if -1 in s:
                    at_boundary = True
            if at_boundary:               # word start / end: word-start tokens
                conts.update(k for k in tries[cat].root if k != -1)
            return conts, at_boundary

        def __call__(self, input_ids, scores):
            if self.prev_len is not None and input_ids.shape[1] > self.prev_len:
                self.observe(input_ids[0, -1].item())
            self.prev_len = input_ids.shape[1]

            sets_ = self.constrained_token_sets()
            if sets_ is None:
                return scores
            boost, at_boundary = sets_
            if not hard:
                if boost:
                    idx = torch.tensor(sorted(boost), device=scores.device)
                    scores[:, idx] += bias
                return scores
            allowed = boost | oov_ids | sep_ids
            if at_boundary:
                allowed |= quote_ids
            idx = torch.tensor(sorted(allowed), device=scores.device)
            masked = torch.full_like(scores, float("-inf"))
            masked[:, idx] = scores[:, idx]
            return masked

    return FieldAwareTrieLogitsProcessor()


def build_category_tries(scoped, tokenizer) -> dict[str, TokenTrie]:
    """One TokenTrie per lexicon category referenced by FIELD_LEXICON_MAP."""
    category_words = {
        "surnames": scoped.all_surnames,
        "given_names": scoped.given_names,
        "towns": scoped.nearby_towns,
        "occupations": scoped.occupations,
    }
    return {cat: TokenTrie.from_lexicon(words, tokenizer)
            for cat, words in category_words.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--parish", default=config.DEFAULT_PARISH)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--bias", type=float, default=4.0,
                    help="additive logit boost for lexicon-consistent tokens")
    ap.add_argument("--hard", action="store_true",
                    help="hard-mask: once inside a lexicon word, only trie "
                         "continuations are allowed (strictest CVPR setting)")
    ap.add_argument("--structured", action="store_true",
                    help="C3 mode: STRUCTURED_PROMPT JSON output with one "
                         "trie per lexicon category, applied only inside the "
                         "string values of FIELD_LEXICON_MAP fields")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              LogitsProcessorList)
    from pipeline import PASS1_PROMPT

    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon(args.parish)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto")

    if args.structured:
        tries = build_category_tries(scoped, processor.tokenizer)
        print("Field-aware constraining for "
              f"{scoped.matched_town!r} (schema v{schema.SCHEMA_VERSION}): "
              + ", ".join(f"{f}→{c}" for f, c in
                          schema.FIELD_LEXICON_MAP.items()))
        logits_processor = LogitsProcessorList(
            [make_structured_processor(tries, processor.tokenizer,
                                       args.bias, args.hard)])
        prompt = schema.STRUCTURED_PROMPT
    else:
        vocab = (scoped.all_surnames + scoped.given_names
                 + scoped.nearby_towns + scoped.occupations)
        print(f"Constraining toward {len(vocab)} lexicon entries "
              f"for {scoped.matched_town!r}")
        trie = TokenTrie.from_lexicon(vocab, processor.tokenizer)
        logits_processor = LogitsProcessorList(
            [make_processor(trie, args.bias, args.hard)])
        prompt = PASS1_PROMPT

    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(args.image).convert("RGB")},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False,
                             logits_processor=logits_processor)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]
    print(text)


if __name__ == "__main__":
    main()
