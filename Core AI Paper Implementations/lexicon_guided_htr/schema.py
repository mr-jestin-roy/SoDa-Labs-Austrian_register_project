"""Frozen structured-output schema for register transcription (C3).

Single source of truth shared by three consumers:
  * pipeline.py            — the instruction prompt sent to the base VLM now
  * logit_constrained.py   — which JSON fields get trie-constrained decoding
  * (future) LoRA formatter — training targets must use the IDENTICAL prompt
                              and JSON layout, or the constraint trigger logic
                              breaks (see C3 coupling note in README).

Do not edit field names or the prompt casually: any change invalidates
previously formatted training data and requires re-tuning the decoder.
Bump SCHEMA_VERSION on every breaking change.
"""

SCHEMA_VERSION = "1.0"

# ── Field → lexicon-category mapping ─────────────────────────────────────────
# Which lexicon.json.gz category builds the token trie for each JSON field.
# Fields absent from this map (date, notes, role) decode unconstrained.
FIELD_LEXICON_MAP = {
    "surname": "surnames",
    "given_name": "given_names",
    "place": "towns",
    "profession": "occupations",
}

# Recommended role values per register type. Open vocabulary — the model may
# emit other German role labels it reads on the page.
ROLES = {
    "Trauungsbuch": ["Bräutigam", "Braut", "Vater des Bräutigams",
                     "Vater der Braut", "Mutter des Bräutigams",
                     "Mutter der Braut", "Beistand", "Priester"],
    "Taufbuch": ["Kind", "Vater", "Mutter", "Pate", "Patin",
                 "Hebamme", "Priester"],
    "Sterbebuch": ["Verstorbener", "Verstorbene", "Gatte", "Gattin",
                   "Vater", "Mutter", "Priester"],
}

# JSON Schema (draft-07 subset) for one transcribed page.
PAGE_SCHEMA = {
    "type": "object",
    "required": ["register_type", "entries"],
    "properties": {
        "register_type": {"enum": list(ROLES)},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "persons"],
                "properties": {
                    # Verbatim as written on the page ("17. Jänner 1873");
                    # normalisation happens in post-processing, not here.
                    "date": {"type": "string"},
                    "persons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["role", "surname"],
                            "properties": {
                                "role": {"type": "string"},
                                "given_name": {"type": "string"},
                                "surname": {"type": "string"},
                                "profession": {"type": "string"},
                                "place": {"type": "string"},
                            },
                        },
                    },
                    # Everything on the row not captured above, verbatim
                    # (Krankheit und Todesart, marginalia, formulae).
                    "notes": {"type": "string"},
                },
            },
        },
    },
}

# ── The frozen instruction ───────────────────────────────────────────────────
# Same uncertainty convention as PASS1_PROMPT: uncertain readings stay wrapped
# in ⟦double brackets⟧ inside field values, so soft-mode constrained decoding
# and the fuzzy reranker keep working on structured output.
STRUCTURED_PROMPT = """\
You are an expert reader of 19th/early-20th-century Austrian parish registers \
(Trauungsbuch, Taufbuch, Sterbebuch) written in German Kurrent and Latin \
cursive script.

Transcribe this register page as JSON with exactly this structure:

{
  "register_type": "Trauungsbuch" | "Taufbuch" | "Sterbebuch",
  "entries": [
    {
      "date": "<date verbatim as written, e.g. '17. Jänner 1873'>",
      "persons": [
        {
          "role": "<German role as on the page, e.g. Bräutigam, Braut, \
Kind, Vater, Pate, Verstorbener, Beistand>",
          "given_name": "<given name(s)>",
          "surname": "<family name>",
          "profession": "<occupation/standing if given, e.g. Bauer, \
Taglöhner, Schmied>",
          "place": "<place of origin/residence if given>"
        }
      ],
      "notes": "<all remaining text of the row verbatim: cause of death, \
legitimacy, marginalia, formulae>"
    }
  ]
}

Rules:
1. One entry per register row. One person object per named individual, \
including parents and witnesses.
2. Omit a field entirely when the page does not give it. Never invent values.
3. Keep original German spelling and abbreviations verbatim; do not \
modernise or expand.
4. Do NOT guess. Wrap any smudged, faded, or ambiguous reading in \
⟦double brackets⟧ inside the field value, e.g. "surname": "⟦Hu?er⟧".
5. Output ONLY the JSON object — no Markdown fences, no commentary.
"""


def validate_page(obj: dict) -> list[str]:
    """Cheap structural check (no jsonschema dependency). Returns problems."""
    problems = []
    if obj.get("register_type") not in ROLES:
        problems.append(f"register_type invalid: {obj.get('register_type')!r}")
    entries = obj.get("entries")
    if not isinstance(entries, list) or not entries:
        return problems + ["entries missing or empty"]
    for i, e in enumerate(entries):
        if not isinstance(e.get("date"), str):
            problems.append(f"entries[{i}].date missing")
        persons = e.get("persons")
        if not isinstance(persons, list) or not persons:
            problems.append(f"entries[{i}].persons missing or empty")
            continue
        for j, p in enumerate(persons):
            for req in ("role", "surname"):
                if not isinstance(p.get(req), str) or not p[req]:
                    problems.append(f"entries[{i}].persons[{j}].{req} missing")
            unknown = set(p) - {"role", "given_name", "surname",
                                "profession", "place"}
            if unknown:
                problems.append(
                    f"entries[{i}].persons[{j}] unknown fields: {unknown}")
    return problems
