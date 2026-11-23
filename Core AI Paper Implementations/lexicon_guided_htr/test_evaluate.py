"""Unit tests for the structured-mode entity metrics in evaluate.py.

Run:
    venv/bin/python -m unittest test_evaluate -v
"""
import copy
import unittest
from collections import Counter

from evaluate import (align_persons, canonical_text, count_persons,
                      count_unresolved, entry_aligned_exact, field_bags,
                      multiset_prf, strip_field, STRUCT_FIELDS)


def make_gold() -> dict:
    return {
        "register_type": "Trauungsbuch",
        "entries": [
            {
                "date": "17. Jänner 1873",
                "persons": [
                    {"role": "Bräutigam", "given_name": "Johann",
                     "surname": "Gruber", "profession": "Bauer",
                     "place": "Althofen"},
                    {"role": "Braut", "given_name": "Maria",
                     "surname": "Huber", "place": "Treibach"},
                ],
                "notes": "beide ledig",
            },
            {
                "date": "3. Feber 1873",
                "persons": [
                    {"role": "Bräutigam", "given_name": "Franz",
                     "surname": "Mayr", "profession": "Schmied"},
                    {"role": "Braut", "given_name": "Anna",
                     "surname": "Gruber"},
                ],
            },
        ],
    }


class TestStripField(unittest.TestCase):
    def test_mask_stripping(self):
        self.assertEqual(strip_field("⟦Hu?er⟧"), "Hu?er")
        self.assertEqual(strip_field("  ⟦Gruber⟧ "), "Gruber")
        self.assertEqual(strip_field("Johann ⟦Nep.⟧ Mayr"), "Johann Nep. Mayr")
        self.assertEqual(strip_field("Gruber"), "Gruber")


class TestMultisetPRF(unittest.TestCase):
    def test_math(self):
        pred = Counter({"Gruber": 2, "Mayr": 1})
        gold = Counter({"Gruber": 1, "Mayr": 1, "Huber": 1})
        p, r, f1 = multiset_prf(pred, gold)          # overlap = 2
        self.assertAlmostEqual(p, 2 / 3)
        self.assertAlmostEqual(r, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_duplicates_count_once_each(self):
        p, r, _ = multiset_prf(Counter({"Gruber": 3}), Counter({"Gruber": 1}))
        self.assertAlmostEqual(p, 1 / 3)
        self.assertAlmostEqual(r, 1.0)

    def test_empty(self):
        self.assertEqual(multiset_prf(Counter(), Counter()), (1.0, 1.0, 1.0))
        self.assertEqual(multiset_prf(Counter({"x": 1}), Counter()),
                         (0.0, 0.0, 0.0))
        self.assertEqual(multiset_prf(Counter(), Counter({"x": 1})),
                         (0.0, 0.0, 0.0))


class TestPerfectMatch(unittest.TestCase):
    def test_bags_and_exact(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        pb, gb = field_bags(pred), field_bags(gold)
        for f in STRUCT_FIELDS:
            self.assertEqual(multiset_prf(pb[f], gb[f]), (1.0, 1.0, 1.0), f)
        for f, (hits, total) in entry_aligned_exact(pred, gold).items():
            self.assertEqual(hits, total, f)
        self.assertEqual(entry_aligned_exact(pred, gold)["surname"], [4, 4])


class TestOneWrongSurname(unittest.TestCase):
    def test_metrics_drop(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        pred["entries"][0]["persons"][1]["surname"] = "Haber"  # gold: Huber
        p, r, f1 = multiset_prf(field_bags(pred)["surname"],
                                field_bags(gold)["surname"])
        self.assertAlmostEqual(p, 3 / 4)
        self.assertAlmostEqual(r, 3 / 4)
        counts = entry_aligned_exact(pred, gold)
        self.assertEqual(counts["surname"], [3, 4])
        # role still matches for the misread person
        self.assertEqual(counts["role"], [4, 4])


class TestMissingPerson(unittest.TestCase):
    def test_recall_drops_precision_holds(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        del pred["entries"][1]["persons"][1]  # drop Anna Gruber
        p, r, _ = multiset_prf(field_bags(pred)["given_name"],
                               field_bags(gold)["given_name"])
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 3 / 4)
        self.assertEqual(count_persons(pred), 3)
        self.assertEqual(count_persons(gold), 4)
        # entry counts still agree, so aligned matching runs; the missing
        # person costs one hit in every field she carries
        self.assertEqual(entry_aligned_exact(pred, gold)["surname"], [3, 4])


class TestExtraEntry(unittest.TestCase):
    def test_precision_drops(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        pred["entries"].append({
            "date": "9. März 1873",
            "persons": [{"role": "Bräutigam", "surname": "Erfunden"}],
        })
        p, r, _ = multiset_prf(field_bags(pred)["surname"],
                               field_bags(gold)["surname"])
        self.assertAlmostEqual(p, 4 / 5)
        self.assertAlmostEqual(r, 1.0)
        self.assertNotEqual(len(pred["entries"]), len(gold["entries"]))


class TestMaskStrippingInBags(unittest.TestCase):
    def test_masked_value_matches_gold(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        pred["entries"][0]["persons"][0]["surname"] = "⟦Gruber⟧"
        pred["entries"][0]["date"] = " ⟦17. Jänner 1873⟧ "
        pb, gb = field_bags(pred), field_bags(gold)
        self.assertEqual(multiset_prf(pb["surname"], gb["surname"]),
                         (1.0, 1.0, 1.0))
        self.assertEqual(multiset_prf(pb["date"], gb["date"]),
                         (1.0, 1.0, 1.0))
        self.assertEqual(entry_aligned_exact(pred, gold)["surname"], [4, 4])
        self.assertEqual(count_unresolved(pred), 2)
        self.assertEqual(count_unresolved(gold), 0)


class TestPersonAlignment(unittest.TestCase):
    def test_order_independent(self):
        gold = make_gold()
        pred = copy.deepcopy(gold)
        pred["entries"][0]["persons"].reverse()
        for f, (hits, total) in entry_aligned_exact(pred, gold).items():
            self.assertEqual(hits, total, f)

    def test_near_surname_pairs_by_edit_distance(self):
        gold = [{"role": "Braut", "surname": "Huber"},
                {"role": "Braut", "surname": "Gruber"}]
        pred = [{"role": "Braut", "surname": "Gruber"},
                {"role": "Braut", "surname": "Huher"}]
        matched = {g["surname"]: p["surname"]
                   for p, g in align_persons(pred, gold)}
        self.assertEqual(matched, {"Gruber": "Gruber", "Huber": "Huher"})


class TestCanonicalText(unittest.TestCase):
    def test_document_order_and_stripping(self):
        page = {
            "register_type": "Sterbebuch",
            "entries": [{
                "date": "1. Mai 1900",
                "persons": [{"role": "Verstorbener", "given_name": "Josef",
                             "surname": "⟦Wieser⟧"}],
                "notes": "an Typhus",
            }],
        }
        self.assertEqual(
            canonical_text(page),
            "Sterbebuch 1. Mai 1900 Verstorbener Josef Wieser an Typhus")


if __name__ == "__main__":
    unittest.main()
