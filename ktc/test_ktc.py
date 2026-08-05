"""Unit tests for KTC sub-steps."""

import json
import unittest
from pathlib import Path

from ktc.coreference import resolve_coreferences
from ktc.entity_extraction import CATEGORY_CRIME, extract_entities
from ktc.extraction import _relation_span
from ktc.filtering import passes_filters
from ktc.query_builder import build_queries
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_template

DATA_PATH = Path(__file__).resolve().parents[2].parent / "KARE-data" / "KARE" / "Data" / "KARE.jsonl"
if not DATA_PATH.exists():
    DATA_PATH = Path(r"C:\Users\pg\Downloads\pro\KARE-data\KARE\Data\KARE.jsonl")

BAD_COREF_HEADS = frozenset({"all the tasks", "millennials", "the matter"})


class _FakeToken:
    """Minimal stand-in for a spaCy Token, just enough to test _relation_span."""

    def __init__(self, i, text, dep_):
        self.i = i
        self.text = text
        self.dep_ = dep_
        self.children = []


class TestRelationSpan(unittest.TestCase):
    def test_plain_verb_excludes_subject(self):
        verb = _FakeToken(1, "is", "ROOT")
        self.assertEqual(_relation_span(verb), "is")

    def test_includes_negation_and_aux(self):
        verb = _FakeToken(3, "include", "ROOT")
        does = _FakeToken(1, "does", "aux")
        not_ = _FakeToken(2, "not", "neg")
        verb.children = [does, not_]
        self.assertEqual(_relation_span(verb), "does not include")

    def test_includes_trailing_particle(self):
        verb = _FakeToken(0, "give", "ROOT")
        up = _FakeToken(1, "up", "prt")
        verb.children = [up]
        self.assertEqual(_relation_span(verb), "give up")


class TestFiltering(unittest.TestCase):
    def test_rejects_identical_head_tail(self):
        triplet = Triplet("police", "helps", "police")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_head_without_noun(self):
        triplet = Triplet("quickly", "runs", "station")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_relation_tail_overlap(self):
        triplet = Triplet("victim", "file complaint", "complaint online")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_conjunction_head(self):
        triplet = Triplet("and police", "file", "complaint")
        self.assertFalse(passes_filters(triplet))

    def test_accepts_valid_triplet(self):
        triplet = Triplet("victim", "can file", "online complaint")
        self.assertTrue(passes_filters(triplet))


class TestVerbalization(unittest.TestCase):
    def test_template_adds_punctuation(self):
        sentence = verbalize_template(Triplet("Cyber Cells", "are present in", "every state"))
        self.assertTrue(sentence.endswith("."))
        self.assertIn("Cyber Cells", sentence)


class TestCoreference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")

    def _resolve(self, knowledge: str, triplets: list[Triplet]) -> list[Triplet]:
        return resolve_coreferences(triplets, knowledge, nlp=self.nlp)

    def test_same_sentence_antecedent(self):
        knowledge = "John filed the complaint because he was listed as a witness."
        raw = [Triplet("he", "was listed", "as a witness")]
        resolved = self._resolve(knowledge, raw)
        self.assertEqual(len(resolved), 1)
        self.assertIn("John", resolved[0].head)

    def test_previous_sentence_antecedent(self):
        knowledge = "Maria called the police. She filed an online complaint."
        raw = [Triplet("She", "filed", "an online complaint")]
        resolved = self._resolve(knowledge, raw)
        self.assertEqual(len(resolved), 1)
        self.assertIn("Maria", resolved[0].head)

    def test_no_antecedent_left_unsubstituted(self):
        knowledge = "It is important to file complaints quickly."
        raw = [Triplet("It", "is", "important to file complaints quickly")]
        resolved = self._resolve(knowledge, raw)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].head, "It")

    def test_no_document_global_fallback_on_real_dialogues(self):
        if not DATA_PATH.exists():
            self.skipTest("KARE.jsonl not available")

        from ktc.cleaning import clean_knowledge_text
        from ktc.extraction import extract_triplets

        for did in ("100", "3000", "4500"):
            with self.subTest(dialogue_id=did):
                dialogue = None
                with DATA_PATH.open(encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        if str(record["dialogue_id"]) == did:
                            dialogue = record
                            break
                self.assertIsNotNone(dialogue)
                cleaned = clean_knowledge_text(dialogue["knowledge"])
                raw = extract_triplets(cleaned, nlp=self.nlp)
                resolved = resolve_coreferences(raw, cleaned, nlp=self.nlp)
                bad_heads = [t.head for t in resolved if t.head in BAD_COREF_HEADS]
                self.assertEqual(
                    bad_heads,
                    [],
                    f"dialogue {did} still has document-global coref substitutions: {bad_heads}",
                )


class TestEntityExtraction(unittest.TestCase):
    def test_crime_and_medium_entities(self):
        text = "Someone is stalking me on Instagram and sending threats."
        entities = extract_entities(text)
        texts = {e["text"].lower() for e in entities}
        self.assertIn("stalking", texts)
        self.assertIn("instagram", texts)

    def test_queries_are_specific(self):
        entities = [{"text": "stalking", "category": CATEGORY_CRIME}]
        queries = build_queries(entities, max_queries=2)
        self.assertGreaterEqual(len(queries), 1)
        for q in queries:
            self.assertGreaterEqual(len(q.text.split()), 4)


if __name__ == "__main__":
    unittest.main()
