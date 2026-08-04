"""Unit tests for KTC sub-steps."""

import unittest

from ktc.extraction import _relation_span
from ktc.filtering import passes_filters
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_template


class _FakeToken:
    """Minimal stand-in for a spaCy Token, just enough to test _relation_span."""

    def __init__(self, i, text, dep_):
        self.i = i
        self.text = text
        self.dep_ = dep_
        self.children = []


class TestRelationSpan(unittest.TestCase):
    def test_plain_verb_excludes_subject(self):
        # "Cyberstalking is an offense ..." -> relation must be just "is",
        # not "Cyberstalking is" (regression test for the subtree bug).
        verb = _FakeToken(1, "is", "ROOT")
        self.assertEqual(_relation_span(verb), "is")

    def test_includes_negation_and_aux(self):
        # "does not include" -> aux "does" + neg "not" + verb "include"
        verb = _FakeToken(3, "include", "ROOT")
        does = _FakeToken(1, "does", "aux")
        not_ = _FakeToken(2, "not", "neg")
        verb.children = [does, not_]
        self.assertEqual(_relation_span(verb), "does not include")

    def test_includes_trailing_particle(self):
        # "give up" -> verb "give" + particle "up"
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


if __name__ == "__main__":
    unittest.main()
