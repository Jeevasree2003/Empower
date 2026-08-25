"""Unit tests for KTC sub-steps."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ktc.coreference import resolve_coreferences
from ktc.entity_extraction import (
    CATEGORY_CRIME,
    CATEGORY_LEGAL,
    CATEGORY_MEDIUM,
    CATEGORY_MENTAL_HEALTH,
    extract_entities,
)
from ktc.extraction import _relation_span, extract_triplets
from ktc.filtering import passes_filters
from ktc.knowledge_item import KnowledgeCandidate
from ktc.live_config import ApiCallBudget, LiveRetrievalConfig
from ktc.live_knowledge import fetch_live_knowledge, static_candidates_from_triplets
from ktc.live_retrieval import SearchResult, _domain_allowed
from ktc.live_summarize import summarize_search_results
from ktc.query_builder import build_queries
from ktc.ranking import _stable_rank_order, apply_score_gate, ranking_query_from_history
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_llm, verbalize_template, verbalize_triplets, _sanitize_llm_sentence

# KARE.jsonl path: env override, then repo-relative default; skip integration tests if missing.
_DATA_ENV = os.environ.get("KARE_JSONL_PATH")
_WORKSPACE = Path(__file__).resolve().parents[1]
if _DATA_ENV:
    DATA_PATH = Path(_DATA_ENV)
else:
    DATA_PATH = _WORKSPACE / "KARE-data" / "KARE" / "Data" / "KARE.jsonl"
SAMPLE_PATH = _WORKSPACE / "dataset" / "KARE-Sample.json"

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


class TestExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")

    def _extract(self, sentence: str):
        return extract_triplets(sentence, nlp=self.nlp)

    def test_passive_with_by_agent(self):
        triplets = self._extract("The complaint was filed by the victim at the police station.")
        self.assertGreater(len(triplets), 0)
        heads = {t.head.lower() for t in triplets}
        self.assertTrue(any("victim" in h for h in heads))

    def test_coordinated_subjects(self):
        triplets = self._extract("Cyber Cell and police can file online complaints.")
        self.assertGreaterEqual(len(triplets), 2)
        heads = {t.head.lower() for t in triplets}
        self.assertTrue(any("cyber" in h for h in heads))
        self.assertTrue(any("police" in h for h in heads))

    def test_real_kare_sentence_patterns(self):
        path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
        if not path.exists():
            self.skipTest("KARE.jsonl not available (set KARE_JSONL_PATH)")

        samples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                knowledge = record.get("knowledge", "")
                if "Cyber Cell" in knowledge or "complaint" in knowledge.lower():
                    samples.append(knowledge[:800])
                if len(samples) >= 3:
                    break

        for text in samples:
            triplets = extract_triplets(text, nlp=self.nlp)
            self.assertIsInstance(triplets, list)
            for t in triplets:
                self.assertTrue(t.head and t.relation and t.tail)

    def test_relative_clause_verb(self):
        triplets = self._extract(
            "Victims who report cyberstalking to the Cyber Cell, which operates under the IT Act, "
            "can also approach the NCW."
        )
        relations = {t.relation.lower() for t in triplets}
        self.assertGreaterEqual(len(triplets), 2)
        self.assertTrue(any("report" in r for r in relations))
        self.assertTrue(any("approach" in r for r in relations))

    def test_coordinated_verbs(self):
        triplets = self._extract("Cyber Cell investigates cases and prosecutes offenders.")
        relations = {t.relation.lower() for t in triplets}
        self.assertGreaterEqual(len(triplets), 2)
        self.assertTrue(any("investigat" in r for r in relations))
        self.assertTrue(any("prosecut" in r for r in relations))

    def test_non_locative_infinitive_not_a_prep_triple(self):
        triplets = self._extract("The officer arranged to face the accused in court.")
        self.assertFalse(
            any(" to" in f" {t.relation.lower()} " or t.relation.lower().endswith(" to") for t in triplets),
            msg=f"unexpected to-prep triples: {[(t.relation, t.tail) for t in triplets]}",
        )

    def test_direct_object_not_replaced_by_pobj(self):
        triplets = self._extract("Victims can lodge a complaint at the police station.")
        self.assertTrue(triplets)
        lodge_bare = [
            t
            for t in triplets
            if "lodge" in t.relation.lower() and " at" not in f" {t.relation.lower()}"
        ]
        self.assertTrue(
            any("complaint" in t.tail.lower() for t in lodge_bare),
            msg=f"expected lodge+complaint, got {[(t.relation, t.tail) for t in triplets]}",
        )
        self.assertFalse(
            any("police" in t.tail.lower() and "complaint" not in t.tail.lower() for t in lodge_bare),
            msg="pobj clobbered dobj: " + repr([(t.relation, t.tail) for t in lodge_bare]),
        )

    def test_prepositional_object_keeps_prep_on_relation(self):
        triplets = self._extract("Victims can lodge a complaint at the police station.")
        prep_hits = [
            t
            for t in triplets
            if "lodge" in t.relation.lower() and "at" in t.relation.lower()
        ]
        self.assertTrue(
            any("police" in t.tail.lower() for t in prep_hits),
            msg=f"expected lodge-at + police station, got {[(t.relation, t.tail) for t in triplets]}",
        )


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

    def test_rejects_bare_pronoun_head(self):
        triplet = Triplet("it", "is", "a cyber crime portal")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_unresolved_pronoun_tail(self):
        triplet = Triplet("victim", "can contact", "it")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_stopword_only_relation(self):
        triplet = Triplet("police station", "to the", "complaint desk")
        self.assertFalse(passes_filters(triplet))

    def test_rejects_weak_copula_prep_relation(self):
        self.assertFalse(passes_filters(Triplet("victim", "is to", "file a complaint")))
        self.assertFalse(passes_filters(Triplet("portal", "was of", "the government")))

    def test_rejects_repeated_tokens(self):
        self.assertFalse(passes_filters(Triplet("People", "Do", "Yoga Yoga")))

    def test_rejects_deictic_comment_heads(self):
        self.assertFalse(passes_filters(Triplet("they", "called", "me")))
        self.assertFalse(passes_filters(Triplet("he", "is", "law")))
        self.assertFalse(passes_filters(Triplet("wlhe", "ca n't control", "his anger")))
        self.assertFalse(passes_filters(Triplet("t Team Online Legal India", "will be in", "touch with you")))

    def test_rejects_near_duplicate_head_tail(self):
        self.assertFalse(passes_filters(Triplet("the police station", "helps", "police station")))


class TestVerbalization(unittest.TestCase):
    def test_template_adds_punctuation(self):
        sentence = verbalize_template(Triplet("Cyber Cells", "are present in", "every state"))
        self.assertTrue(sentence.endswith("."))
        self.assertIn("Cyber Cells", sentence)
        self.assertTrue(sentence[0].isupper())

    def test_sanitize_llm_sentence_strips_meta_commentary(self):
        raw = (
            "A victim can file an online complaint. "
            "Note: The given relation does not fit grammatically."
        )
        sentence = _sanitize_llm_sentence(raw)
        self.assertEqual(sentence, "A victim can file an online complaint.")

    def test_passive_inverts_head_tail(self):
        sentence = verbalize_template(Triplet("the police", "was filed by", "the complaint"))
        self.assertIn("complaint", sentence)
        self.assertIn("police", sentence)
        self.assertTrue(sentence.lower().startswith("the complaint"))

    def test_has_perfect_tense(self):
        sentence = verbalize_template(Triplet("NCW", "has launched", "a helpline"))
        self.assertIn("NCW", sentence)
        self.assertIn("helpline", sentence)

    def test_present_participle(self):
        sentence = verbalize_template(Triplet("victims", "are facing", "online harassment"))
        self.assertIn("victims", sentence.lower())
        self.assertIn("harassment", sentence)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_llm_falls_back_to_template_without_api_key(self):
        triplets = [Triplet("victim", "can file", "an online complaint")]
        sentences = verbalize_triplets(triplets, backend="llm")
        self.assertEqual(len(sentences), 1)
        self.assertIn("victim", sentences[0].lower())
        self.assertIn("complaint", sentences[0].lower())

    @mock.patch("ktc.verbalization._make_llm_client")
    def test_llm_verbalization_uses_chat_api(self, mock_make_client):
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content="A victim can file an online complaint"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig(llm_model="test-model"))

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                sentences = verbalize_llm([Triplet("victim", "can file", "an online complaint")])

        self.assertEqual(sentences, ["A victim can file an online complaint."])
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model")


class TestEvidenceSynthesis(unittest.TestCase):
    def _candidates(self) -> list[KnowledgeCandidate]:
        return [
            KnowledgeCandidate(
                text="KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.",
                source="counseling_bank",
                url="https://www.mohfw.gov.in/",
                domain="clinical",
            ),
            KnowledgeCandidate(
                text="A victim can file an FIR at the nearest police station.",
                source="static_dataset",
            ),
        ]

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_llm_falls_back_to_concatenation_without_api_key(self):
        from ktc.synthesis import synthesize_evidence

        result = synthesize_evidence(self._candidates(), "user: I am going insane.")
        self.assertFalse(result.used_llm)
        self.assertIn("1800-599-0019", result.text)
        self.assertIn("FIR", result.text)
        self.assertIn("KIRAN", result.text)

    def test_template_backend_concatenates(self):
        from ktc.synthesis import synthesize_evidence

        result = synthesize_evidence(
            self._candidates(),
            "user: I am going insane.",
            backend="template",
        )
        self.assertFalse(result.used_llm)
        self.assertTrue(result.text.startswith("KIRAN"))
        self.assertIn("police station", result.text.lower())

    @mock.patch("ktc.synthesis._make_llm_client")
    def test_llm_synthesis_uses_chat_api(self, mock_make_client):
        from ktc.synthesis import DEFAULT_SYNTHESIS_MODEL, synthesize_evidence

        grounded = (
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress "
            "support in India. A victim can file an FIR at the nearest police station."
        )
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content=grounded))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig(llm_model="ignored-model"))

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "ollama"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                result = synthesize_evidence(self._candidates(), "user: I am going insane.")

        self.assertTrue(result.used_llm)
        self.assertEqual(result.text, grounded if grounded.endswith(".") else grounded + ".")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_SYNTHESIS_MODEL)
        self.assertEqual(call_kwargs["temperature"], 0)
        self.assertEqual(call_kwargs["max_tokens"], 2048)
        self.assertIn("1800-599-0019", call_kwargs["messages"][1]["content"])
        system_prompt = call_kwargs["messages"][0]["content"]
        self.assertIn("You are given a numbered list of candidate facts below", system_prompt)
        self.assertIn("NOT to select a subset", system_prompt)
        self.assertIn("do not drop any fact's core content", system_prompt)
        self.assertIn("Do not select a subset", call_kwargs["messages"][1]["content"])

    @mock.patch("ktc.synthesis._make_llm_client")
    def test_grounding_check_rejects_hallucinated_phone(self, mock_make_client):
        from ktc.synthesis import synthesize_evidence

        hallucinated = (
            "Call iCall on 9152987821 and KIRAN on 1800-599-0019 for distress support."
        )
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content=hallucinated))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig())

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "ollama"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                result = synthesize_evidence(self._candidates(), "user: I am going insane.")

        self.assertFalse(result.used_llm)
        self.assertNotIn("9152987821", result.text)
        self.assertIn("1800-599-0019", result.text)
        self.assertIn("FIR", result.text)

    @mock.patch("ktc.synthesis._make_llm_client")
    def test_malformed_output_falls_back(self, mock_make_client):
        from ktc.synthesis import synthesize_evidence

        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content=""))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig())

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "ollama"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                result = synthesize_evidence(self._candidates(), "user: help")

        self.assertFalse(result.used_llm)
        self.assertIn("KIRAN", result.text)

    def test_coverage_gap_logs_dropped_phone_fact(self):
        from ktc.synthesis import log_coverage_gaps

        candidates = self._candidates() + [
            KnowledgeCandidate(
                text="iCall psychosocial helpline 9152987821 provides confidential counseling.",
                source="counseling_bank",
            )
        ]
        passage = (
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress "
            "support in India. A victim can file an FIR at the nearest police station."
        )
        with mock.patch("ktc.synthesis.logger") as mock_logger:
            gaps = log_coverage_gaps(passage, candidates)
        self.assertEqual(len(gaps), 1)
        self.assertIn("9152987821", gaps[0])
        warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(any("synthesis_coverage_gap" in msg for msg in warning_messages))
        self.assertTrue(any("9152987821" in str(call.args) for call in mock_logger.warning.call_args_list))

    def test_coverage_gap_silent_when_all_tokens_present(self):
        from ktc.synthesis import log_coverage_gaps

        candidates = self._candidates() + [
            KnowledgeCandidate(
                text="iCall psychosocial helpline 9152987821 provides confidential counseling.",
                source="counseling_bank",
            )
        ]
        passage = (
            "KIRAN 1800-599-0019 and iCall 9152987821 offer distress support. "
            "A victim can file an FIR at the nearest police station."
        )
        with mock.patch("ktc.synthesis.logger") as mock_logger:
            gaps = log_coverage_gaps(passage, candidates)
        self.assertEqual(gaps, [])
        mock_logger.warning.assert_not_called()

    @mock.patch("ktc.synthesis._make_llm_client")
    def test_successful_synthesis_logs_coverage_gap_for_dropped_fact(self, mock_make_client):
        from ktc.synthesis import synthesize_evidence

        candidates = self._candidates() + [
            KnowledgeCandidate(
                text="iCall psychosocial helpline 9152987821 provides confidential counseling.",
                source="counseling_bank",
            )
        ]
        trimmed = (
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress "
            "support in India. A victim can file an FIR at the nearest police station."
        )
        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content=trimmed))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig())

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "ollama"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                with self.assertLogs("ktc.synthesis", level="WARNING") as captured:
                    result = synthesize_evidence(candidates, "user: I am going insane.")

        self.assertFalse(result.used_llm)
        self.assertIn("9152987821", result.text)
        self.assertTrue(any("synthesis_coverage_gap" in line and "9152987821" in line for line in captured.output))
        self.assertTrue(any("falling back to concatenated evidence" in line for line in captured.output))

    @mock.patch("ktc.synthesis._make_llm_client")
    def test_incomplete_llm_output_falls_back_and_reports_missing_indices(self, mock_make_client):
        from ktc.synthesis import completeness_report, synthesize_evidence

        candidates = [
            KnowledgeCandidate(text="KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.", source="counseling_bank"),
            KnowledgeCandidate(text="A victim can file an FIR at the nearest police station.", source="static_dataset"),
            KnowledgeCandidate(text="iCall psychosocial helpline 9152987821 provides confidential counseling for people in emotional distress.", source="counseling_bank"),
            KnowledgeCandidate(text="Rape is a cognizable offence under IPC Section 376; a survivor can file an FIR and seek medical and legal aid without delay.", source="counseling_bank"),
            KnowledgeCandidate(text="Gang rape is an aggravated offence under IPC Section 376D; a delayed FIR is still valid and a survivor can request a medical examination and police protection.", source="counseling_bank"),
            KnowledgeCandidate(text="You do not have to file a police case before getting emotional support; 181 or NALSA legal aid can help if you later need legal information or protection.", source="counseling_bank"),
            KnowledgeCandidate(text="You can ask for help even if you are not sure what the problem is called; a counselor can listen first.", source="counseling_bank"),
            KnowledgeCandidate(text="iCall psychosocial helpline, Tata Institute of Social Sciences, offers telephone counseling services.", source="live_sentence_direct"),
        ]
        trimmed = (
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India. "
            "A victim can file an FIR at the nearest police station. "
            "iCall psychosocial helpline 9152987821 provides confidential counseling for people in emotional distress. "
            "Rape is a cognizable offence under IPC Section 376; a survivor can file an FIR and seek medical and legal aid without delay. "
            "iCall psychosocial helpline, Tata Institute of Social Sciences, offers telephone counseling services."
        )
        report = completeness_report(trimmed, candidates)
        missing = [index for index, covered, _preview in report if not covered]
        self.assertEqual(missing, [4, 5, 6])

        mock_client = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock(message=mock.Mock(content=trimmed))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_make_client.return_value = (mock_client, LiveRetrievalConfig())

        fake_openai = mock.Mock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "ollama"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                with self.assertLogs("ktc.synthesis", level="INFO") as captured:
                    result = synthesize_evidence(candidates, "victim: I was gang raped and need help.")

        self.assertFalse(result.used_llm)
        self.assertIn("376D", result.text)
        self.assertIn("NALSA", result.text)
        self.assertIn("listen first", result.text)
        joined_logs = "\n".join(captured.output)
        self.assertIn("completeness_check candidate=4 covered=False", joined_logs)
        self.assertIn("completeness_check candidate=5 covered=False", joined_logs)
        self.assertIn("completeness_check candidate=6 covered=False", joined_logs)
        self.assertIn("missing_indices=[4, 5, 6]", joined_logs)

    def test_pipeline_synthesize_defaults_off(self):
        from ktc.pipeline import KnowledgeTripletPipeline
        from ktc.ranking import CandidateRankingResult

        class _PassthroughRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.9] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.9 if cl else 0.0)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_PassthroughRanker(),
            min_cosine=0.0,
        )
        result = pipeline.run_hybrid(
            "Victims can file a complaint online.",
            "agent: Hi. victim: I need help filing a complaint.",
            enable_live=False,
        )
        self.assertIsNone(result.synthesized_knowledge)

    def test_pipeline_synthesize_true_fills_field(self):
        from ktc.pipeline import KnowledgeTripletPipeline
        from ktc.ranking import CandidateRankingResult

        class _EmptyRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.9] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.9 if cl else 0.0)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_EmptyRanker(),
            min_cosine=0.0,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = pipeline.run_hybrid(
                "Victims can file a complaint online.",
                "agent: Hi. victim: I need help filing a complaint.",
                enable_live=False,
                synthesize=True,
            )
        self.assertIsNotNone(result.synthesized_knowledge)
        self.assertNotEqual(result.final_knowledge_sources, ["llm_synthesis"])

    def _passthrough_pipeline(self):
        from ktc.pipeline import KnowledgeTripletPipeline
        from ktc.ranking import CandidateRankingResult

        class _PassthroughRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.9] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.9 if cl else 0.0)

        return KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_PassthroughRanker(),
            min_cosine=0.0,
        )

    @mock.patch("ktc.pipeline.synthesize_evidence")
    def test_final_knowledge_uses_successful_llm_synthesis(self, mock_synth):
        from ktc.synthesis import SynthesisResult

        synthesized = "A survivor can file an FIR and seek free legal aid."
        mock_synth.return_value = SynthesisResult(text=synthesized, used_llm=True)
        result = self._passthrough_pipeline().run_hybrid(
            "Victims can file a complaint online.",
            "agent: Hi. victim: I need help filing a complaint.",
            enable_live=False,
            synthesize=True,
        )
        self.assertEqual(result.synthesized_knowledge, synthesized)
        self.assertEqual(result.final_knowledge_text, synthesized)
        self.assertEqual(result.final_knowledge_sources, ["llm_synthesis"])

    @mock.patch("ktc.pipeline.synthesize_evidence")
    def test_final_knowledge_falls_back_when_synthesis_fails(self, mock_synth):
        from ktc.synthesis import SynthesisResult

        mock_synth.return_value = SynthesisResult(text="concatenated fallback blob", used_llm=False)
        result = self._passthrough_pipeline().run_hybrid(
            "Victims can file a complaint online.",
            "agent: Hi. victim: I need help filing a complaint.",
            enable_live=False,
            synthesize=True,
        )
        self.assertEqual(result.synthesized_knowledge, "concatenated fallback blob")
        self.assertEqual(result.final_knowledge_text, "concatenated fallback blob")
        self.assertEqual(result.final_knowledge_sources, ["concatenation_fallback"])


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

    def test_two_sentence_lookback(self):
        knowledge = "Ravi contacted the helpline. He was scared. He filed a complaint."
        raw = [Triplet("He", "filed", "a complaint")]
        resolved = self._resolve(knowledge, raw)
        self.assertIn("Ravi", resolved[0].head)

    def test_single_candidate_fallback_rejects_mistagged_pronoun(self):
        from ktc.coreference import _pick_antecedent

        class Root:
            def __init__(self, pos_, ent_type_=""):
                self.pos_ = pos_
                self.ent_type_ = ent_type_

        class Chunk:
            def __init__(self, text, pos_, ent_type_=""):
                self.text = text
                self.root = Root(pos_, ent_type_)

        priya = Chunk("Priya", "PROPN", "PERSON")
        mistagged_she = Chunk("She", "NOUN", "")
        sents = [[priya], [mistagged_she], []]
        with mock.patch(
            "ktc.coreference._collect_candidates",
            side_effect=lambda sent_doc, before_char=None: list(sent_doc),
        ):
            chosen = _pick_antecedent(sents, sent_idx=2, pronoun_char=None, pronoun_class="fem")
        self.assertEqual(chosen, "Priya")

    def test_single_candidate_fallback_skips_neuter_pronoun(self):
        from ktc.coreference import _pick_antecedent

        class Root:
            def __init__(self, pos_, ent_type_=""):
                self.pos_ = pos_
                self.ent_type_ = ent_type_
                self.morph = type("Morph", (), {"get": lambda self, key: []})()

        class Chunk:
            def __init__(self, text, pos_, ent_type_=""):
                self.text = text
                self.root = Root(pos_, ent_type_)

            def __iter__(self):
                return iter(())

        portal = Chunk("the Cyber Cell", "NOUN")
        mistagged_it = Chunk("It", "NOUN")
        sents = [[portal], [mistagged_it], []]
        with mock.patch(
            "ktc.coreference._collect_candidates",
            side_effect=lambda sent_doc, before_char=None: list(sent_doc),
        ):
            chosen = _pick_antecedent(sents, sent_idx=2, pronoun_char=None, pronoun_class="neuter")
        self.assertEqual(chosen, "the Cyber Cell")

    def test_ambiguous_gender_requires_person_antecedent(self):
        knowledge = "The portal and Maria are available. He filed a complaint."
        raw = [Triplet("He", "filed", "a complaint")]
        resolved = self._resolve(knowledge, raw)
        # Should resolve to Maria (PERSON), not "the portal".
        self.assertIn("Maria", resolved[0].head)
        self.assertNotIn("portal", resolved[0].head.lower())

    def test_no_antecedent_left_unsubstituted(self):
        knowledge = "It is important to file complaints quickly."
        raw = [Triplet("It", "is", "important to file complaints quickly")]
        resolved = self._resolve(knowledge, raw)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].head, "It")

    def test_no_document_global_fallback_on_real_dialogues(self):
        path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
        if not path.exists():
            self.skipTest("KARE.jsonl / KARE-Sample.json not available")

        from ktc.cleaning import clean_knowledge_text

        for did in ("100", "3000", "4500"):
            with self.subTest(dialogue_id=did):
                dialogue = None
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        if str(record["dialogue_id"]) == did:
                            dialogue = record
                            break
                if dialogue is None:
                    self.skipTest(f"dialogue {did} not in {path}")
                cleaned = clean_knowledge_text(dialogue["knowledge"])
                raw = extract_triplets(cleaned, nlp=self.nlp)
                resolved = resolve_coreferences(raw, cleaned, nlp=self.nlp)
                bad_heads = [t.head for t in resolved if t.head in BAD_COREF_HEADS]
                self.assertEqual(
                    bad_heads,
                    [],
                    f"dialogue {did} still has document-global coref substitutions: {bad_heads}",
                )


class TestRanking(unittest.TestCase):
    def test_stable_rank_order_breaks_ties_by_index(self):
        scores = np.array([0.5, 0.5, 0.9, 0.5])
        order = _stable_rank_order(scores, top_k=4)
        self.assertEqual(int(order[0]), 2)
        self.assertEqual(list(order[1:]), [0, 1, 3])

    def test_stable_rank_order_empty(self):
        order = _stable_rank_order(np.array([]), top_k=5)
        self.assertEqual(len(order), 0)

    def test_score_gate_drops_weak_matches(self):
        pool = [
            KnowledgeCandidate(text="relevant FIR procedure", source="static_dataset"),
            KnowledgeCandidate(text="unrelated yoga tip", source="static_dataset"),
        ]
        kept, scores, top1 = apply_score_gate(pool, [0.51, 0.21], min_cosine=0.38, top_k=5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].text, "relevant FIR procedure")
        self.assertAlmostEqual(top1, 0.51)
        self.assertEqual(scores, [0.51])

    def test_score_gate_empty_when_none_pass(self):
        pool = [KnowledgeCandidate(text="noise", source="static_dataset")]
        kept, scores, top1 = apply_score_gate(pool, [0.22], min_cosine=0.38, top_k=5)
        self.assertEqual(kept, [])
        self.assertEqual(scores, [])
        self.assertAlmostEqual(top1, 0.22)

    def test_fact_embeddings_are_cached_across_calls(self):
        from ktc.passages import select_relevant_passages
        from ktc.ranking import clear_text_embedding_cache

        clear_text_embedding_cache()
        model = mock.Mock()
        model.encode.side_effect = lambda texts, **kwargs: np.ones((len(list(texts)), 4), dtype=float)
        ranker = mock.Mock(model=model)
        facts = [
            "KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India."
        ]
        first = select_relevant_passages(facts, "women helpline support", ranker, min_cosine=0.0)
        first_calls = model.encode.call_count
        self.assertGreater(first_calls, 0)
        second = select_relevant_passages(facts, "women helpline support", ranker, min_cosine=0.0)
        self.assertEqual(model.encode.call_count, first_calls)
        self.assertEqual(len(first), len(second))

    def test_ranking_query_ignores_bot_greeting(self):
        history = "agent: Greetings from Rakshak. user: my husband threatened to murder me."
        with mock.patch("ktc.entity_extraction.extract_entities", return_value=[{"text": "murder", "category": "crime"}]):
            query = ranking_query_from_history(history)
        self.assertNotIn("Rakshak", query)
        self.assertIn("murder", query.lower())
        self.assertIn("husband", query.lower())

    def test_mixed_pool_ranking_no_length_bias(self):
        """Static and live candidates with equal relevance should both appear in top-k."""
        from ktc.ranking import SentenceBertRanker

        class _MockRanker(SentenceBertRanker):
            def __init__(self):
                pass

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                from ktc.ranking import CandidateRankingResult

                candidate_list = list(candidates)
                scores = []
                for c in candidate_list:
                    base = 0.8 if "domestic violence" in c.text.lower() else 0.3
                    scores.append(base)
                order = _stable_rank_order(np.array(scores), top_k)
                ranked = [candidate_list[i] for i in order]
                ranked_scores = [scores[i] for i in order]
                return CandidateRankingResult(
                    candidates=ranked, scores=ranked_scores, top1_score=ranked_scores[0]
                )

        pool = [
            KnowledgeCandidate(text="short static", source="static_dataset"),
            KnowledgeCandidate(
                text="domestic violence helpline India 2026 official procedure",
                source="live_api",
                url="https://ncw.nic.in/example",
            ),
            KnowledgeCandidate(
                text="another long static triplet about unrelated fraud scams online",
                source="static_dataset",
            ),
        ]
        ranker = _MockRanker()
        result = ranker.rank_candidates_with_scores("domestic violence threat", pool, top_k=2)
        sources = {c.source for c in result.candidates}
        self.assertIn("live_api", sources)
        self.assertEqual(result.candidates[0].source, "live_api")

    def test_cross_encoder_reranker_uses_biencoder_shortlist(self):
        from ktc.ranking import CrossEncoderReranker, SentenceBertRanker

        pool = [
            KnowledgeCandidate(text="irrelevant fraud scam", source="static_dataset"),
            KnowledgeCandidate(
                text="domestic violence helpline number India official",
                source="live_api",
                url="https://ncw.nic.in/x",
            ),
            KnowledgeCandidate(text="unrelated static text", source="static_dataset"),
        ]

        class _MockBi(SentenceBertRanker):
            def __init__(self):
                pass

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                from ktc.ranking import CandidateRankingResult

                cl = list(candidates)
                scores = [0.9 if c.source == "live_api" else 0.1 for c in cl]
                order = _stable_rank_order(np.array(scores), top_k)
                ranked = [cl[i] for i in order]
                ranked_scores = [scores[i] for i in order]
                return CandidateRankingResult(ranked, ranked_scores, ranked_scores[0])

        class _MockCE:
            def predict(self, pairs):
                return [2.0 if "helpline" in b else -1.0 for _a, b in pairs]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.bi_encoder = _MockBi()
        reranker.cross_encoder = _MockCE()
        reranker.retrieve_top_n = 32

        result = reranker.rank_candidates_with_scores("domestic violence threat", pool, top_k=2)
        self.assertEqual(result.candidates[0].source, "live_api")


class TestLiveConfig(unittest.TestCase):
    def test_trusted_domains_lowercased_at_load(self):
        config = LiveRetrievalConfig.load()
        self.assertTrue(all(d == d.lower() for d in config.trusted_domains))
        self.assertIn("icallhelpline.org", config.trusted_domains)
        self.assertIn("indiankanoon.org", config.trusted_domains)

    def test_groq_llm_api_base_from_config(self):
        config = LiveRetrievalConfig.load()
        self.assertEqual(config.llm_model, "openai/gpt-oss-120b")
        self.assertEqual(config.llm_api_base, "https://api.groq.com/openai/v1")

    def test_api_call_budget(self):
        budget = ApiCallBudget(2)
        self.assertTrue(budget.can_call())
        budget.record(1)
        budget.record(1)
        self.assertFalse(budget.can_call())

    def test_loaded_live_timeouts(self):
        config = LiveRetrievalConfig.load()
        self.assertEqual(config.page_fetch_timeout, 8)
        self.assertEqual(config.search_timeout, 10)
        self.assertEqual(config.max_live_retrieval_seconds, 20)
        self.assertEqual(config.max_concurrent_fetches, 4)
        self.assertEqual(config.max_concurrent_queries, 3)
        self.assertEqual(config.live_sentence_candidates_per_page, 8)
        self.assertEqual(config.live_sentence_top_k, 8)


class TestLiveRetrieval(unittest.TestCase):
    def test_domain_allowed_case_insensitive(self):
        domains = ["icallhelpline.org", "ncw.nic.in"]
        self.assertTrue(_domain_allowed("https://iCALLhelpline.org/page", domains))
        self.assertTrue(_domain_allowed("https://www.ncw.nic.in/", domains))
        self.assertFalse(_domain_allowed("https://example.com/", domains))

    @mock.patch("ktc.live_retrieval._search_tavily")
    def test_search_allowlisted_filters_domains(self, mock_search):
        mock_search.return_value = [
            {"url": "https://cybercrime.gov.in/report", "title": "Report", "content": "info"},
            {"url": "https://spam.example.com/fake", "title": "Fake", "content": "junk"},
        ]
        config = LiveRetrievalConfig(
            trusted_domains=["cybercrime.gov.in"],
            results_per_query=3,
            search_provider="tavily",
        )
        with mock.patch.dict(os.environ, {"LIVE_SEARCH_API_KEY": "test-key"}):
            from ktc.live_retrieval import search_allowlisted

            results = search_allowlisted("how to report cybercrime", config)
        self.assertEqual(len(results), 1)
        self.assertIn("cybercrime.gov.in", results[0].domain)

    @mock.patch("ktc.live_retrieval._search_tavily")
    def test_search_allowlisted_skips_pdf_and_bitstream(self, mock_search):
        mock_search.return_value = [
            {"url": "https://indiacode.nic.in/bitstream/123/act.pdf", "title": "PDF", "content": "is not rape"},
            {"url": "https://indiacode.nic.in/a376.html", "title": "HTML", "content": "Section 376"},
        ]
        config = LiveRetrievalConfig(
            trusted_domains=["indiacode.nic.in"],
            results_per_query=3,
            search_provider="tavily",
        )
        with mock.patch.dict(os.environ, {"LIVE_SEARCH_API_KEY": "test-key"}):
            from ktc.live_retrieval import search_allowlisted

            results = search_allowlisted("IPC 376 pdf skip", config, cache_dir=Path(tempfile.mkdtemp()))
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].url.endswith(".html"))

    @mock.patch("requests.get")
    def test_fetch_page_text_uses_url_cache(self, mock_get):
        import uuid

        from ktc.live_retrieval import DEFAULT_CACHE_DIR, fetch_page_text

        mock_get.return_value = mock.MagicMock(
            status_code=200,
            text="<html><main><p>Helpline number 9152987821 listed for crisis support.</p></main></html>",
        )
        mock_get.return_value.raise_for_status = mock.MagicMock()
        url = f"https://vandrevala.org/test-cache-page-{uuid.uuid4().hex}"
        cache_dir = DEFAULT_CACHE_DIR / "pages_test"
        first = fetch_page_text(url, cache_dir=cache_dir, cache_ttl_days=30)
        second = fetch_page_text(url, cache_dir=cache_dir, cache_ttl_days=30)
        self.assertIn("9152987821", first)
        self.assertEqual(first, second)
        mock_get.assert_called_once()

    @mock.patch("requests.get")
    def test_failed_url_is_skipped_on_second_fetch(self, mock_get):
        import uuid

        from ktc.live_retrieval import fetch_page_text

        mock_get.side_effect = TimeoutError("slow domain")
        url = f"https://indiacode.nic.in/fail-{uuid.uuid4().hex}"
        cache_dir = Path(tempfile.mkdtemp())
        with self.assertRaises(TimeoutError):
            fetch_page_text(url, cache_dir=cache_dir, timeout=1, failure_cache_ttl_days=1)
        second = fetch_page_text(url, cache_dir=cache_dir, timeout=1, failure_cache_ttl_days=1)
        self.assertEqual(second, "")
        self.assertEqual(mock_get.call_count, 1)

    def test_page_fetch_failure_logs_distinct_reasons(self):
        import uuid
        import requests

        from ktc.live_retrieval import fetch_page_text

        cache_dir = Path(tempfile.mkdtemp())
        timeout_url = f"https://indiacode.nic.in/timeout-{uuid.uuid4().hex}"
        conn_url = f"https://indiacode.nic.in/conn-{uuid.uuid4().hex}"
        with mock.patch("requests.get", side_effect=requests.Timeout("slow")):
            with self.assertLogs("ktc.live_retrieval", level="WARNING") as cm:
                with self.assertRaises(requests.Timeout):
                    fetch_page_text(timeout_url, cache_dir=cache_dir, timeout=1)
        self.assertTrue(any("reason=timeout" in line for line in cm.output))
        with mock.patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with self.assertLogs("ktc.live_retrieval", level="WARNING") as cm:
                with self.assertRaises(requests.ConnectionError):
                    fetch_page_text(conn_url, cache_dir=cache_dir, timeout=1)
        self.assertTrue(any("reason=connection_error" in line for line in cm.output))

    @mock.patch("ktc.live_retrieval.fetch_page_text")
    def test_concurrent_enrich_keeps_per_url_attribution(self, mock_fetch):
        from ktc.live_retrieval import SearchResult, enrich_search_results

        def _page(url, **kwargs):
            if "good" in url:
                return "KIRAN helpline 1800-599-0019 offers 24x7 support in India for people in distress."
            raise ConnectionError("down")

        mock_fetch.side_effect = _page
        results = [
            SearchResult(url="https://ncw.nic.in/good", title="ok", snippet="short", domain="ncw.nic.in"),
            SearchResult(url="https://indiacode.nic.in/bad", title="bad", snippet="thin", domain="indiacode.nic.in"),
        ]
        config = LiveRetrievalConfig(max_concurrent_fetches=2, page_fetch_timeout=8)
        enriched = enrich_search_results(results, config)
        self.assertEqual(len(enriched), 2)
        self.assertIn("1800-599-0019", enriched[0].snippet)
        self.assertEqual(enriched[0].url, "https://ncw.nic.in/good")
        self.assertEqual(enriched[1].url, "https://indiacode.nic.in/bad")
        self.assertEqual(enriched[1].snippet, "thin")

    def test_api_budget_holds_under_concurrent_record(self):
        from concurrent.futures import ThreadPoolExecutor

        budget = ApiCallBudget(5)

        def worker():
            for _ in range(20):
                budget.record(1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: worker(), range(8)))
        self.assertEqual(budget.used, 5)
        self.assertFalse(budget.can_call())

    def test_live_deadline_returns_before_slow_fetch_finishes(self):
        from ktc.live_retrieval import LiveDeadline, run_io_tasks

        deadline = LiveDeadline(0.25)
        release = threading.Event()

        def slow():
            release.wait(timeout=5)
            return "too-late"

        started = time.monotonic()
        results = run_io_tasks([slow, slow], max_workers=2, deadline=deadline)
        elapsed = time.monotonic() - started
        release.set()
        self.assertLess(elapsed, 1.5)
        self.assertTrue(all(item is None for item in results))


class TestLiveSummarize(unittest.TestCase):
    def test_parse_sentences_mixed_no_relevant_info_keeps_valid_lines(self):
        from ktc.live_summarize import _parse_sentences

        raw = (
            "The portal cybercrime.gov.in has an option to report a crime.\n"
            "The helpline number listed for financial cyber fraud is 1930.\n"
            "NO_RELEVANT_INFO on the official procedure for reporting murder or homicide threats in India."
        )
        parsed = _parse_sentences(raw)
        self.assertEqual(len(parsed), 2)
        self.assertIn("report a crime", parsed[0])
        self.assertIn("1930", parsed[1])

    def test_parse_sentences_pure_no_relevant_info_returns_empty(self):
        from ktc.live_summarize import _parse_sentences

        self.assertEqual(_parse_sentences("NO_RELEVANT_INFO"), [])
        self.assertEqual(_parse_sentences("  NO_RELEVANT_INFO  "), [])

    def test_parse_sentences_keeps_short_helpline_facts(self):
        from ktc.live_summarize import _parse_sentences

        parsed = _parse_sentences("Helpline number is 1930.")
        self.assertEqual(len(parsed), 1)
        self.assertIn("1930", parsed[0])

    def test_parse_sentences_filters_meta_commentary(self):
        from ktc.live_summarize import _parse_sentences

        meta_examples = [
            "The source does not provide the actual definition of domestic violence in the given text.",
            "This page does not mention any helpline numbers.",
            "No information is provided about reporting procedures.",
            "The text does not mention suicide prevention resources.",
        ]
        for line in meta_examples:
            self.assertEqual(_parse_sentences(line), [], msg=line)

    def test_parse_sentences_keeps_valid_facts_while_dropping_meta(self):
        from ktc.live_summarize import _parse_sentences

        raw = (
            "Section 506 of the Indian law deals with punishment for criminal intimidation.\n"
            "The source does not provide the actual definition of domestic violence in the given text.\n"
            "The helpline number listed for financial cyber fraud is 1930."
        )
        parsed = _parse_sentences(raw)
        self.assertEqual(len(parsed), 2)
        self.assertIn("Section 506", parsed[0])
        self.assertIn("1930", parsed[1])

    def test_parse_sentences_drops_statistical_who_prose(self):
        from ktc.live_summarize import _parse_sentences

        dropped = [
            "The World Health Organization estimates the age-adjusted suicide rate per 100 000 population in India is 21.1.",
            "The burden of mental health problems in India is 2443 disability-adjusted life years (DALYs) per 100 000 population.",
            "The economic loss due to mental health conditions in India, between 2012-2030, is estimated at USD 1.03 trillion.",
            "Policy makers are encouraged to promote availability of and access to cost-effective treatment of common mental disorders at the primary health care level.",
        ]
        for fact in dropped:
            self.assertEqual(_parse_sentences(fact), [], msg=fact)

        kept = "The KIRAN helpline 1800-599-0019 provides 24x7 mental health support in India."
        self.assertEqual(_parse_sentences(kept), [kept])

    def test_per_source_attribution(self):
        import sys

        results = [
            SearchResult(url="https://ncw.nic.in/a", title="NCW", snippet="181 helpline", domain="ncw.nic.in"),
            SearchResult(
                url="https://cybercrime.gov.in/b", title="Cyber", snippet="file FIR", domain="cybercrime.gov.in"
            ),
        ]
        config = LiveRetrievalConfig(
            llm_model="gpt-4o-mini", results_per_query=2, summarize_backend="llm"
        )

        def fake_summarize(query, result, config, client):
            if "ncw" in result.url:
                return ["Helpline number is 181."]
            return ["FIR can be filed online."]

        fake_openai = mock.MagicMock()
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                with mock.patch("ktc.live_summarize._summarize_one_source", side_effect=fake_summarize):
                    sentences = summarize_search_results("domestic violence helpline", results, config)
        self.assertEqual(len(sentences), 2)
        urls = {s.source_url for s in sentences}
        self.assertEqual(urls, {"https://ncw.nic.in/a", "https://cybercrime.gov.in/b"})

    def test_no_relevant_info_returns_empty(self):
        import sys

        results = [
            SearchResult(url="https://who.int/x", title="WHO", snippet="unrelated", domain="who.int"),
        ]
        config = LiveRetrievalConfig(
            llm_model="gpt-4o-mini", results_per_query=1, summarize_backend="llm"
        )
        fake_openai = mock.MagicMock()

        with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            with mock.patch.dict(sys.modules, {"openai": fake_openai}):
                with mock.patch("ktc.live_summarize._summarize_one_source", return_value=[]):
                    sentences = summarize_search_results("query", results, config)
        self.assertEqual(sentences, [])

    def test_extractive_summarize_works_without_llm_key(self):
        snippet = (
            "Section 376 of the Indian Penal Code prescribes punishment for the offence of rape. "
            "The offence is cognizable, non-bailable, and triable by the Court of Session."
        )
        results = [
            SearchResult(
                url="https://indiacode.nic.in/376",
                title="IPC 376",
                snippet=snippet,
                domain="indiacode.nic.in",
            )
        ]
        config = LiveRetrievalConfig(summarize_backend="extractive", results_per_query=1)
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            with mock.patch(
                "ktc.live_summarize.enrich_search_results",
                side_effect=lambda results, config, **k: list(results),
            ):
                sentences = summarize_search_results(
                    "IPC Section 376 rape India", results, config
                )
        self.assertGreaterEqual(len(sentences), 1)
        self.assertEqual(sentences[0].source_url, "https://indiacode.nic.in/376")
        self.assertIn("376", sentences[0].sentence)
        self.assertIn("rape", sentences[0].sentence.lower())

    def test_extractive_sentences_skip_policy_maker_lines(self):
        from ktc.live_summarize import extractive_sentences

        text = (
            "Policy makers should be encouraged to promote availability of and access to "
            "cost-effective treatment of common mental disorders at the primary health care level. "
            "KIRAN is a 24x7 mental health helpline 1800-599-0019 operated for distress support in India."
        )
        selected = extractive_sentences("mental health helpline India", text)
        joined = " ".join(selected).lower()
        self.assertNotIn("policy makers", joined)
        self.assertTrue("kiran" in joined or "1800" in joined)

    def test_extractive_sentences_skip_nav_footer(self):
        from ktc.live_summarize import extractive_sentences

        text = (
            "EMAIL US AT icall@tiss.edu We Mental Health & Psychosocial Support – iCALL | "
            "MON - SAT 10am - 8pm. KIRAN is a 24x7 mental health helpline 1800-599-0019 in India."
        )
        selected = extractive_sentences("KIRAN helpline India", text)
        joined = " ".join(selected).lower()
        self.assertNotIn("email us at", joined)
        self.assertNotIn("mon - sat", joined)

    def test_is_scraped_boilerplate(self):
        from ktc.live_summarize import is_scraped_boilerplate

        self.assertTrue(
            is_scraped_boilerplate(
                "EMAIL US AT icall@tiss.edu We Mental Health & Psychosocial Support – iCALL"
            )
        )
        self.assertFalse(is_scraped_boilerplate("KIRAN is a 24x7 mental health helpline in India."))


class TestLiveKnowledge(unittest.TestCase):
    def test_victim_utterance_accepts_kare_user_role(self):
        from ktc.live_knowledge import victim_utterance_from_history

        history = (
            "agent: Hello, this is Rakshak. "
            "user: I'm an actress in a channel series, and the staff members raped me."
        )
        text = victim_utterance_from_history(history)
        self.assertIn("raped me", text.lower())

    def test_victim_utterance_prefers_mapped_victim_role(self):
        from ktc.live_knowledge import victim_utterance_from_history

        history = "agent: Hi. victim: He threatened to kill me."
        self.assertIn("threatened", victim_utterance_from_history(history).lower())

    @mock.patch("ktc.live_knowledge.summarize_search_results")
    @mock.patch("ktc.live_knowledge.search_allowlisted")
    def test_fetch_live_knowledge_offline(self, mock_search, mock_summarize):
        mock_search.return_value = [
            SearchResult(url="https://ncw.nic.in/h", title="t", snippet="s", domain="ncw.nic.in"),
        ]
        from ktc.live_summarize import LiveKnowledgeSentence

        mock_summarize.return_value = [
            LiveKnowledgeSentence(
                sentence="The NCW helpline number is 7827170170.",
                source_url="https://ncw.nic.in/h",
                query="domestic violence helpline India",
            )
        ]
        config = LiveRetrievalConfig(enable_live_retrieval=True, max_live_queries_per_dialogue=1)
        budget = ApiCallBudget(10)
        candidates, queries, raw, _funnel = fetch_live_knowledge(
            "my husband threatens me", config, budget
        )
        self.assertTrue(queries)
        self.assertEqual(len(raw), 1)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].source, "live_api")
        self.assertEqual(candidates[0].url, "https://ncw.nic.in/h")
        self.assertIsNotNone(candidates[0].triplet)

    @mock.patch("ktc.live_knowledge.summarize_search_results")
    @mock.patch("ktc.live_knowledge.search_allowlisted")
    def test_fetch_live_drops_nav_footer_and_keeps_triplets(self, mock_search, mock_summarize):
        mock_search.return_value = [
            SearchResult(url="https://icallhelpline.org/", title="iCall", snippet="s", domain="icallhelpline.org"),
        ]
        from ktc.live_summarize import LiveKnowledgeSentence

        mock_summarize.return_value = [
            LiveKnowledgeSentence(
                sentence="EMAIL US AT icall@tiss.edu We Mental Health & Psychosocial Support – iCALL | MON - SAT 10am - 8pm",
                source_url="https://icallhelpline.org/",
                query="iCall helpline India",
            ),
            LiveKnowledgeSentence(
                sentence="The NCW helpline number is 7827170170.",
                source_url="https://icallhelpline.org/",
                query="iCall helpline India",
            ),
        ]
        config = LiveRetrievalConfig(enable_live_retrieval=True, max_live_queries_per_dialogue=1)
        candidates, _queries, _raw, _funnel = fetch_live_knowledge(
            "I am going insane where to go for help", config, ApiCallBudget(10)
        )
        joined = " ".join(c.text.lower() for c in candidates)
        self.assertNotIn("email us at", joined)
        self.assertNotIn("mon - sat", joined)
        self.assertTrue(candidates)
        self.assertTrue(all(c.triplet is not None for c in candidates))

    def test_fetch_live_knowledge_respects_time_budget(self):
        from ktc.query_builder import SearchQuery

        release = threading.Event()

        def slow_search(*args, **kwargs):
            release.wait(timeout=5)
            return []

        config = LiveRetrievalConfig(
            enable_live_retrieval=True,
            max_live_queries_per_dialogue=3,
            max_concurrent_queries=3,
            max_live_retrieval_seconds=0.3,
        )
        queries = [
            SearchQuery("q1", "rape", "crime", "crime_report_india"),
            SearchQuery("q2", "rape", "crime", "crime_statute_indiacode"),
            SearchQuery("q3", "helpline", "mental_health", "mh_crisis_helpline"),
        ]
        started = time.monotonic()
        with mock.patch("ktc.live_knowledge.search_allowlisted", side_effect=slow_search):
            with mock.patch("ktc.live_knowledge.build_queries", return_value=queries):
                with mock.patch("ktc.live_knowledge.extract_entities", return_value=[]):
                    fetch_live_knowledge("urgent rape help", config, ApiCallBudget(10), enabled=True)
        elapsed = time.monotonic() - started
        release.set()
        self.assertLess(elapsed, 1.5)

    def test_static_candidates_from_triplets(self):
        t = Triplet("victim", "can file", "complaint")
        candidates = static_candidates_from_triplets([t])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "static_dataset")
        self.assertIsNotNone(candidates[0].triplet)

    def test_sentence_relevance_works_when_openie_empty(self):
        from ktc.live_knowledge import sentence_relevance_candidates
        from ktc.live_summarize import LivePageStats

        pages = [
            LivePageStats(
                url="https://icallhelpline.org/",
                query="KIRAN mental health helpline India",
                sentences_extracted=2,
                sentences=[
                    "KIRAN is a 24x7 mental health helpline 1800-599-0019 operated for distress support in India.",
                    "The campus newsletter mentions a sports meet at IIT Bombay this weekend.",
                ],
            )
        ]

        class _QueryRanker:
            def cosine_to_query(self, query, texts):
                scores = []
                for text in texts:
                    scores.append(0.81 if "helpline" in text.lower() else 0.11)
                return scores

        with mock.patch("ktc.live_knowledge.extract_triplets", return_value=[]):
            candidates = sentence_relevance_candidates(
                pages, _QueryRanker(), min_cosine=0.38, top_k_per_page=3
            )
        joined = " ".join(c.text for c in candidates)
        self.assertIn("1800-599-0019", joined)
        self.assertNotIn("IIT Bombay", joined)
        self.assertTrue(all(c.extraction_method == "sentence_relevance" for c in candidates))

    def test_sentence_relevance_encodes_all_pages_in_one_batch(self):
        from ktc.live_knowledge import sentence_relevance_candidates
        from ktc.live_summarize import LivePageStats
        from ktc.ranking import clear_text_embedding_cache

        clear_text_embedding_cache()
        pages = [
            LivePageStats(
                url=f"https://icallhelpline.org/p{index}",
                query=f"helpline query {index}",
                sentences_extracted=2,
                sentences=[
                    f"This helpline page {index} explains confidential telephone counseling in India.",
                    f"This second sentence on page {index} mentions distress support hours.",
                ],
            )
            for index in range(3)
        ]
        model = mock.Mock()
        model.encode.side_effect = lambda texts, **kwargs: np.ones((len(list(texts)), 4), dtype=float)
        ranker = mock.Mock(model=model)
        sentence_relevance_candidates(pages, ranker, min_cosine=0.0, top_k_per_page=8)
        self.assertEqual(model.encode.call_count, 1)
        encoded = list(model.encode.call_args[0][0])
        for page in pages:
            for sentence in page.sentences:
                self.assertIn(sentence, encoded)

    def test_openie_empty_keeps_live_sentence_direct_in_verbalized(self):
        from ktc.live_knowledge import LiveFunnel, direct_sentence_candidates
        from ktc.live_summarize import LivePageStats
        from ktc.pipeline import KnowledgeTripletPipeline
        from ktc.ranking import CandidateRankingResult

        sentences = [
            "iCALL is a psychosocial helpline that offers telephone counseling for people in emotional distress in India.",
            "A rape survivor should be provided free medical treatment, psychological counselling, and legal aid by the state.",
            "KIRAN is a 24x7 mental health helpline 1800-599-0019 operated for distress support in India.",
            "Survivors can request a medical examination and police protection after reporting the offence to authorities.",
            "Confidential counseling is available even if you are not sure what the problem is called by the counselor.",
        ]
        pages = [
            LivePageStats(
                url="https://icallhelpline.org/",
                query="mental health helpline India",
                sentences_extracted=5,
                sentences=sentences,
            )
        ]

        class _AllRelevantRanker:
            model = None

            def cosine_to_query(self, query, texts):
                return [0.82] * len(list(texts))

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.9] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.9 if cl else 0.0)

        with mock.patch("ktc.live_knowledge.extract_triplets", return_value=[]):
            extra_openie, direct = direct_sentence_candidates(
                pages, [], _AllRelevantRanker(), nlp=None, min_cosine=0.38, top_k_per_page=8
            )
        self.assertEqual(extra_openie, [])
        self.assertEqual(len(direct), 5)
        self.assertTrue(all(c.source == "live_sentence_direct" for c in direct))

        funnel = LiveFunnel(
            live_sentences=5,
            live_triplets=0,
            live_sentence_relevance=5,
            live_sentences_used_directly=5,
        )
        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_AllRelevantRanker(),
            min_cosine=0.0,
        )
        with mock.patch(
            "ktc.pipeline.fetch_live_knowledge",
            return_value=(direct, [], [], funnel),
        ):
            result = pipeline.run_hybrid(
                "",
                "agent: Hi. victim: I was raped and I need a helpline and legal aid.",
                enable_live=True,
            )
        self.assertGreaterEqual(len(result.verbalized), 1)
        self.assertTrue(any(c.source == "live_sentence_direct" for c in result.ranked_candidates))
        blob = " ".join(result.verbalized).lower()
        self.assertTrue("helpline" in blob or "legal aid" in blob)
        self.assertEqual(result.knowledge_funnel["live_sentences_used_directly"], 5)

    def test_dedup_openie_and_sentence_relevance(self):
        from ktc.live_knowledge import dedup_candidates
        from ktc.pipeline import _dedup_texts

        openie = KnowledgeCandidate(
            text="KIRAN helpline 1800-599-0019 offers 24x7 support in India.",
            source="live_api",
            extraction_method="openie",
        )
        relevance = KnowledgeCandidate(
            text="KIRAN helpline 1800-599-0019 offers 24x7 support in India",
            source="live_api",
            extraction_method="sentence_relevance",
        )
        merged = dedup_candidates([openie, relevance])
        self.assertEqual(len(merged), 1)
        verbalized = _dedup_texts([openie.text + ".", relevance.text])
        self.assertEqual(len(verbalized), 1)


class TestCitationAndSubstringDedup(unittest.TestCase):
    def test_citation_filter_removes_docket_pattern(self):
        from ktc.cleaning import strip_legal_citations
        from ktc.live_summarize import split_live_sentences

        raw = "A rape survivor should be provided legal aid 23 2026:JHHC:16350-DB by the state."
        cleaned = strip_legal_citations(raw)
        self.assertNotIn("2026:JHHC:16350-DB", cleaned)
        self.assertNotIn("JHHC", cleaned)
        self.assertIn("should be provided legal aid", cleaned)

        page = (
            "A rape survivor should be provided free medical treatment, psychological counselling, and legal aid. "
            "23 2026:JHHC:16350-DB confirms the High Court directions on survivor care."
        )
        blob = " ".join(split_live_sentences(page))
        self.assertNotIn("JHHC", blob)
        self.assertNotIn("16350-DB", blob)

    def test_substring_dedup_keeps_longer_clause(self):
        from ktc.pipeline import _dedup_texts

        short = "should be provided legal aid"
        long = (
            "should be provided free medical treatment, psychological counselling, "
            "and legal aid"
        )
        kept = _dedup_texts([short, long])
        self.assertEqual(kept, [long])
        kept_reversed = _dedup_texts([long, short])
        self.assertEqual(kept_reversed, [long])
        nested = _dedup_texts(
            [
                "Victims should be provided legal aid.",
                "Victims should be provided legal aid immediately.",
            ]
        )
        self.assertEqual(nested, ["Victims should be provided legal aid immediately."])


class TestKnowledgeItem(unittest.TestCase):
    def test_to_dict_includes_source(self):
        item = KnowledgeCandidate(
            text="Section 498A addresses cruelty.",
            source="live_api",
            url="https://indiacode.nic.in/x",
            query="domestic violence law",
        )
        payload = item.to_dict()
        self.assertEqual(payload["source"], "live_api")
        self.assertEqual(payload["url"], "https://indiacode.nic.in/x")


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

    def test_crisis_entities_get_helpline_queries(self):
        entities = [{"text": "dying", "category": CATEGORY_MENTAL_HEALTH}]
        queries = build_queries(entities, max_queries=3)
        templates = {q.template for q in queries}
        self.assertIn("mh_crisis_helpline", templates)
        self.assertIn("mh_crisis_support", templates)
        texts = " ".join(q.text.lower() for q in queries)
        self.assertIn("helpline", texts)
        self.assertIn("suicide", texts)

    def test_crime_statute_indiacode_supplements_report(self):
        entities = [{"text": "rape", "category": CATEGORY_CRIME}]
        queries = build_queries(entities, max_queries=3)
        templates = {q.template for q in queries}
        self.assertIn("crime_definition", templates)
        self.assertIn("crime_report_india", templates)
        self.assertIn("crime_statute_indiacode", templates)
        statute = next(q for q in queries if q.template == "crime_statute_indiacode")
        self.assertIn("376", statute.text)
        self.assertIn("indiacode", statute.text.lower())
        self.assertIn("indiankanoon", statute.text.lower())

    def test_murder_report_avoids_how_to_prefix(self):
        entities = [{"text": "murder", "category": CATEGORY_CRIME}]
        queries = build_queries(entities, max_queries=3)
        report = next(q for q in queries if q.template == "crime_report_india")
        self.assertFalse(report.text.lower().startswith("how to"))

    def test_complaint_uses_fir_procedure_template(self):
        entities = [{"text": "complaint", "category": CATEGORY_LEGAL}]
        queries = build_queries(entities, max_queries=2)
        templates = {q.template for q in queries}
        self.assertIn("legal_fir_procedure", templates)
        self.assertNotIn("legal_general", templates)
        fir = next(q for q in queries if q.template == "legal_fir_procedure")
        self.assertIn("154", fir.text)

    def test_crime_queries_not_crowded_out_by_crisis(self):
        entities = [
            {"text": "dying", "category": CATEGORY_MENTAL_HEALTH},
            {"text": "murder", "category": CATEGORY_CRIME},
        ]
        queries = build_queries(entities, max_queries=3)
        templates = {q.template for q in queries}
        self.assertIn("mh_crisis_helpline", templates)
        self.assertIn("crime_statute_indiacode", templates)
        self.assertTrue(any(t.startswith("crime_") for t in templates))

    def test_medium_query_does_not_repeat_online(self):
        entities = [{"text": "online", "category": CATEGORY_MEDIUM}]
        queries = build_queries(entities, max_queries=3)
        self.assertTrue(queries)
        for q in queries:
            lowered = q.text.lower()
            self.assertNotIn("on online", lowered)
            self.assertNotIn("online abuse on online", lowered)

    def test_platform_medium_keeps_on_preposition(self):
        entities = [
            {"text": "stalking", "category": CATEGORY_CRIME},
            {"text": "instagram", "category": CATEGORY_MEDIUM},
        ]
        queries = build_queries(entities, max_queries=8)
        texts = [q.text.lower() for q in queries]
        self.assertTrue(any("on instagram" in t for t in texts), msg=texts)
        self.assertFalse(any("on online" in t for t in texts))

    def test_help_seeking_dialogue_constructs_contact_queries(self):
        from ktc.query_builder import dialogue_situations

        utterance = (
            "I am going insane . I don't understand where to go and whom to ask for help ."
        )
        self.assertIn("help_seeking", dialogue_situations(utterance))
        queries = build_queries(
            [{"text": "insane", "category": CATEGORY_MENTAL_HEALTH}],
            max_queries=3,
            victim_text=utterance,
        )
        joined = " ".join(q.text.lower() for q in queries)
        self.assertTrue(queries)
        self.assertIn("kiran", joined)
        self.assertRegex(joined, r"whom to contact|where to get mental health help")
        self.assertNotIn("treatment for", joined)
        self.assertNotIn("policy", joined)
        self.assertNotIn("suicide prevention", joined)
        qtexts = [q.text.lower() for q in queries]
        self.assertFalse(any("online abuse" in t for t in qtexts))

    def test_homicide_dialogue_constructs_fir_and_intimidation_queries(self):
        utterance = "Hi Rakshak, my husband is planning to kill me with her secret"
        queries = build_queries(
            [{"text": "kill", "category": CATEGORY_CRIME}],
            max_queries=3,
            victim_text=utterance,
        )
        joined = " ".join(q.text.lower() for q in queries)
        self.assertRegex(joined, r"fir|police")
        self.assertRegex(joined, r"506|intimidation|302")
        self.assertNotIn("what is kill under indian law", joined)
        self.assertNotIn("romance", joined)

    def test_utterance_without_lexicon_entity_still_builds_queries(self):
        queries = build_queries(
            [],
            max_queries=2,
            victim_text="I don't understand where to go and whom to ask for help.",
        )
        self.assertTrue(queries)
        self.assertGreaterEqual(len(queries[0].text.split()), 4)
        self.assertIn("helpline", queries[0].text.lower())

    def test_generic_help_has_no_kiran_fallback_query(self):
        queries = build_queries(
            [],
            max_queries=3,
            victim_text="Hi..I need some urgent help.",
        )
        self.assertEqual(queries, [])

    def test_kicked_out_dialogue_queries_pwdva(self):
        queries = build_queries(
            [],
            max_queries=3,
            victim_text="my husband has kicked me out of the house along with kids",
        )
        joined = " ".join(q.text.lower() for q in queries)
        self.assertIn("residence", joined)
        self.assertIn("domestic violence", joined)


class TestCounselingBank(unittest.TestCase):
    def test_bank_returns_both_domains_for_victim_utterance(self):
        from ktc.counseling_bank import DOMAIN_CLINICAL, counseling_candidates

        items = counseling_candidates(
            [{"text": "dying", "category": CATEGORY_MENTAL_HEALTH}],
            "I am dying everyday bit by bit",
        )
        domains = {c.domain for c in items}
        self.assertEqual(domains, {DOMAIN_CLINICAL})
        texts = " ".join(c.text.lower() for c in items)
        self.assertIn("helpline", texts)
        self.assertIn("112", texts)

    def test_bank_empty_without_victim_text(self):
        from ktc.counseling_bank import counseling_candidates

        self.assertEqual(counseling_candidates([], ""), [])

    def test_generic_help_does_not_fire_always_facts(self):
        from ktc.counseling_bank import counseling_candidates

        items = counseling_candidates([], "Hi..I need some urgent help.")
        self.assertEqual(items, [])

    def test_generic_helpline_support_fires_bank_fact(self):
        from ktc.counseling_bank import counseling_candidates

        items = counseling_candidates(
            [{"text": "helpline", "category": CATEGORY_LEGAL}],
            "I want to take women helpline support",
        )
        self.assertTrue(items)
        joined = " ".join(c.text.lower() for c in items)
        self.assertTrue("181" in joined or "1800-599-0019" in joined or "kiran" in joined)

    def test_insane_turn_prefers_helplines_not_fir(self):
        from ktc.counseling_bank import counseling_candidates

        items = counseling_candidates(
            [{"text": "insane", "category": CATEGORY_MENTAL_HEALTH}],
            "I am going insane. I don't understand where to go and whom to ask for help.",
        )
        joined = " ".join(c.text.lower() for c in items)
        self.assertIn("kiran", joined)
        self.assertIn("icall", joined)
        self.assertNotIn("crpc section 154", joined)
        self.assertNotIn("policy makers", joined)

    def test_kicked_out_adds_residence_knowledge(self):
        from ktc.counseling_bank import counseling_candidates

        items = counseling_candidates(
            [],
            "my husband has kicked me out of the house along with kids",
        )
        joined = " ".join(c.text.lower() for c in items)
        self.assertIn("residence", joined)
        self.assertNotIn("kiran", joined)


class TestReplyKnowledge(unittest.TestCase):
    def test_drops_comment_thread_anecdotes_and_policy_text(self):
        from ktc.knowledge_item import KnowledgeCandidate
        from ktc.reply_knowledge import assemble_reply_knowledge

        ranked = assemble_reply_knowledge(
            [
                KnowledgeCandidate(
                    text="My husband has kicked me.",
                    source="static_dataset",
                ),
                KnowledgeCandidate(
                    text="Policy makers should be encouraged to promote availability of treatment.",
                    source="live_api",
                    url="https://www.who.int/india/health-topics/mental-health",
                ),
                KnowledgeCandidate(
                    text="KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.",
                    source="counseling_bank",
                    domain="clinical",
                ),
                KnowledgeCandidate(
                    text="You do not have to file a police case before getting emotional support; 181 or NALSA legal aid can help if you later need legal information or protection.",
                    source="counseling_bank",
                    domain="legal",
                ),
            ]
        )
        texts = [c.text.lower() for c in ranked]
        self.assertTrue(any("kiran" in t for t in texts))
        self.assertTrue(any("nalsa" in t for t in texts))
        self.assertFalse(any("kicked me" in t for t in texts))
        self.assertFalse(any("policy makers" in t for t in texts))


class TestNltkSetup(unittest.TestCase):
    def test_resource_table_covers_filter_tagger(self):
        from ktc.nltk_setup import NLTK_RESOURCES, setup_command

        names = {name for _, name in NLTK_RESOURCES}
        self.assertIn("punkt", names)
        self.assertIn("averaged_perceptron_tagger", names)
        self.assertIn("scripts/setup_nltk.py", setup_command())


class TestFinalKnowledgeText(unittest.TestCase):
    def test_empty_verbalized_uses_supplemental_facts(self):
        from ktc.pipeline import assemble_final_knowledge_text

        facts = [
            KnowledgeCandidate(
                text="Clinical fact one about KIRAN 1800-599-0019.",
                source="counseling_bank",
                domain="clinical",
            ),
            KnowledgeCandidate(
                text="Legal fact two about filing an FIR.",
                source="counseling_bank",
                domain="legal",
            ),
            KnowledgeCandidate(
                text="Clinical fact three about iCall 9152987821.",
                source="counseling_bank",
                domain="clinical",
            ),
        ]
        text, sources = assemble_final_knowledge_text([], facts)
        self.assertIn("KIRAN 1800-599-0019", text)
        self.assertIn("FIR", text)
        self.assertIn("9152987821", text)
        self.assertEqual(sources, ["supplemental_counseling"])

    def test_non_overlapping_supplemental_is_appended(self):
        from ktc.pipeline import assemble_final_knowledge_text

        verbalized = ["Community centres run evening support groups."]
        facts = [
            KnowledgeCandidate(
                text="KIRAN, the national mental health helpline 1800-599-0019, offers 24x7 distress support in India.",
                source="counseling_bank",
                domain="clinical",
            ),
        ]
        text, sources = assemble_final_knowledge_text(verbalized, facts)
        self.assertIn("Community centres", text)
        self.assertIn("1800-599-0019", text)
        self.assertEqual(sources, ["verbalized", "supplemental_counseling"])

    def test_both_empty_yields_empty_string(self):
        from ktc.pipeline import assemble_final_knowledge_text

        text, sources = assemble_final_knowledge_text([], [])
        self.assertEqual(text, "")
        self.assertEqual(sources, [])


class TestPipelineIntegration(unittest.TestCase):
    def _passthrough_ranker(self):
        from ktc.ranking import CandidateRankingResult

        class _Ranker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.9] * len(cl)
                sliced = cl[:top_k]
                scored = scores[:top_k]
                return CandidateRankingResult(
                    sliced, scored, scored[0] if scored else 0.0
                )

        return _Ranker()

    def test_config_defaults_live_retrieval_off(self):
        config = LiveRetrievalConfig.load()
        self.assertFalse(config.enable_live_retrieval)

    def test_static_run_does_not_call_live_fetch(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        knowledge = "Victims can lodge a complaint at the police station."
        history = "agent: Hello. victim: The staff members raped me."
        with mock.patch("ktc.pipeline.fetch_live_knowledge") as fetch:
            pipeline = KnowledgeTripletPipeline(
                verbalization_backend="template",
                coref_backend="heuristic",
                ranker=self._passthrough_ranker(),
                min_cosine=0.0,
            )
            sentences = pipeline.run(knowledge, history, enable_live=False)
        fetch.assert_not_called()
        self.assertTrue(sentences)
        joined = " ".join(sentences).lower()
        self.assertNotIn("lodge a police station", joined)

    def test_run_omitted_enable_live_follows_config_off(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        with mock.patch("ktc.pipeline.fetch_live_knowledge") as fetch:
            pipeline = KnowledgeTripletPipeline(
                verbalization_backend="template",
                coref_backend="heuristic",
                ranker=self._passthrough_ranker(),
                min_cosine=0.0,
            )
            pipeline.run(
                "Victims can file a complaint online.",
                "agent: Hi. victim: I need help.",
            )
        fetch.assert_not_called()

    def test_enable_live_true_calls_fetch(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        with mock.patch(
            "ktc.pipeline.fetch_live_knowledge", return_value=([], [], [], None)
        ) as fetch:
            pipeline = KnowledgeTripletPipeline(
                verbalization_backend="template",
                coref_backend="heuristic",
                ranker=self._passthrough_ranker(),
                min_cosine=0.0,
            )
            pipeline.run(
                "Victims can file a complaint.",
                "agent: Hi. user: I was raped.",
                enable_live=True,
            )
        fetch.assert_called_once()
        victim_arg = fetch.call_args[0][0]
        self.assertIn("raped", victim_arg.lower())
        self.assertTrue(fetch.call_args.kwargs.get("enabled", True))

    def test_greeting_without_victim_yields_no_static_knowledge(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=self._passthrough_ranker(),
        )
        result = pipeline.run_hybrid(
            "Do yoga daily for stress relief. Kinjal Singh was arrested in a romance scam.",
            "agent: Greetings, this is Rakshak to help you.",
            enable_live=False,
        )
        self.assertEqual(result.verbalized, [])
        self.assertTrue(result.no_passages_used)
        self.assertEqual(result.final_knowledge_text, "")
        self.assertEqual(result.final_knowledge_sources, [])

    def test_knowledge_funnel_log_line(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=self._passthrough_ranker(),
            min_cosine=0.0,
        )
        with self.assertLogs("ktc.pipeline", level="INFO") as cm:
            result = pipeline.run_hybrid(
                "Victims can file a complaint at the police station.",
                "agent: Hi. victim: I need help filing a complaint.",
                enable_live=False,
            )
        funnel_lines = [line for line in cm.output if "knowledge_funnel" in line]
        self.assertTrue(funnel_lines)
        self.assertIn("static_triplets=", funnel_lines[0])
        self.assertIn(f"final_verbalized_count={len(result.verbalized)}", funnel_lines[0])
        self.assertIn("live_sentences=0", funnel_lines[0])
        self.assertIn("live_sentences_used_directly=0", funnel_lines[0])

    def test_off_topic_blob_yields_no_static_when_gate_fails(self):
        from ktc.ranking import CandidateRankingResult
        from ktc.pipeline import KnowledgeTripletPipeline

        class _LowRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.21] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.21)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_LowRanker(),
            min_cosine=0.38,
        )
        result = pipeline.run_hybrid(
            "Romance scams often involve fake dating profiles and requests for money.",
            "user: my husband said he will murder me tonight.",
            enable_live=False,
        )
        joined = " ".join(result.verbalized).lower()
        self.assertNotIn("romance scam", joined)
        self.assertNotIn("kinjal", joined)
        extra = " ".join(c.text.lower() for c in result.supplemental_counseling)
        self.assertRegex(extra, r"112|police|murder|intimidation")
        self.assertFalse(any(c.source == "counseling_bank" for c in result.ranked_candidates))
        self.assertTrue(result.final_knowledge_text)
        self.assertIn("supplemental_counseling", result.final_knowledge_sources)
        self.assertRegex(result.final_knowledge_text.lower(), r"112|police|murder|intimidation")

    def test_dying_turn_covers_legal_and_clinical(self):
        from ktc.ranking import CandidateRankingResult
        from ktc.pipeline import KnowledgeTripletPipeline

        class _LowRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.21] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.21)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_LowRanker(),
        )
        result = pipeline.run_hybrid(
            "Once you deal with stress yoga will help you succeed in every step.",
            "bot: Hello. user: I am dying everyday bit by bit .",
            enable_live=False,
        )
        joined = " ".join(result.verbalized).lower()
        extra = " ".join(c.text.lower() for c in result.supplemental_counseling)
        self.assertNotIn("yoga", joined)
        self.assertNotIn("policy makers", joined)
        self.assertIn("kiran", extra)
        self.assertIn("112", extra)

    def test_insane_dialogue_brief_is_reply_ready(self):
        from ktc.ranking import CandidateRankingResult
        from ktc.pipeline import KnowledgeTripletPipeline

        class _LowRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.21] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.21)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_LowRanker(),
        )
        blob = (
            "We have received your complaint request and will get back to you soon "
            "to resolve your issue by filing a Consumer Complaint against Mental Harassment. "
            "Team Online Legal India will be in touch. my husband has kicked me. "
            "Policy makers should be encouraged to promote availability of treatment."
        )
        result = pipeline.inspect(
            blob,
            "bot: Hello. user: I am going insane . I don't understand where to go and whom to ask for help .",
            enable_live=False,
        )
        joined = " ".join(result["verbalized"]).lower()
        extra = " ".join(c["text"].lower() for c in result.get("supplemental_counseling") or [])
        self.assertNotIn("online legal india", joined)
        self.assertNotIn("policy makers", joined)
        self.assertNotIn("kicked me", joined)
        self.assertIn("kiran", extra)
        self.assertTrue(result["constructed_queries"])
        qtext = " ".join(q["query"].lower() for q in result["constructed_queries"])
        self.assertIn("kiran", qtext)
        self.assertRegex(qtext, r"whom to contact|where to get mental health help")
        self.assertNotIn("online abuse", qtext)
        self.assertFalse(any(c.get("source") == "counseling_bank" for c in result["ranked_candidates"]))
        extra_texts = [c["text"] for c in result.get("supplemental_counseling") or []]
        self.assertNotEqual(result["verbalized"], extra_texts)

    def test_generic_help_and_insane_do_not_share_canned_verbalized(self):
        from ktc.ranking import CandidateRankingResult
        from ktc.pipeline import KnowledgeTripletPipeline

        class _LowRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.21] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.21)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_LowRanker(),
        )
        blob = "Romance scams often involve fake dating profiles and requests for money."
        insane = pipeline.inspect(
            blob,
            "bot: Hello. user: I am going insane . I don't understand where to go and whom to ask for help .",
            enable_live=False,
        )
        generic = pipeline.inspect(
            blob,
            "bot: Hello. user: Hi..I need some urgent help.",
            enable_live=False,
        )
        self.assertTrue(insane.get("constructed_queries"))
        self.assertEqual(generic.get("constructed_queries"), [])
        self.assertTrue(insane.get("supplemental_counseling"))
        self.assertEqual(generic.get("supplemental_counseling"), [])
        extra = [c["text"] for c in insane["supplemental_counseling"]]
        self.assertNotEqual(insane["verbalized"], extra)

    def test_torture_turn_is_not_generic_kiran_only(self):
        from ktc.ranking import CandidateRankingResult
        from ktc.pipeline import KnowledgeTripletPipeline

        class _LowRanker:
            model = None

            def rank_candidates_with_scores(self, dialog_history, candidates, top_k=26):
                cl = list(candidates)
                scores = [0.21] * len(cl)
                return CandidateRankingResult(cl[:top_k], scores[:top_k], 0.21)

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=_LowRanker(),
        )
        result = pipeline.inspect(
            "Yoga reduces stress.",
            "bot: Hi. user: i am so frustrated . i am being tortured .",
            enable_live=False,
        )
        extra = " ".join(c["text"].lower() for c in result.get("supplemental_counseling") or [])
        self.assertIn("torture", " ".join(result.get("situations") or []))
        self.assertRegex(extra, r"498a|domestic violence|abuse|fir")
        qtext = " ".join(q["query"].lower() for q in result["constructed_queries"])
        self.assertRegex(qtext, r"498a|torture|domestic violence")

    def test_full_kare_dialogues_100_500_3000(self):
        if not DATA_PATH.exists():
            self.skipTest("KARE.jsonl not available")
        from ktc.pipeline import KnowledgeTripletPipeline

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=self._passthrough_ranker(),
        )
        specs = {"100": 0, "500": 0, "3000": 1}
        found = {}
        with DATA_PATH.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                did = str(record["dialogue_id"])
                if did in specs and did not in found:
                    found[did] = record
                if len(found) == len(specs):
                    break
        self.assertEqual(set(found), set(specs))
        for did, record in found.items():
            with self.subTest(dialogue_id=did):
                utterances = sorted(record["utterances"], key=lambda u: int(u["utterance_no"]))
                history = []
                target = None
                bot_turn = 0
                for utterance in utterances:
                    role = utterance["author_role"]
                    text = f"{role}: {utterance['utterance'].strip()}"
                    if role in {"bot", "agent"} and history:
                        if bot_turn == specs[did]:
                            target = " ".join(history)
                            break
                        bot_turn += 1
                    history.append(text)
                self.assertIsNotNone(target)
                result = pipeline.inspect(record["knowledge"], target, enable_live=False)
                joined = " ".join(result["verbalized"]).lower()
                extra = " ".join(c["text"].lower() for c in result.get("supplemental_counseling") or [])
                self.assertNotIn("kinjal", joined + extra, msg=did)
                self.assertNotIn("romance scam", joined)
                self.assertNotIn("policy makers", joined)
                self.assertNotIn("online legal india", joined)
                self.assertTrue(
                    result["verbalized"] or result.get("supplemental_counseling"),
                    msg=did,
                )

    def test_sample_dialogues_100_and_500_do_not_crash(self):
        if not SAMPLE_PATH.exists():
            self.skipTest("dataset/KARE-Sample.json not available")
        from ktc.pipeline import KnowledgeTripletPipeline

        pipeline = KnowledgeTripletPipeline(
            verbalization_backend="template",
            coref_backend="heuristic",
            ranker=self._passthrough_ranker(),
            min_cosine=0.38,
        )
        wanted = {"100": 0, "500": 0}
        found = {}
        with SAMPLE_PATH.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                did = str(record["dialogue_id"])
                if did in wanted and did not in found:
                    found[did] = record
                if len(found) == len(wanted):
                    break
        self.assertEqual(set(found), set(wanted))
        for did, record in found.items():
            with self.subTest(dialogue_id=did):
                utterances = sorted(record["utterances"], key=lambda u: int(u["utterance_no"]))
                history = []
                target = None
                bot_turn = 0
                for utterance in utterances:
                    role = utterance["author_role"]
                    text = f"{role}: {utterance['utterance'].strip()}"
                    if role in {"bot", "agent"} and history:
                        if bot_turn == wanted[did]:
                            target = " ".join(history)
                            break
                        bot_turn += 1
                    history.append(text)
                self.assertIsNotNone(target)
                result = pipeline.inspect(record["knowledge"], target, enable_live=False)
                self.assertIn("verbalized", result)
                self.assertIn("no_passages_used", result)
                joined = " ".join(result["verbalized"]).lower()
                self.assertNotIn("kinjal singh", joined)


if __name__ == "__main__":
    unittest.main()
