"""Unit tests for KTC sub-steps."""

import json
import os
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
from ktc.ranking import _stable_rank_order
from ktc.triplet import Triplet
from ktc.verbalization import verbalize_llm, verbalize_template, verbalize_triplets, _sanitize_llm_sentence

# KARE.jsonl path: env override, then repo-relative default; skip integration tests if missing.
_DATA_ENV = os.environ.get("KARE_JSONL_PATH")
if _DATA_ENV:
    DATA_PATH = Path(_DATA_ENV)
else:
    DATA_PATH = Path(__file__).resolve().parents[2].parent / "KARE-data" / "KARE" / "Data" / "KARE.jsonl"

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
        if not DATA_PATH.exists():
            self.skipTest("KARE.jsonl not available (set KARE_JSONL_PATH)")

        samples = []
        with DATA_PATH.open(encoding="utf-8") as f:
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

        with mock.patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            sentences = verbalize_llm([Triplet("victim", "can file", "an online complaint")])

        self.assertEqual(sentences, ["A victim can file an online complaint."])
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model")


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
        if not DATA_PATH.exists():
            self.skipTest("KARE.jsonl not available (set KARE_JSONL_PATH)")

        from ktc.cleaning import clean_knowledge_text

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


class TestRanking(unittest.TestCase):
    def test_stable_rank_order_breaks_ties_by_index(self):
        scores = np.array([0.5, 0.5, 0.9, 0.5])
        order = _stable_rank_order(scores, top_k=4)
        self.assertEqual(int(order[0]), 2)
        self.assertEqual(list(order[1:]), [0, 1, 3])

    def test_stable_rank_order_empty(self):
        order = _stable_rank_order(np.array([]), top_k=5)
        self.assertEqual(len(order), 0)

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

    def test_parse_sentences_keeps_dialogue_100_style_facts(self):
        from ktc.live_summarize import _parse_sentences

        facts = [
            "The World Health Organization estimates the age-adjusted suicide rate per 100 000 population in India is 21.1.",
            "The burden of mental health problems in India is 2443 disability-adjusted life years (DALYs) per 100 000 population.",
            "The economic loss due to mental health conditions in India, between 2012-2030, is estimated at USD 1.03 trillion.",
            "Policy makers are encouraged to promote availability of and access to cost-effective treatment of common mental disorders at the primary health care level.",
        ]
        for fact in facts:
            parsed = _parse_sentences(fact)
            self.assertEqual(parsed, [fact], msg=fact)

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
            with mock.patch("ktc.live_summarize.enrich_search_result", side_effect=lambda r, **k: r):
                sentences = summarize_search_results(
                    "IPC Section 376 rape India", results, config
                )
        self.assertGreaterEqual(len(sentences), 1)
        self.assertEqual(sentences[0].source_url, "https://indiacode.nic.in/376")
        self.assertIn("376", sentences[0].sentence)
        self.assertIn("rape", sentences[0].sentence.lower())


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
        candidates, queries, raw = fetch_live_knowledge(
            "my husband threatens me", config, budget
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "live_api")
        self.assertEqual(candidates[0].url, "https://ncw.nic.in/h")

    def test_static_candidates_from_triplets(self):
        t = Triplet("victim", "can file", "complaint")
        candidates = static_candidates_from_triplets([t])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "static_dataset")
        self.assertIsNotNone(candidates[0].triplet)


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


class TestNltkSetup(unittest.TestCase):
    def test_resource_table_covers_filter_tagger(self):
        from ktc.nltk_setup import NLTK_RESOURCES, setup_command

        names = {name for _, name in NLTK_RESOURCES}
        self.assertIn("punkt", names)
        self.assertIn("averaged_perceptron_tagger", names)
        self.assertIn("scripts/setup_nltk.py", setup_command())


class TestPipelineIntegration(unittest.TestCase):
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
            )
            pipeline.run(
                "Victims can file a complaint online.",
                "agent: Hi. victim: I need help.",
            )
        fetch.assert_not_called()

    def test_enable_live_true_calls_fetch(self):
        from ktc.pipeline import KnowledgeTripletPipeline

        with mock.patch(
            "ktc.pipeline.fetch_live_knowledge", return_value=([], [], [])
        ) as fetch:
            pipeline = KnowledgeTripletPipeline(
                verbalization_backend="template",
                coref_backend="heuristic",
            )
            pipeline.run(
                "Victims can file a complaint.",
                "agent: Hi. user: I was raped.",
                enable_live=True,
            )
        fetch.assert_called_once()
        victim_arg = fetch.call_args[0][0]
        self.assertIn("raped", victim_arg.lower())


if __name__ == "__main__":
    unittest.main()
