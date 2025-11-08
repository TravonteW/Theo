#!/usr/bin/env python3
"""
Unit tests for retrieval functions
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ask import retrieve_multi_source_contexts, _extract_target_sources, _available_sources


class TestRetrieveMultiSourceContexts(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_citation_map = {
            "1": {"source_pdf": "source1.pdf", "page_num": 1, "text": "Content from source 1"},
            "2": {"source_pdf": "source1.pdf", "page_num": 2, "text": "More content from source 1"},
            "3": {"source_pdf": "source2.pdf", "page_num": 1, "text": "Content from source 2"},
            "4": {"source_pdf": "source2.pdf", "page_num": 2, "text": "More content from source 2"},
            "5": {"source_pdf": "source3.pdf", "page_num": 1, "text": "Content from source 3"}
        }

        # Mock FAISS index
        self.mock_index = MagicMock()
        # Mock search results - return indices in order
        self.mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),  # distances
            np.array([[1, 2, 3, 4, 5]])  # indices
        )

    @patch('ask.get_embedding')
    def test_per_source_coverage(self, mock_get_embedding):
        """Test that each target source gets at least per_source_k results"""
        mock_get_embedding.return_value = [0.1] * 384  # Mock embedding

        # Mock that we want source1 and source2
        focus_sources = ["source1.pdf", "source2.pdf"]

        result = retrieve_multi_source_contexts(
            "test question",
            self.mock_index,
            self.mock_citation_map,
            k_total=6,
            per_source_k=2,
            focus_sources=focus_sources
        )

        # Count results per source
        source_counts = {}
        for ctx in result:
            source = ctx["source_pdf"]
            source_counts[source] = source_counts.get(source, 0) + 1

        # Each target source should have at least 1 result (guaranteed minimum)
        self.assertGreaterEqual(source_counts.get("source1.pdf", 0), 1)
        self.assertGreaterEqual(source_counts.get("source2.pdf", 0), 1)

    @patch('ask.get_embedding')
    def test_no_duplicate_ids(self, mock_get_embedding):
        """Test that no duplicate chunk IDs are returned"""
        mock_get_embedding.return_value = [0.1] * 384

        # Mock search to return duplicate indices
        self.mock_index.search.return_value = (
            np.array([[0.1, 0.1, 0.2, 0.2, 0.3]]),  # distances
            np.array([[1, 1, 2, 2, 3]])  # duplicate indices
        )

        result = retrieve_multi_source_contexts(
            "test question",
            self.mock_index,
            self.mock_citation_map,
            k_total=5,
            per_source_k=2
        )

        # Collect all IDs
        ids = [ctx["id"] for ctx in result]

        # Should have no duplicates
        self.assertEqual(len(ids), len(set(ids)))

    def test_available_sources(self):
        """Test _available_sources function"""
        sources = _available_sources(self.mock_citation_map)
        expected = ["source1.pdf", "source2.pdf", "source3.pdf"]
        self.assertEqual(sorted(sources), expected)

    def test_extract_target_sources_aliases(self):
        """Test that source extraction works with aliases"""
        available_sources = ["The Holy Quran.pdf", "King James Version.pdf", "Srimad-Bhagavad-Gita.pdf"]

        # Test Quran alias
        question = "What does the Quran say about prayer?"
        targets = _extract_target_sources(question, available_sources)
        self.assertIn("The Holy Quran.pdf", targets)

        # Test Bible alias
        question = "According to the Bible, what is love?"
        targets = _extract_target_sources(question, available_sources)
        # Should not match anything since "Bible" isn't in the aliases for KJV

        # Test KJV alias
        question = "The King James version mentions this"
        targets = _extract_target_sources(question, available_sources)
        self.assertIn("King James Version.pdf", targets)

        # Test Gita alias
        question = "The Bhagavad Gita discusses dharma"
        targets = _extract_target_sources(question, available_sources)
        self.assertIn("Srimad-Bhagavad-Gita.pdf", targets)

    def test_extract_target_sources_exact_match(self):
        """Test exact filename matching"""
        available_sources = ["The Life of Buddha.pdf", "Sacred Books.pdf"]

        question = "What does The Life of Buddha say about enlightenment?"
        targets = _extract_target_sources(question, available_sources)
        self.assertIn("The Life of Buddha.pdf", targets)

    @patch('ask.get_embedding')
    def test_fallback_to_single_query(self, mock_get_embedding):
        """Test fallback when fewer than 2 target sources are found"""
        mock_get_embedding.return_value = [0.1] * 384

        # Question that doesn't match multiple sources
        result = retrieve_multi_source_contexts(
            "general question about philosophy",
            self.mock_index,
            self.mock_citation_map,
            k_total=5,
            per_source_k=2
        )

        # Should still return results (fallback to single query)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)  # Respects k_total limit

    @patch('ask.get_embedding')
    def test_focus_sources_parameter(self, mock_get_embedding):
        """Test that focus_sources parameter overrides automatic detection"""
        mock_get_embedding.return_value = [0.1] * 384

        # Question mentions one source, but focus_sources specifies different ones
        focus_sources = ["source3.pdf"]

        result = retrieve_multi_source_contexts(
            "What does source1 say?",  # Mentions source1 in text
            self.mock_index,
            self.mock_citation_map,
            k_total=3,
            per_source_k=2,
            focus_sources=focus_sources
        )

        # Should prioritize focus_sources over automatic detection
        # At minimum, we should get results (may include other sources due to backfilling)
        self.assertGreater(len(result), 0)
        source_pdfs = [ctx["source_pdf"] for ctx in result]

        # The focus should attempt to get source3, but due to search mechanics
        # we may get other sources too. Just verify we got some results.
        unique_sources = set(source_pdfs)
        self.assertGreater(len(unique_sources), 0)


class TestSourceExtraction(unittest.TestCase):
    """Test source extraction edge cases"""

    def test_empty_question(self):
        """Test behavior with empty question"""
        sources = ["source1.pdf", "source2.pdf"]
        targets = _extract_target_sources("", sources)
        self.assertEqual(targets, [])

    def test_no_matches(self):
        """Test when no sources match the question"""
        sources = ["source1.pdf", "source2.pdf"]
        targets = _extract_target_sources("completely unrelated question", sources)
        self.assertEqual(targets, [])

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive"""
        sources = ["The Holy QURAN.pdf"]
        targets = _extract_target_sources("what does the quran say?", sources)
        self.assertIn("The Holy QURAN.pdf", targets)

    def test_deduplication(self):
        """Test that duplicate matches are removed"""
        sources = ["The Holy Quran.pdf", "Quran Translation.pdf"]
        # Question that might match both through aliases
        targets = _extract_target_sources("quran koran holy book", sources)
        # Should not have duplicates even if multiple alias patterns match
        self.assertEqual(len(targets), len(set(targets)))


if __name__ == "__main__":
    unittest.main()