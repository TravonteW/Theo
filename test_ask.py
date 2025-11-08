#!/usr/bin/env python3
"""
Unit tests for ask.py functions
"""

import unittest
from unittest.mock import patch, MagicMock
import json
from ask import extract_citations, group_and_deduplicate_citations, deduplicate_and_stitch_chunks


class TestExtractCitations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_contexts = [
            {
                "source_pdf": "source1.pdf",
                "page_num": 1,
                "text": "This is the first snippet from source 1"
            },
            {
                "source_pdf": "source1.pdf",
                "page_num": 2,
                "text": "This is the second snippet from source 1"
            },
            {
                "source_pdf": "source2.pdf",
                "page_num": 1,
                "text": "This is a snippet from source 2"
            }
        ]

    def test_extract_citations_basic_mapping(self):
        """Test basic citation extraction and mapping"""
        answer = "According to the text [1] and also [2], we can see that [3] provides additional context."
        result = extract_citations(answer, self.sample_contexts)

        self.assertEqual(result["answer"], answer)
        self.assertEqual(len(result["citations"]), 3)

        # Check citation mapping
        citations = result["citations"]
        self.assertEqual(citations[0]["index"], 1)
        self.assertEqual(citations[0]["pdf"], "source1.pdf")
        self.assertEqual(citations[0]["page"], 1)

        self.assertEqual(citations[1]["index"], 2)
        self.assertEqual(citations[1]["pdf"], "source1.pdf")
        self.assertEqual(citations[1]["page"], 2)

        self.assertEqual(citations[2]["index"], 3)
        self.assertEqual(citations[2]["pdf"], "source2.pdf")
        self.assertEqual(citations[2]["page"], 1)

    def test_extract_citations_ordering(self):
        """Test that citations are ordered correctly"""
        answer = "Sources mention [3], [1], and [2] in various contexts."
        result = extract_citations(answer, self.sample_contexts)

        # Should be ordered by citation number: 1, 2, 3
        citations = result["citations"]
        self.assertEqual(len(citations), 3)
        self.assertEqual(citations[0]["index"], 1)
        self.assertEqual(citations[1]["index"], 2)
        self.assertEqual(citations[2]["index"], 3)

    def test_extract_citations_out_of_range(self):
        """Test that out-of-range citation numbers are ignored"""
        answer = "The text [1] and [2] are valid, but [5] and [10] are out of range."
        result = extract_citations(answer, self.sample_contexts)

        # Should only include [1] and [2], ignoring [5] and [10]
        citations = result["citations"]
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["index"], 1)
        self.assertEqual(citations[1]["index"], 2)

    def test_extract_citations_no_citations(self):
        """Test behavior when no citations are found"""
        answer = "This is an answer without any citation numbers."
        result = extract_citations(answer, self.sample_contexts)

        self.assertEqual(result["answer"], answer)
        self.assertEqual(len(result["citations"]), 0)

    def test_extract_citations_duplicate_numbers(self):
        """Test handling of duplicate citation numbers"""
        answer = "The source [1] mentions this, and [1] also says that [2] confirms it."
        result = extract_citations(answer, self.sample_contexts)

        # Should only include each citation once
        citations = result["citations"]
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["index"], 1)
        self.assertEqual(citations[1]["index"], 2)


class TestGroupAndDeduplicateCitations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_citations = [
            {"pdf": "source1.pdf", "page": 1, "snippet": "First snippet", "index": 1},
            {"pdf": "source1.pdf", "page": 1, "snippet": "Another snippet from same page", "index": 2},
            {"pdf": "source1.pdf", "page": 2, "snippet": "Different page", "index": 3},
            {"pdf": "source2.pdf", "page": 1, "snippet": "Different source", "index": 4}
        ]

    def test_group_citations_by_source_and_page(self):
        """Test that citations are grouped by source and page"""
        result = group_and_deduplicate_citations(self.sample_citations)

        # Should have 3 results: source1 page 1 (combined), source1 page 2, source2 page 1
        self.assertEqual(len(result), 3)

        # Find the combined citation from source1 page 1
        combined = next(c for c in result if c["pdf"] == "source1.pdf" and c["page"] == 1)
        self.assertIn("First snippet", combined["snippet"])
        self.assertIn("Another snippet from same page", combined["snippet"])
        self.assertIn("...", combined["snippet"])  # Should contain separator

    def test_single_citation_per_page_unchanged(self):
        """Test that single citations per page remain unchanged"""
        single_citations = [
            {"pdf": "source1.pdf", "page": 1, "snippet": "Only snippet", "index": 1},
            {"pdf": "source2.pdf", "page": 1, "snippet": "Another single", "index": 2}
        ]

        result = group_and_deduplicate_citations(single_citations)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["snippet"], "Only snippet")
        self.assertEqual(result[1]["snippet"], "Another single")


class TestDeduplicateAndStitchChunks(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_contexts = [
            {
                "id": "1",
                "source_pdf": "source1.pdf",
                "page_num": 1,
                "text": "First page content"
            },
            {
                "id": "2",
                "source_pdf": "source1.pdf",
                "page_num": 2,
                "text": "Second page content"
            },
            {
                "id": "3",
                "source_pdf": "source1.pdf",
                "page_num": 3,
                "text": "Third page content"
            },
            {
                "id": "4",
                "source_pdf": "source2.pdf",
                "page_num": 1,
                "text": "Different source content"
            }
        ]

    def test_stitch_adjacent_pages(self):
        """Test that adjacent pages from same source are stitched together"""
        result = deduplicate_and_stitch_chunks(self.sample_contexts)

        # Should have 2 results: stitched source1 pages and source2
        self.assertEqual(len(result), 2)

        # Find the stitched result from source1
        stitched = next(r for r in result if r["source_pdf"] == "source1.pdf")
        self.assertIn("First page content", stitched["text"])
        self.assertIn("Second page content", stitched["text"])
        self.assertIn("Third page content", stitched["text"])
        self.assertEqual(stitched["page_num"], "1-3")
        self.assertEqual(stitched["pages_stitched"], 3)

    def test_separate_sources_not_stitched(self):
        """Test that different sources are not stitched together"""
        result = deduplicate_and_stitch_chunks(self.sample_contexts)

        source2_result = next(r for r in result if r["source_pdf"] == "source2.pdf")
        self.assertEqual(source2_result["text"], "Different source content")
        self.assertEqual(source2_result["page_num"], 1)
        self.assertEqual(source2_result["pages_stitched"], 1)


if __name__ == "__main__":
    unittest.main()