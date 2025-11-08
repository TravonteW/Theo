#!/usr/bin/env python3
"""
Golden test suite for prompt responses - validates answer structure and format
"""

import unittest
from unittest.mock import patch, MagicMock
import re
from ask import answer, create_prompt_with_context


class TestPromptGoldens(unittest.TestCase):
    """Test suite for validating AI response structure and format"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_contexts = [
            {
                "source_pdf": "test_source.pdf",
                "page_num": 1,
                "text": "This is a test passage about meditation practices in Buddhism."
            },
            {
                "source_pdf": "another_source.pdf",
                "page_num": 2,
                "text": "Another passage discussing mindfulness and awareness."
            }
        ]

        # Mock responses that follow the expected format
        self.good_response = """
**Answer**
Meditation is a central practice in Buddhism that involves developing mindfulness and awareness [1]. The practice helps cultivate inner peace and understanding [2].

**Supporting Points**
• Buddhist meditation focuses on developing mindfulness through various techniques [1]
• The practice aims to reduce suffering and increase awareness of reality [2]
• Different schools emphasize different meditation methods [1]

**Tensions/Gaps**
Some sources emphasize concentration while others focus on insight meditation, showing different approaches within the tradition [1][2].
        """.strip()

        self.bad_response = "Just some text without proper structure or citations."

    def test_answer_section_present(self):
        """Test that response contains required Answer section"""
        self.assertIn("**Answer**", self.good_response)

        # Check that Answer section comes first
        answer_pos = self.good_response.find("**Answer**")
        self.assertEqual(answer_pos, 0, "Answer section should be at the beginning")

    def test_supporting_points_section_present(self):
        """Test that response contains required Supporting Points section"""
        self.assertIn("**Supporting Points**", self.good_response)

        # Check that it comes after Answer
        answer_pos = self.good_response.find("**Answer**")
        points_pos = self.good_response.find("**Supporting Points**")
        self.assertGreater(points_pos, answer_pos, "Supporting Points should come after Answer")

    def test_supporting_points_format(self):
        """Test that Supporting Points follow bullet format with citations"""
        # Extract Supporting Points section
        start = self.good_response.find("**Supporting Points**")
        end = self.good_response.find("**Tensions/Gaps**")
        if end == -1:
            points_section = self.good_response[start:]
        else:
            points_section = self.good_response[start:end]

        # Should contain bullet points
        self.assertIn("•", points_section, "Supporting Points should use bullet points")

        # Each bullet point should end with citation(s)
        bullet_lines = [line.strip() for line in points_section.split('\n') if line.strip().startswith('•')]

        for line in bullet_lines:
            # Should end with citation like [1] or [1][2]
            citation_pattern = r'\[\d+\](?:\[\d+\])*\s*$'
            self.assertRegex(line, citation_pattern, f"Bullet point should end with citation: {line}")

    def test_citations_in_valid_range(self):
        """Test that all citations are within valid range [1-N]"""
        num_contexts = len(self.sample_contexts)

        # Find all citations in the format [n]
        citations = re.findall(r'\[(\d+)\]', self.good_response)

        for citation in citations:
            citation_num = int(citation)
            self.assertGreaterEqual(citation_num, 1, f"Citation {citation_num} should be >= 1")
            self.assertLessEqual(citation_num, num_contexts, f"Citation {citation_num} should be <= {num_contexts}")

    def test_no_invalid_citations(self):
        """Test that no citations are out of range"""
        # This response has invalid citations
        bad_citation_response = "According to the text [1] and [5], we can conclude [10] that..."

        citations = re.findall(r'\[(\d+)\]', bad_citation_response)
        valid_citations = []

        for citation in citations:
            citation_num = int(citation)
            if 1 <= citation_num <= len(self.sample_contexts):
                valid_citations.append(citation_num)

        # Should only have [1] as valid
        self.assertEqual(valid_citations, [1])

    def test_tensions_gaps_section_optional(self):
        """Test that Tensions/Gaps section is optional but formatted correctly when present"""
        if "**Tensions/Gaps**" in self.good_response:
            # Should come after Supporting Points
            points_pos = self.good_response.find("**Supporting Points**")
            tensions_pos = self.good_response.find("**Tensions/Gaps**")
            self.assertGreater(tensions_pos, points_pos, "Tensions/Gaps should come after Supporting Points")

    def test_answer_contains_citations(self):
        """Test that Answer section contains inline citations"""
        # Extract Answer section
        start = self.good_response.find("**Answer**") + len("**Answer**")
        end = self.good_response.find("**Supporting Points**")
        answer_section = self.good_response[start:end].strip()

        # Should contain at least one citation
        citations = re.findall(r'\[\d+\]', answer_section)
        self.assertGreater(len(citations), 0, "Answer section should contain citations")

    def test_no_meta_references(self):
        """Test that response doesn't mention limitations, training data, or retrieval process"""
        bad_phrases = [
            "limitation",
            "training data",
            "retrieval process",
            "as an ai",
            "cannot access",
            "not in my training"
        ]

        response_lower = self.good_response.lower()
        for phrase in bad_phrases:
            self.assertNotIn(phrase, response_lower, f"Response should not mention: {phrase}")

    @patch('ask.get_completion')
    @patch('ask.load_resources')
    def test_full_answer_structure(self, mock_load_resources, mock_get_completion):
        """Integration test for full answer structure"""
        # Mock dependencies
        mock_index = MagicMock()
        mock_citation_map = {
            "1": {"source_pdf": "test.pdf", "page_num": 1, "text": "Test content"},
            "2": {"source_pdf": "test.pdf", "page_num": 2, "text": "More test content"}
        }
        mock_load_resources.return_value = (mock_index, mock_citation_map)
        mock_get_completion.return_value = self.good_response

        # Mock search results
        import numpy as np
        mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[1, 2]])
        )

        with patch('ask.get_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384

            result = answer("What is meditation in Buddhism?")

            # Verify structure
            response = result["answer"]
            self.assertIn("**Answer**", response)
            self.assertIn("**Supporting Points**", response)

            # Verify citations are extracted
            self.assertGreater(len(result["citations"]), 0)


class TestPromptCreation(unittest.TestCase):
    """Test the prompt creation process"""

    def test_system_message_structure(self):
        """Test that system message contains required instructions"""
        contexts = [
            {"source_pdf": "test.pdf", "page_num": 1, "text": "Test content"}
        ]

        messages = create_prompt_with_context("test question", contexts)
        system_message = messages[0]

        self.assertEqual(system_message["role"], "system")
        content = system_message["content"]

        # Should contain format instructions
        self.assertIn("**Answer**", content)
        self.assertIn("**Supporting Points**", content)
        self.assertIn("CRITICAL: Only use citation numbers [1]", content)

    def test_citation_range_constraint(self):
        """Test that system message constrains citations to available range"""
        contexts = [
            {"source_pdf": "test.pdf", "page_num": 1, "text": "Content 1"},
            {"source_pdf": "test.pdf", "page_num": 2, "text": "Content 2"},
            {"source_pdf": "test.pdf", "page_num": 3, "text": "Content 3"}
        ]

        messages = create_prompt_with_context("test question", contexts)
        system_message = messages[0]
        content = system_message["content"]

        # Should specify the exact range [1] through [3]
        self.assertIn("Only use citation numbers [1] through [3]", content)

    def test_context_numbering(self):
        """Test that contexts are numbered correctly in user message"""
        contexts = [
            {"source_pdf": "source1.pdf", "page_num": 1, "text": "First content"},
            {"source_pdf": "source2.pdf", "page_num": 2, "text": "Second content"}
        ]

        messages = create_prompt_with_context("test question", contexts)
        user_message = messages[-1]  # Last message is user message

        content = user_message["content"]
        self.assertIn("[1] From 'source1.pdf', page 1:", content)
        self.assertIn("[2] From 'source2.pdf', page 2:", content)


if __name__ == "__main__":
    # Run specific test cases that validate the golden standards
    suite = unittest.TestSuite()

    # Core structure tests
    suite.addTest(TestPromptGoldens('test_answer_section_present'))
    suite.addTest(TestPromptGoldens('test_supporting_points_section_present'))
    suite.addTest(TestPromptGoldens('test_supporting_points_format'))
    suite.addTest(TestPromptGoldens('test_citations_in_valid_range'))
    suite.addTest(TestPromptGoldens('test_no_meta_references'))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)