"""Unit tests for smart query parser."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from llm_parser import QueryIntent, SmartQueryParser


class SmartQueryParserTests(unittest.TestCase):
    def test_fallback_parse_splits_visual_and_location(self) -> None:
        parser = SmartQueryParser()
        with patch.object(parser.client, "generate", side_effect=TimeoutError("timeout")):
            intent = parser.parse("black cat in turkey")

        self.assertIsInstance(intent, QueryIntent)
        self.assertEqual(intent.parse_mode, "fallback")
        self.assertEqual(intent.visual_query, "black cat")
        self.assertEqual(intent.location_text, "turkey")
        self.assertIn("cat", [x.lower() for x in intent.objects])
        self.assertIn("black", [x.lower() for x in intent.attributes])
        self.assertGreaterEqual(len(intent.expanded_queries), 1)

    def test_llm_parse_accepts_valid_json(self) -> None:
        parser = SmartQueryParser()
        payload = {
            "visual_query": "black cat",
            "objects": ["cat"],
            "attributes": ["black"],
            "location_text": "Turkey",
            "time_text": None,
            "expanded_queries": ["black cat", "a photo of black cat"],
        }
        with patch.object(parser.client, "generate", return_value=json.dumps(payload)):
            intent = parser.parse("black cat in turkey")

        self.assertEqual(intent.parse_mode, "llm")
        self.assertEqual(intent.visual_query, "black cat")
        self.assertEqual(intent.location_text, "Turkey")
        self.assertEqual(intent.objects, ["cat"])

    def test_llm_bad_output_falls_back(self) -> None:
        parser = SmartQueryParser()
        with patch.object(parser.client, "generate", return_value="not-json"):
            intent = parser.parse("whiteboard in office")
        self.assertEqual(intent.parse_mode, "fallback")
        self.assertTrue(intent.visual_query)


if __name__ == "__main__":
    unittest.main()
