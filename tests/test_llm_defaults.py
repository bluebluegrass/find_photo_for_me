"""Unit tests for smart-query default settings helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from utils import default_llm_endpoint, default_llm_model, default_llm_timeout, default_location_mode


class LLMDefaultsTests(unittest.TestCase):
    def test_defaults_without_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(default_llm_model())
            self.assertTrue(default_llm_endpoint().startswith("http://"))
            self.assertGreater(default_llm_timeout(), 0)
            self.assertEqual(default_location_mode(), "hybrid")

    def test_env_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOCALPIX_LLM_MODEL": "test-model",
                "LOCALPIX_LLM_ENDPOINT": "http://127.0.0.1:1/test",
                "LOCALPIX_LLM_TIMEOUT": "7.5",
                "LOCALPIX_LOCATION_MODE": "strict",
            },
            clear=True,
        ):
            self.assertEqual(default_llm_model(), "test-model")
            self.assertEqual(default_llm_endpoint(), "http://127.0.0.1:1/test")
            self.assertEqual(default_llm_timeout(), 7.5)
            self.assertEqual(default_location_mode(), "strict")


if __name__ == "__main__":
    unittest.main()
