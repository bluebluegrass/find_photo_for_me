# Release Notes

## Smart Query Rollout

This release adds local LLM-assisted Smart Query to LocalPix.

Highlights:

- Added local smart-query parser with strict normalization and fallback behavior.
- Added multi-prompt search scoring path for parsed prompt expansions.
- Added CLI smart-query controls:
  - `--smart-query`
  - `--show-parse`
  - `--llm-model`
  - `--llm-timeout`
  - `--llm-endpoint`
  - `--media-filter`
- Added Streamlit smart-query controls and parsed-intent preview.
- Centralized smart-query defaults and env-based overrides.
- Added unit tests for parser behavior, prompt scoring, and LLM default resolution.

Quality gate command:

```bash
scripts/quality_gate.sh
```
