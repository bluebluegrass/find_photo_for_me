#!/usr/bin/env bash
set -euo pipefail

python3.11 -m unittest discover -s tests -p "test_*.py"
python3.11 -m compileall app.py indexer.py llm_parser.py searcher.py store.py ui.py simple_ui.py utils.py

echo "Quality gate passed."
