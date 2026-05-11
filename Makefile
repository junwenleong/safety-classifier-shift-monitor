.PHONY: smoke canary finetune build-paraphrase

smoke:
	.venv/bin/python -m pytest tests/ --ignore=tests/integration -q --tb=short

canary:
	.venv/bin/python scripts/canary_run.py

finetune:
	@echo "Run on Mac Studio (169.254.1.2):"
	@echo "  ssh 169.254.1.2"
	@echo "  cd /path/to/safety-classifier-shift-monitor"
	@echo "  python scripts/finetune_deberta.py"
	@echo ""
	@echo "See scripts/README_finetune.md for details."

build-paraphrase:
	.venv/bin/python -c "from pathlib import Path; from shift_detection_monitor.stream.dataset_builder import ShiftDatasetBuilder; b = ShiftDatasetBuilder(use_bedrock=True); b.build('paraphrase', Path('data/reference/source.jsonl'), Path('data/shifted/paraphrase/output.jsonl'), seed=42)"
