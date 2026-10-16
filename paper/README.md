# SENTINEL v2 manuscript

- `main.tex` + `sections/*.tex` (IEEEtran). Build: `latexmk -pdf main.tex`.
- Rule: every number in the text comes from `results/` through the analysis scripts
  (`sentinel_sim/analyze.py`); no hand-typed results.
- `scripts/lint_language.py` enforces the banned-word list of REWORK_PLAN §6/Phase 8.
- `references.bib` is built only from entries verified in `docs/related_work_notes.md`.
