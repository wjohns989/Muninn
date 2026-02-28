# Phase 4Z Plan: Packaging Dependency Surface Alignment

Date: 2026-02-15  
Owner: Codex  
Status: Implemented in branch `feat/phase4v-task-metadata-cursor-compliance`

## Objective

Resolve the roadmap/package mismatch where optional feature surfaces were implemented in code but not represented as explicit install profiles in `pyproject.toml`.

## Implemented

1. Added optional dependency group:
   - `conflict`:
     - `transformers>=4.41.0`
     - `torch>=2.2.0`
2. Added optional dependency group:
   - `sdk`:
     - `requests>=2.31.0`
     - `httpx>=0.24.0`
3. Updated aggregate profile:
   - `all` now includes `conflict` and `sdk` in addition to existing extras.

## Validation

1. `python -c "import tomllib, pathlib; data=tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); extras=data['project']['optional-dependencies']; assert 'conflict' in extras and 'sdk' in extras and 'all' in extras; print('ok')"`
2. Result: `ok`

## ROI / Impact

1. Reduces install-profile ambiguity when enabling NLI conflict detection and SDK integrations.
2. Aligns packaging metadata with implemented roadmap capability surfaces.
3. Improves reproducibility of environment setup across local/dev/CI workflows.
