# Development Commands
## Installation
- `pip install -e .[all]`

## Verification & Testing
- **Test**: `pytest` (Standard suite: 500+ tests)
- **Lint**: `ruff check .`
- **Benchmark**: `python -m eval.ollama_local_benchmark sota-verdict`
- **Hygiene**: `python -m eval.phase_hygiene`

## Specific Verification
- **ColBERT Efficiency**: `python scripts/verify_colbert_efficiency.py`
- **ColBERT v3.5**: `python scripts/verify_colbert_v3_5.py`
- **MCP Wrapper**: `pytest tests/test_mcp_wrapper_protocol.py`
