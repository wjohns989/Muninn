# Muninn MCP - Project Purpose & Architecture
Muninn is a local-first, assistant-agnostic persistent memory infrastructure. It uses a 6-signal hybrid retrieval engine (Vector, Graph, BM25, Temporal, Goal, Chain) with RRF fusion and late-interaction (ColBERT) reranking.

## Tech Stack
- **Core Engine**: Python (Native Engine, Stores, Extraction, Retrieval)
- **Vector Store**: Qdrant (INT8 Quantization, HNSW)
- **Graph Store**: Kuzu (Entity-linked memories)
- **Metadata Store**: SQLite (Bi-temporal queries, ACID)
- **Keyword Search**: BM25 (In-memory inverted index)
- **LLM Integrations**: Tiered pipeline (Rules -> Instructor/xLAM -> Ollama)
- **Transport**: MCP stdio wrapper
