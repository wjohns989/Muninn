# Citations & Credits

Muninn is built upon and inspired by the following open-source projects and research:

## Core Infrastructure

- **Qdrant**: High-performance vector database. [qdrant.tech](https://qdrant.tech/) (Apache-2.0)
- **Kuzu**: Embedded graph database. [kuzudb.com](https://kuzudb.com/) (MIT)
- **FastAPI**: Modern async web framework. [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) (MIT)
- **SQLite**: Embedded relational database (public domain)

## Precision & Reranking

- **FastEmbed**: Efficient embedding and reranking. [github.com/qdrant/fastembed](https://github.com/qdrant/fastembed) (Apache-2.0)
- **Jina AI Reranker**: Cross-encoder reranking models. [jina.ai](https://jina.ai/) (Apache-2.0)

## Local LLM Infrastructure

- **Ollama**: Local LLM runner for embeddings and inference. [ollama.com](https://ollama.com/) (MIT)
- **xLAM**: Function-calling language models by Salesforce Research. [Salesforce/xLAM](https://huggingface.co/Salesforce/xLAM-2-1B-fc-r) (CC-BY-NC-4.0)
- **llama.cpp**: Inference engine for GGUF models. [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT)

## Research Foundations

### Memory Architecture
- McClelland, McNaughton & O'Reilly (1995) — *Complementary Learning Systems*
- Walker & Stickgold (2006) — *Sleep-Dependent Memory Consolidation*
- Tononi & Cirelli (2003) — *Synaptic Homeostasis Hypothesis*

### Retrieval & Ranking
- Khattab & Zaharia (2020) — *ColBERT: Efficient and Effective Passage Search*
- Croft, Metzler & Strohman (2009) — *Reciprocal Rank Fusion*
- Robertson & Zaragoza (2009) — *The Probabilistic Relevance Framework: BM25 and Beyond*

### Tool-Calling & Extraction
- Zhang et al. (2024) — *xLAM: A Family of Large Action Models for AI Agent Systems*
- Liu et al. (2024) — *PA-Tool: Aligning Tool Schemas for Small LMs*

### Cross-Domain Inspirations
- Matzinger (2002) — *The Danger Model: A Renewed Sense of Self* (Immunology → Importance Scoring)
- Rissanen (1978) — *Minimum Description Length Principle* (Information Theory → Consolidation)
- Shapiro et al. (2011) — *Conflict-Free Replicated Data Types* (Distributed Systems → Future Sync)

## Legacy Dependencies (Being Replaced)

- **Mem0**: The memory layer for AI Agents. [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) (Apache-2.0)
  - *Muninn is migrating to a native memory framework. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)*
