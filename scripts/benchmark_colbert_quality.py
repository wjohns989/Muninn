import os
import sys
import logging
import numpy as np
import time
import asyncio
from typing import List, Dict, Any

# Setup path to include muninn
sys.path.append(os.getcwd())

from muninn.retrieval.colbert_index import ColBERTIndexer
from muninn.retrieval.hybrid import HybridRetriever
from muninn.store.vector_store import VectorStore
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.graph_store import GraphStore
from muninn.retrieval.bm25 import BM25Index
from muninn.core.types import MemoryRecord
from muninn.core.feature_flags import FeatureFlags
from muninn.core.config import MuninnConfig, VectorConfig, MetadataConfig, GraphConfig

import traceback

import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ColBERTBenchmark")

async def run_benchmark():
    try:
        data_dir = "./temp_benchmark_colbert"
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        
        # Test queries and data
        test_data = [
            "Muninn is a local-first persistent memory infrastructure.",
            "ColBERT uses late interaction to achieve high precision retrieval.",
            "PLAID optimizes late interaction by using centroid-based filtering.",
            "Quantization reduces the memory footprint of vector embeddings.",
            "Scalar quantization to INT8 can save up to 75% space in Qdrant.",
            "Hybrid retrieval combines dense vectors, BM25, and graph signals.",
            "Knowledge Items are distilled snapshots of past conversations.",
            "Loki-Mode orchestrates autonomous multi-agent software development.",
            "The project follows SOTA+ standards for precision and quality.",
            "Memory records have importance scores for decay-based pruning."
        ]
        
        query = "How does ColBERT optimize memory efficiency?"
        
        results_summary = {}

        class MockTokenizer:
            def __call__(self, text, return_tensors=None, **kwargs):
                return {"input_ids": [[1] * len(text.split())]}
            def convert_ids_to_tokens(self, ids):
                return ["token"] * len(ids)

        class DummyEncoder:
            def __init__(self): 
                self.is_available = True
                self._tokenizer = MockTokenizer()
            def encode(self, text): 
                # Return dummy vectors (num_tokens, 128)
                tokens = text.split()
                return np.random.rand(max(1, len(tokens)), 128).astype(np.float32)
            def encode_query(self, query):
                return self.encode(query)

        def setup_system(flags: FeatureFlags, suffix: str):
            # Use a localized data dir for each config
            local_data_dir = os.path.join(data_dir, suffix)
            if not os.path.exists(local_data_dir):
                os.makedirs(local_data_dir)
                
            config = MuninnConfig(
                data_dir=local_data_dir,
                feature_flags=flags,
                vector=VectorConfig(path=os.path.join(local_data_dir, "qdrant")),
                metadata=MetadataConfig(path=os.path.join(local_data_dir, "metadata.db")),
                graph=GraphConfig(path=os.path.join(local_data_dir, "graph.db"))
            )
            
            metadata = SQLiteMetadataStore(db_path=config.metadata.path)
            vector_store = VectorStore(data_path=config.vector.path)
            graph = GraphStore(db_path=config.graph.path)
            bm25 = BM25Index()
            
            indexer = ColBERTIndexer(vector_store, config=config, encoder=DummyEncoder())
            retriever = HybridRetriever(
                metadata_store=metadata,
                vector_store=vector_store,
                graph_store=graph,
                bm25_index=bm25,
                colbert_indexer=indexer
            )
            retriever._config = config
            retriever._embed_fn = lambda x: [0.1] * 768 # Match VectorStore default (768)
            
            return metadata, indexer, retriever

        # 1. Baseline: FP32, No PLAID
        logger.info("### Stage 1: Baseline (FP32, No PLAID) ###")
        from dataclasses import replace
        flags_v1 = replace(FeatureFlags.from_env(), colbert=True, colbert_plaid=False, colbert_int8=False)
        
        metadata_v1, indexer_v1, retriever_v1 = setup_system(flags_v1, "fp32")
        
        for i, content in enumerate(test_data):
            mem_id = f"mem_{i}"
            metadata_v1.add(MemoryRecord(id=mem_id, content=content, user_id="u1", namespace="n1"))
            indexer_v1.index_text(mem_id, content)
            
        start = time.time()
        res_v1 = await retriever_v1.search(query, limit=5, user_id="u1")
        latency_v1 = (time.time() - start) * 1000
        results_summary["Baseline"] = {"results": [r.memory.id for r in res_v1], "latency": latency_v1}
        logger.info(f"Baseline Latency: {latency_v1:.2f}ms")

        # 2. INT8 Quantization
        logger.info("### Stage 2: INT8 Quantized ###")
        flags_v2 = replace(FeatureFlags.from_env(), colbert=True, colbert_plaid=False, colbert_int8=True)
        
        metadata_v2, indexer_v2, retriever_v2 = setup_system(flags_v2, "int8")
        
        for i, content in enumerate(test_data):
            mem_id = f"mem_{i}"
            metadata_v2.add(MemoryRecord(id=mem_id, content=content, user_id="u1", namespace="n1"))
            indexer_v2.index_text(mem_id, content)
            
        start = time.time()
        res_v2 = await retriever_v2.search(query, limit=5, user_id="u1")
        latency_v2 = (time.time() - start) * 1000
        results_summary["INT8"] = {"results": [r.memory.id for r in res_v2], "latency": latency_v2}
        logger.info(f"INT8 Latency: {latency_v2:.2f}ms")

        # 3. PLAID (Centroid Filtering) + INT8
        logger.info("### Stage 3: PLAID + INT8 ###")
        flags_v3 = replace(FeatureFlags.from_env(), colbert=True, colbert_plaid=True, colbert_int8=True)
        
        metadata_v3, indexer_v3, retriever_v3 = setup_system(flags_v3, "plaid")
        
        for i, content in enumerate(test_data):
            mem_id = f"mem_{i}"
            metadata_v3.add(MemoryRecord(id=mem_id, content=content, user_id="u1", namespace="n1"))
            indexer_v3.index_text(mem_id, content)
            
        start = time.time()
        res_v3 = await retriever_v3.search(query, limit=5, user_id="u1")
        latency_v3 = (time.time() - start) * 1000
        results_summary["PLAID+INT8"] = {"results": [r.memory.id for r in res_v3], "latency": latency_v3}
        logger.info(f"PLAID+INT8 Latency: {latency_v3:.2f}ms")

        # 4. Drift Monitoring
        logger.info("### Stage 4: Drift Monitoring ###")
        # Increase sample size to avoid population errors (need > 512 for default centroids)
        sample_vectors = np.random.rand(600, 128).astype(np.float32)
        sample_vectors /= np.linalg.norm(sample_vectors, axis=1, keepdims=True)
        drift_score = indexer_v3.check_centroid_relevance(sample_vectors)
        logger.info(f"Centroid Relevance (Drift): {drift_score:.4f}")
        
        logger.info("Triggering re-clustering (Self-Healing)...")
        success = indexer_v3.recluster_centroids(sample_vectors)
        logger.info(f"Re-clustering successful: {success}")

        # Final Comparison
        logger.info("\n" + "="*40)
        logger.info("QUANTITATIVE BENCHMARK RESULTS")
        logger.info("="*40)
        
        ref_set = set(results_summary["Baseline"]["results"])
        for mode, data in results_summary.items():
            if mode == "Baseline":
                recall = 1.0
            else:
                intersection = ref_set.intersection(set(data["results"]))
                recall = len(intersection) / len(ref_set) if ref_set else 1.0
                
            logger.info(f"{mode:12} | Latency: {data['latency']:6.2f}ms | Recall@5: {recall:.2%}")
        
        logger.info("="*40)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(run_benchmark())
