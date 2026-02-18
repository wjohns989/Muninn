"""
ColBERT Indexing & Multi-Vector Pipeline
----------------------------------------
Handles tokenization, encoding, and storage of multi-vector embeddings
in Qdrant for late interaction matching.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client.models import PointStruct, Distance, VectorParams

from muninn.store.vector_store import VectorStore

logger = logging.getLogger("Muninn.ColBERTIndex")


class ColBERTEncoder:
    """
    Handles tokenization and encoding for late interaction.
    Uses a transformer model with a linear projection layer.
    """
    
    DEFAULT_MODEL = "colbert-ir/colbertv2.0"
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._tokenizer = None
        self._model = None
        self._available = False
        self._initialize()

    def _initialize(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._available = True
            logger.info(f"ColBERT Encoder initialized: model={self.model_name}")
        except Exception as e:
            logger.warning(f"ColBERT Encoder initialization failed: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def encode(self, text: str) -> np.ndarray:
        """Encode text into token-level embeddings."""
        if not self._available:
            return np.array([])
            
        import torch
        
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # ColBERT uses the embedding of each token
            # shape: (1, num_tokens, hidden_dim)
            embeddings = outputs.last_hidden_state
            
            # If the model has a linear projection layer (linear), apply it.
            # Official ColBERT checkpoints usually have a 'linear' attribute.
            if hasattr(self._model, "linear"):
                embeddings = self._model.linear(embeddings)
            elif embeddings.shape[-1] != 128:
                # If not standard ColBERT dim, we might need a projection or just use as is.
                # For v3.5.0 in this environment, we'll assume the model is compatible
                # or we'd need to train/load a specific projection.
                pass
            
            # Normalize embeddings (ColBERT requirement)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            
            return embeddings[0].cpu().numpy()

class ColBERTIndexer:
    """
    Manages the ColBERT indexing pipeline.
    Encodes documents into token-level vectors and stores them in Qdrant.
    """

    def __init__(
        self, 
        vector_store: VectorStore, 
        collection_name: str = "muninn_colbert_tokens",
        encoder: Optional[ColBERTEncoder] = None,
        config: Optional[Any] = None
    ):
        self.vectors = vector_store
        self.collection_name = collection_name
        self.encoder = encoder or ColBERTEncoder()
        from muninn.core.feature_flags import get_flags
        self.config = config or get_flags()
        self._initialized = False
        self._centroid_cache: Optional[np.ndarray] = None
        self._centroid_ids: Optional[np.ndarray] = None

    def _get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Helper to safely get a feature flag."""
        if not self.config or not hasattr(self.config, "feature_flags") or self.config.feature_flags is None:
            return default
        return getattr(self.config.feature_flags, flag_name, default)

    def _ensure_collection(self):
        """Ensure the token-level collection exists with correct dimensions."""
        if self._initialized:
            return
        client = self.vectors._get_client()
        
        # Determine actual dimension from encoder
        dim = 128 # Default fallback
        if self.encoder.is_available:
            sample_vecs = self.encoder.encode("sample")
            if sample_vecs.size > 0:
                dim = sample_vecs.shape[-1]
                logger.info(f"Detected ColBERT encoder dimension: {dim}")

        from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType
        
        collections = [c.name for c in client.get_collections().collections]
        if self.collection_name not in collections:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            logger.info(f"Created ColBERT token collection '{self.collection_name}' ({dim} dims) with INT8 quantization")
        self._initialized = True

    # Static list of common English stop-words to reduce vector count
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is", "are", 
        "was", "were", "be", "been", "being", "it", "that", "this", "these", "those", 
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", 
        "her", "hers", "it", "its", "they", "them", "their"
    }

    def _should_prune(self, token: str) -> bool:
        """Determine if a token should be pruned."""
        token = token.lower().strip()
        # Prune punctuation and ultra-short tokens
        if not any(c.isalnum() for c in token) or len(token) < 2:
            return True
        return token in self.STOP_WORDS

    def index_text(self, memory_id: str, content: str):
        """Encode and index text with token pruning."""
        # Check feature flag
        if not self._get_feature_flag("colbert") or not self.encoder.is_available:
            return
            
        # Get raw vectors and tokens
        inputs = self.encoder._tokenizer(
            content,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokens = self.encoder._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        vectors = self.encoder.encode(content)
        if vectors.size == 0 or len(tokens) != vectors.shape[0]:
            return
            
        # Apply pruning
        filtered_vectors = []
        for i, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]:
                continue
            if self._should_prune(token):
                continue
            filtered_vectors.append(vectors[i])
            
        if not filtered_vectors:
            return
            
        self.index_memory(memory_id, content, np.array(filtered_vectors))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _ensure_centroid_collection(self):
        """Ensure the centroid collection exists for token pruning."""
        if not self.config.feature_flags.colbert_plaid:
            return

        client = self.vectors._get_client()
        collections = [c.name for c in client.get_collections().collections]
        centroid_collection = f"{self.collection_name}_centroids"
        
        if centroid_collection not in collections:
            # We'll use 512 centroids for a moderate memory store
            dim = 768 # Standard ColBERT dim
            if self.encoder.is_available:
                sample = self.encoder.encode("sample")
                if sample.size > 0: dim = sample.shape[-1]

            client.create_collection(
                collection_name=centroid_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            
            # Initialize centroids from encoder's embeddings or random
            # For a real PLAID implementation, we'd use KMeans on a sample.
            # Here we'll generate 512 random normalized vectors as a baseline.
            centroids = np.random.randn(512, dim)
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            
            client.upsert(
                collection_name=centroid_collection,
                points=[
                    PointStruct(id=i, vector=centroids[i].tolist(), payload={"centroid_id": i})
                    for i in range(512)
                ]
            )
            logger.info(f"Initialized {len(centroids)} centroids for collection {centroid_collection}")

    def _load_centroids(self):
        """Load centroids into memory for fast local search."""
        if not self.config.feature_flags.colbert_plaid:
            return
            
        if not self._get_feature_flag("colbert_plaid", False):
            return
            
        if self._centroid_cache is not None:
            return
            
        client = self.vectors._get_client()
        centroid_collection = f"{self.collection_name}_centroids"
        
        # Double check flag before accessing collection
        if not self._get_feature_flag("colbert_plaid", False):
            return

        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if centroid_collection not in collections:
            self._ensure_centroid_collection()
            
            # Re-check after attempt
            collections = [c.name for c in client.get_collections().collections]
            if centroid_collection not in collections:
                logger.warning("Centroid collection still missing after ensure_centroid_collection. Skipping cache load.")
                return
            
        # Scroll all centroids (usually 512)
        results = client.scroll(
            collection_name=centroid_collection,
            limit=1000,
            with_vectors=True
        )[0]
        
        if results:
            self._centroid_cache = np.array([p.vector for p in results])
            self._centroid_ids = np.array([int(p.id) for p in results])
            logger.info(f"Loaded {len(results)} centroids into memory cache")

    def _get_nearest_centroid(self, vector: np.ndarray) -> int:
        """Find the nearest centroid ID for a given vector."""
        # 1. Try local cache first (O(1) in terms of IO, O(N) in terms of matrix mult)
        if self._centroid_cache is not None:
            # Cosine similarity is just dot product since vectors are normalized
            # shape: (num_centroids,)
            scores = np.dot(self._centroid_cache, vector)
            idx = np.argmax(scores)
            return int(self._centroid_ids[idx])

        # 2. Fallback to Qdrant query if not cached
        client = self.vectors._get_client()
        centroid_collection = f"{self.collection_name}_centroids"
        
        results = client.query_points(
            collection_name=centroid_collection,
            query=vector.tolist(),
            limit=1
        ).points
        
        if results:
            return int(results[0].id)
        return -1

    def index_memory(self, memory_id: str, content: str, vectors: np.ndarray):
        """
        Index a memory record at the token level with centroid assignment.
        """
        self._ensure_collection()
        self._ensure_centroid_collection()
        self._load_centroids()
        
        client = self.vectors._get_client()
        
        points = []
        num_tokens = vectors.shape[0]
        
        plaid_enabled = self._get_feature_flag("colbert_plaid", False)
        
        for i in range(num_tokens):
            centroid_id = self._get_nearest_centroid(vectors[i]) if plaid_enabled else 0
            token_point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, f"{memory_id}_{i}"))
            
            points.append(PointStruct(
                id=token_point_id,
                vector=vectors[i].tolist(),
                payload={
                    "memory_id": memory_id,
                    "token_index": i,
                    "total_tokens": num_tokens,
                    "centroid_id": centroid_id
                }
            ))

        client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.debug(f"Indexed {num_tokens} tokens with centroids for memory {memory_id}")

    def get_document_vectors(self, memory_id: str) -> np.ndarray:
        """
        Retrieve all token vectors for a given document.
        """
        self._ensure_collection()
        client = self.vectors._get_client()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        results = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
            ),
            limit=512,
            with_vectors=True
        )
        
        points = results[0]
        if not points:
            return np.array([])
            
        indexed_vectors = sorted(points, key=lambda x: x.payload.get("token_index", 0))
        return np.array([p.vector for p in indexed_vectors])

    def get_query_centroids(self, query_vectors: np.ndarray, top_k: int = 8) -> List[int]:
        """Get the union of top-K centroids for a set of query vectors."""
        self._ensure_centroid_collection()
        client = self.vectors._get_client()
        centroid_collection = f"{self.collection_name}_centroids"
        
        centroid_ids = set()
        for vec in query_vectors:
            results = client.query_points(
                collection_name=centroid_collection,
                query=vec.tolist(),
                limit=top_k
            ).points
            for hit in results:
                centroid_ids.add(int(hit.id))
        
        return list(centroid_ids)

    def check_centroid_relevance(self, sample_vectors: np.ndarray) -> float:
        """
        Calculate the average distance to the nearest centroid.
        Used to detect drift and trigger re-clustering.
        """
        if self._centroid_cache is None:
            self._load_centroids()
            
        if self._centroid_cache is None:
            return 1.0 # Max distance if no centroids
            
        # For each sample vector, find max cosine similarity
        # sample_vectors: (M, dim), centroid_cache: (N, dim)
        # result: (M, N)
        similarities = np.dot(sample_vectors, self._centroid_cache.T)
        max_sims = np.max(similarities, axis=1)
        
        # Return average 'distance' (1 - cosine similarity)
        avg_drift = 1.0 - np.mean(max_sims)
        logger.info(f"ColBERT Centroid Drift: {avg_drift:.4f}")
        return float(avg_drift)

    def recluster_centroids(self, sample_vectors: np.ndarray):
        """
        Re-generate centroids using KMeans on a sample of document vectors.
        Requires scipy.
        """
        try:
            from scipy.cluster.vq import kmeans2
            
            num_centroids = len(self._centroid_cache) if self._centroid_cache is not None else 512
            
            # Population guard: K-Means needs at least as many points as clusters
            if len(sample_vectors) < num_centroids:
                logger.warning(
                    f"ColBERT re-clustering aborted: population ({len(sample_vectors)}) "
                    f"is smaller than cluster count ({num_centroids}). "
                    "Need more document vectors to re-estimate centroids."
                )
                return False

            centroids, _ = kmeans2(sample_vectors, num_centroids, minit='points')
            
            # Normalize centroids
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            
            # Update collection
            client = self.vectors._get_client()
            centroid_collection = f"{self.collection_name}_centroids"
            
            # Clear existing
            client.delete_collection(centroid_collection)
            self._ensure_centroid_collection() # Re-create
            
            client.upsert(
                collection_name=centroid_collection,
                points=[
                    PointStruct(id=i, vector=centroids[i].tolist(), payload={"centroid_id": i})
                    for i in range(len(centroids))
                ]
            )
            
            # Invalidate cache
            self._centroid_cache = None
            self._load_centroids()
            
            logger.info(f"Successfully re-clustered {len(centroids)} centroids")
            return True
        except Exception as e:
            logger.error(f"Re-clustering failed: {e}")
            return False
