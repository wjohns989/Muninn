"""
Cross-Agent Federation Engine
-----------------------------
Enables decentralized memory sharing between autonomous agents.
Uses a Merkle-DAG inspired sync protocol to efficiently reconcile
memory states between disparate agent runtimes (e.g., Claude local vs.
Cloud hosted vs. IDE assistant).
"""

import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.Federation")

class FederationManager:
    def __init__(self, memory: "MuninnMemory"):
        self.memory = memory

    async def generate_manifest(self, project: str = "global") -> Dict[str, Any]:
        """
        Generate a lightweight sync manifest describing the current state.
        Uses project-scoped memory hashes to detect drift.
        """
        # Fetch all memory IDs and timestamps for the scope
        # This is a naive O(N) scan; production would use a Merkle tree in DB
        records = await self.memory.get_all(limit=10000, project=project)
        
        manifest = {
            "project": project,
            "count": len(records),
            "heads": {}, # user_id -> latest_hash
            "ids": [],   # List of (id, hash) tuples
        }
        
        ids = []
        for r in records:
            # Deterministic content hash
            content_hash = hashlib.sha256(r.content.encode()).hexdigest()[:16]
            ids.append((r.id, content_hash))
            
        manifest["ids"] = sorted(ids, key=lambda x: x[0])
        return manifest

    async def calculate_delta(
        self, 
        local_manifest: Dict[str, Any], 
        remote_manifest: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Compare local and remote manifests to determine missing memories.
        """
        local_set = set(tuple(x) for x in local_manifest.get("ids", []))
        remote_set = set(tuple(x) for x in remote_manifest.get("ids", []))
        
        # Memories remote has but local lacks
        missing_ids = [mid for mid, mhash in (remote_set - local_set)]
        
        # Memories local has but remote lacks (offer to push)
        offer_ids = [mid for mid, mhash in (local_set - remote_set)]
        
        return {
            "missing": missing_ids,
            "offer": offer_ids
        }

    async def create_sync_bundle(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Create a portable bundle for the requested IDs."""
        records = self.memory._metadata.get_by_ids(memory_ids)
        bundle = {
            "version": 1,
            "memories": [
                {
                    "id": r.id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "type": r.memory_type.value,
                    "created_at": r.created_at
                }
                for r in records
            ]
        }
        return bundle

    async def apply_sync_bundle(self, bundle: Dict[str, Any]) -> int:
        """Merge a sync bundle into local storage."""
        applied = 0
        for item in bundle.get("memories", []):
            try:
                # Upsert semantics
                await self.memory.add(
                    content=item["content"],
                    user_id=(item.get("metadata") or {}).get("user_id", "federated"),
                    metadata={**(item.get("metadata") or {}), "federation_sync": True},
                    provenance="federated_sync"
                )
                applied += 1
            except Exception as e:
                logger.warning(f"Failed to sync memory {item.get('id')}: {e}")
        return applied