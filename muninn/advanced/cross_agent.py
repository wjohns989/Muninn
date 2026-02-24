"""
Cross-Agent Federation Engine
-----------------------------
Enables decentralized memory sharing between autonomous agents.
Uses a Merkle-DAG inspired sync protocol to efficiently reconcile
memory states between disparate agent runtimes (e.g., Claude local vs.
Cloud hosted vs. IDE assistant).
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import httpx

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.Federation")

class FederationManager:
    def __init__(self, memory: "MuninnMemory"):
        self.memory = memory
        self.config = memory.config.federation

    async def broadcast_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Push a specific memory to all configured peers.
        """
        if not self.config.enabled or not self.config.peers:
            return {"status": "skipped", "reason": "federation_disabled_or_no_peers"}

        records = self.memory._metadata.get_by_ids([memory_id])
        if not records:
            return {"status": "failed", "reason": "memory_not_found"}

        record = records[0]
        bundle = await self.create_sync_bundle([memory_id], user_id=record.metadata.get("user_id", "global_user"))

        results = {}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            tasks = []
            for peer in self.config.peers:
                url = f"{peer.rstrip('/')}/federation/apply"
                tasks.append(self._push_to_peer(client, url, bundle))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for peer, resp in zip(self.config.peers, responses):
                if isinstance(resp, Exception):
                    results[peer] = {"status": "failed", "error": str(resp)}
                else:
                    results[peer] = resp

        return {"status": "completed", "results": results}

    async def _push_to_peer(self, client: httpx.AsyncClient, url: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # We need the auth token if peers are secured
            headers = {}
            if self.memory.config.server.auth_token:
                headers["Authorization"] = f"Bearer {self.memory.config.server.auth_token}"
            
            resp = await client.post(url, json=bundle, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"status": "failed", "code": resp.status_code, "text": resp.text}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def generate_manifest(
        self,
        project: str = "global",
        user_id: str = "global_user",
        namespace: Optional[str] = None,
        media_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a lightweight sync manifest describing the current state. 
        Uses project-scoped memory hashes to detect drift.
        """
        # Fetch all memory records for the scope
        # We use _metadata directly to get MemoryRecord objects
        records = await asyncio.to_thread(
            self.memory._metadata.get_all,
            limit=10000,
            project=project,
            user_id=user_id,
            namespace=namespace,
            media_type=media_type
        )

        manifest = {
            "project": project,
            "user_id": user_id,
            "namespace": namespace,
            "media_type": media_type,
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

    async def create_sync_bundle(self, memory_ids: List[str], user_id: str = "global_user") -> Dict[str, Any]:
        """Create a portable bundle for the requested IDs."""
        records = self.memory._metadata.get_by_ids(memory_ids)
        bundle = {
            "version": 1,
            "user_id": user_id,
            "memories": [
                {
                    "id": r.id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "type": r.memory_type.value,
                    "media_type": r.media_type.value,
                    "created_at": r.created_at
                }
                for r in records
                if r.metadata.get("user_id") == user_id or user_id == "global_user"
            ]
        }
        return bundle

    async def apply_sync_bundle(self, bundle: Dict[str, Any], user_id: Optional[str] = None) -> int:
        """Merge a sync bundle into local storage."""
        applied = 0
        target_user_id = user_id or bundle.get("user_id", "federated")
        for item in bundle.get("memories", []):
            try:
                # Upsert semantics
                await self.memory.add(
                    content=item["content"],
                    user_id=target_user_id,
                    metadata={**(item.get("metadata") or {}), "federation_sync": True},
                    media_type=item.get("media_type", "text"),
                    provenance="federated_sync"
                )
                applied += 1
            except Exception as e:
                logger.warning(f"Failed to sync memory {item.get('id')}: {e}")
        return applied