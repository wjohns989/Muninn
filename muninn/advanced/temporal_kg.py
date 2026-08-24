"""
Temporal Knowledge Graph Engine
-------------------------------
Adds bi-temporal reasoning capabilities to the Muninn graph store.
Enables queries like "What was true about Project X last week?" or
"Find valid relationships during the outage window."

Uses "Valid Time" (when the fact is true in the world) vs "Transaction Time"
(when the system recorded it, handled by core metadata).
"""

import logging
import time
from typing import List, Dict, Any, Optional
from muninn.store.graph_store import GraphStore

logger = logging.getLogger("Muninn.TemporalKG")

class TemporalKnowledgeGraph:
    def __init__(self, graph_store: GraphStore):
        self.graph = graph_store

    def initialize_schema(self):
        """Ensure temporal schema extensions exist in Kuzu."""
        conn = self.graph._get_conn()
        try:
            # Temporal relation extension: ValidTime
            # Allows tagging any edge with a validity window
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS VALID_DURING (
                    FROM Entity TO Entity,
                    start_time DOUBLE,
                    end_time DOUBLE,
                    predicate STRING,
                    source_memory STRING
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Temporal schema init: {e}")

    def add_temporal_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_start: float,
        source_memory: str,
        valid_end: Optional[float] = None,
    ) -> bool:
        """
        Record a fact that is true only for a specific time window.
        If valid_end is None, it is currently true (open-ended).
        """
        return self.add_temporal_facts_batch([{
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
            "valid_start": valid_start,
            "source_memory": source_memory,
            "valid_end": valid_end
        }])

    def add_temporal_facts_batch(self, facts: List[Dict[str, Any]]) -> bool:
        """Batch record facts that are true for specific time windows."""
        if not facts:
            return True

        # Pre-create all entities first to ensure they exist
        entities_to_add = []
        for f in facts:
            entities_to_add.append({"name": f["subject"], "entity_type": "unknown"})
            entities_to_add.append({"name": f["obj"], "entity_type": "unknown"})
        self.graph.add_entities_batch(entities_to_add)

        conn = self.graph._get_conn()
        
        facts_params = []
        for f in facts:
            valid_end = f.get("valid_end")
            end = valid_end if valid_end is not None else float("inf")
            facts_params.append({
                "subj": f["subject"],
                "obj": f["obj"],
                "pred": f["predicate"],
                "start": float(f["valid_start"]),
                "valid_until": float(end),
                "source_mem": f["source_memory"],
            })

        try:
            conn.execute(
                """
                UNWIND $facts AS f
                MATCH (a:Entity {name: f.subj}), (b:Entity {name: f.obj})
                CREATE (a)-[:VALID_DURING {predicate: f.pred, start_time: f.start, end_time: f.valid_until, source_memory: f.source_mem}]->(b)
                """,
                {"facts": facts_params}
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to batch add temporal facts: {e}")
            return False

    def shadow_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        superseded_at: float,
    ) -> bool:
        """
        Closes the validity window of an active temporal fact, creating a "Shadow Edge".
        This preserves the fact in history but bypasses it for current-day retrieval.
        """
        return self.shadow_edges_batch([{
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
            "superseded_at": superseded_at
        }])

    def shadow_edges_batch(self, edges: List[Dict[str, Any]]) -> bool:
        """Closes the validity window of multiple active temporal facts."""
        if not edges:
            return True

        conn = self.graph._get_conn()
        shadow_params = []
        for e in edges:
            shadow_params.append({
                "subj": e["subject"],
                "pred": e["predicate"],
                "obj": e["obj"],
                "ts": float(e["superseded_at"]),
            })

        try:
            conn.execute(
                """
                UNWIND $shadows AS s
                MATCH (a:Entity {name: s.subj})-[r:VALID_DURING {predicate: s.pred}]->(b:Entity {name: s.obj})
                WHERE r.end_time >= s.ts
                SET r.end_time = s.ts
                """,
                {"shadows": shadow_params}
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to batch shadow temporal edges: {e}")
            return False

    def shadow_memory_edges(
        self,
        memory_id: str,
        superseded_at: float,
    ) -> bool:
        """
        Closes the validity window of all active temporal facts sourced from a specific memory.
        """
        return self.shadow_memory_edges_batch([{"memory_id": memory_id, "superseded_at": superseded_at}])

    def shadow_memory_edges_batch(self, memories: List[Dict[str, Any]]) -> bool:
        """Closes the validity window of temporal facts sourced from specific memories."""
        if not memories:
            return True

        conn = self.graph._get_conn()
        shadow_params = []
        for m in memories:
            shadow_params.append({
                "mem_id": m["memory_id"],
                "ts": float(m["superseded_at"]),
            })

        try:
            conn.execute(
                """
                UNWIND $shadows AS s
                MATCH (:Entity)-[r:VALID_DURING {source_memory: s.mem_id}]->(:Entity)
                WHERE r.end_time >= s.ts
                SET r.end_time = s.ts
                """,
                {"shadows": shadow_params}
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to batch shadow memory edges: {e}")
            return False

    def query_valid_at(self, timestamp: float, limit: int = 50) -> List[Dict[str, Any]]:
        """Find all relationships valid at a specific point in time."""
        conn = self.graph._get_conn()
        facts = []
        try:
            result = conn.execute(
                """
                MATCH (a:Entity)-[r:VALID_DURING]->(b:Entity)
                WHERE r.start_time <= $ts AND r.end_time >= $ts
                RETURN a.name, r.predicate, b.name, r.start_time, r.end_time
                LIMIT $limit
                """,
                {"ts": float(timestamp), "limit": limit}
            )
            while result.has_next():
                row = result.get_next()
                facts.append({
                    "subject": row[0],
                    "predicate": row[1],
                    "object": row[2],
                    "valid_start": row[3],
                    "valid_end": row[4] if row[4] != float("inf") else None,
                })
        except Exception as e:
            logger.warning(f"Temporal query failed: {e}")
        return facts

    def snapshot_diff(self, t1: float, t2: float) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return facts that changed between t1 and t2.
        Added: Valid at t2 but not t1.
        Removed: Valid at t1 but not t2.
        """
        valid_t1 = self.query_valid_at(t1, limit=1000)
        valid_t2 = self.query_valid_at(t2, limit=1000)
        
        # Simple tuple set logic
        set_t1 = {(f["subject"], f["predicate"], f["object"]) for f in valid_t1}
        set_t2 = {(f["subject"], f["predicate"], f["object"]) for f in valid_t2}
        
        added_keys = set_t2 - set_t1
        removed_keys = set_t1 - set_t2
        
        added = [f for f in valid_t2 if (f["subject"], f["predicate"], f["object"]) in added_keys]
        removed = [f for f in valid_t1 if (f["subject"], f["predicate"], f["object"]) in removed_keys]
        
        return {"added": added, "removed": removed}