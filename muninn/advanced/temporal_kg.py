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
                    predicate STRING
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
        valid_end: Optional[float] = None,
    ) -> bool:
        """
        Record a fact that is true only for a specific time window.
        If valid_end is None, it is currently true (open-ended).
        """
        conn = self.graph._get_conn()
        # Ensure entities exist
        self.graph.add_entity(subject, "unknown")
        self.graph.add_entity(obj, "unknown")
        
        end = valid_end if valid_end is not None else float("inf")
        
        try:
            conn.execute(
                "MATCH (a:Entity {name: $subj}), (b:Entity {name: $obj}) "
                "CREATE (a)-[:VALID_DURING {predicate: $pred, start_time: $start, end_time: $valid_until}]->(b)",
                {
                    "subj": subject,
                    "obj": obj,
                    "pred": predicate,
                    "start": float(valid_start),
                    "valid_until": float(end),
                }
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to add temporal fact: {e}")
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