"""
Muninn Graph Store
------------------
Kuzu-based knowledge graph for entity relationships and graph-enhanced retrieval.
"""

import logging
import time
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import math

import kuzu

logger = logging.getLogger("Muninn.Graph")


class GraphStore:
    """Manages entity/relation knowledge graph in embedded Kuzu."""

    def __init__(self, db_path):
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self._db: Optional[kuzu.Database] = None
        self._thread_local = threading.local()
        # Initialize DB immediately to fail fast on lock errors
        self._get_db()
        self._initialize()

    def _get_db(self) -> kuzu.Database:
        if self._db is None:
            self._db = kuzu.Database(str(self.db_path))
        return self._db

    def _get_conn(self) -> kuzu.Connection:
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = kuzu.Connection(self._get_db())
        return self._thread_local.conn

    def _initialize(self):
        conn = self._get_conn()

        # Create node tables
        try:
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity (
                    name STRING,
                    entity_type STRING,
                    first_seen DOUBLE,
                    last_seen DOUBLE,
                    mention_count INT64 DEFAULT 1,
                    PRIMARY KEY (name)
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Entity table creation: {e}")

        try:
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Memory (
                    id STRING,
                    summary STRING,
                    created_at DOUBLE,
                    PRIMARY KEY (id)
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Memory table creation: {e}")

        # Create relationship tables
        try:
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                    FROM Entity TO Entity,
                    predicate STRING,
                    confidence DOUBLE DEFAULT 1.0,
                    source_memory STRING,
                    created_at DOUBLE
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"RELATES_TO table creation: {e}")

        try:
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS (
                    FROM Memory TO Entity,
                    role STRING DEFAULT 'mention'
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"MENTIONS table creation: {e}")

        try:
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS PRECEDES (
                    FROM Memory TO Memory,
                    confidence DOUBLE DEFAULT 1.0,
                    reason STRING,
                    shared_entities_json STRING,
                    hours_apart DOUBLE,
                    created_at DOUBLE
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"PRECEDES table creation: {e}")

        try:
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CAUSES (
                    FROM Memory TO Memory,
                    confidence DOUBLE DEFAULT 1.0,
                    reason STRING,
                    shared_entities_json STRING,
                    hours_apart DOUBLE,
                    created_at DOUBLE
                )
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"CAUSES table creation: {e}")

        logger.info(f"Graph store initialized at {self.db_path}")

    def add_entity(self, name: str, entity_type: str) -> bool:
        conn = self._get_conn()
        now = time.time()
        try:
            # Try to merge (upsert)
            conn.execute(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.entity_type = $type, e.first_seen = $now, e.last_seen = $now, e.mention_count = 1 "
                "ON MATCH SET e.last_seen = $now, e.mention_count = e.mention_count + 1",
                {"name": name, "type": entity_type, "now": now}
            )
            return True
        except Exception as e:
            logger.debug(f"Entity upsert for '{name}': {e}")
            return False

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_memory_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> bool:
        conn = self._get_conn()
        now = time.time()

        # Ensure both entities exist
        self.add_entity(subject, "unknown")
        self.add_entity(obj, "unknown")

        try:
            conn.execute(
                "MATCH (a:Entity {name: $subj}), (b:Entity {name: $obj}) "
                "CREATE (a)-[:RELATES_TO {predicate: $pred, confidence: $conf, "
                "source_memory: $src, created_at: $now}]->(b)",
                {
                    "subj": subject, "obj": obj, "pred": predicate,
                    "conf": confidence, "src": source_memory_id or "", "now": now
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Relation creation: {e}")
            return False

    def add_memory_node(self, memory_id: str, summary: str) -> bool:
        conn = self._get_conn()
        now = time.time()
        try:
            conn.execute(
                "MERGE (m:Memory {id: $id}) "
                "ON CREATE SET m.summary = $summary, m.created_at = $now "
                "ON MATCH SET m.summary = $summary",
                {"id": memory_id, "summary": summary[:500], "now": now}
            )
            return True
        except Exception as e:
            logger.debug(f"Memory node creation: {e}")
            return False

    def link_memory_to_entity(self, memory_id: str, entity_name: str, role: str = "mention") -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "MATCH (m:Memory {id: $mid}), (e:Entity {name: $ename}) "
                "CREATE (m)-[:MENTIONS {role: $role}]->(e)",
                {"mid": memory_id, "ename": entity_name, "role": role}
            )
            return True
        except Exception as e:
            logger.debug(f"Memory-entity link: {e}")
            return False

    def find_related_memories(self, query_entities: List[str], limit: int = 20) -> List[str]:
        """Find memory IDs related to given entity names via graph traversal."""
        if not query_entities:
            return []

        conn = self._get_conn()
        memory_ids = set()

        for entity_name in query_entities:
            try:
                # Direct mentions
                result = conn.execute(
                    "MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $name}) "
                    "RETURN m.id LIMIT $limit",
                    {"name": entity_name, "limit": limit}
                )
                while result.has_next():
                    row = result.get_next()
                    memory_ids.add(row[0])

                # 2-hop: memories mentioning entities related to query entity
                result = conn.execute(
                    "MATCH (m:Memory)-[:MENTIONS]->(e1:Entity)-[:RELATES_TO]-(e2:Entity {name: $name}) "
                    "RETURN DISTINCT m.id LIMIT $limit",
                    {"name": entity_name, "limit": limit}
                )
                while result.has_next():
                    row = result.get_next()
                    memory_ids.add(row[0])
            except Exception as e:
                logger.debug(f"Graph search for '{entity_name}': {e}")

        return list(memory_ids)[:limit]

    def get_entity_centrality(self, entity_name: str) -> float:
        """Get degree centrality of an entity (normalized by max possible degree)."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "MATCH (e:Entity {name: $name})-[r:RELATES_TO]-() RETURN COUNT(r)",
                {"name": entity_name}
            )
            if result.has_next():
                degree = result.get_next()[0]
                # Normalize: Use log-scaling to handle high-degree nodes without losing resolution.
                # Normalized centrality = log(1 + degree) / log(1 + 100)
                # This gives 1.0 at degree 100, but keeps scaling (slower) above that.
                # w["centrality"] is 0.20, so high degree nodes get full boost.
                return min(1.0, math.log1p(degree) / math.log1p(100))
        except Exception:
            pass
        return 0.0

    def get_entity_count(self) -> int:
        conn = self._get_conn()
        try:
            result = conn.execute("MATCH (e:Entity) RETURN COUNT(e)")
            if result.has_next():
                return result.get_next()[0]
        except Exception:
            pass
        return 0

    def get_all_entities(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        entities = []
        try:
            result = conn.execute(
                "MATCH (e:Entity) RETURN e.name, e.entity_type, e.mention_count "
                "ORDER BY e.mention_count DESC LIMIT $limit",
                {"limit": limit}
            )
            while result.has_next():
                row = result.get_next()
                entities.append({
                    "name": row[0],
                    "entity_type": row[1],
                    "mention_count": row[2],
                })
        except Exception as e:
            logger.debug(f"Get all entities: {e}")
        return entities

    def add_chain_link(
        self,
        predecessor_id: str,
        successor_id: str,
        *,
        relation_type: str = "PRECEDES",
        confidence: float = 1.0,
        reason: str = "",
        shared_entities: Optional[List[str]] = None,
        hours_apart: Optional[float] = None,
    ) -> bool:
        """
        Add a directed memory-to-memory chain edge.
        """
        conn = self._get_conn()
        rel = str(relation_type or "PRECEDES").upper()
        if rel not in {"PRECEDES", "CAUSES"}:
            return False
        if predecessor_id == successor_id:
            return False
        now = time.time()
        payload = json.dumps(shared_entities or [], ensure_ascii=False)
        conf = max(0.0, min(1.0, float(confidence)))
        hours = float(hours_apart) if hours_apart is not None else None
        try:
            conn.execute(
                f"MATCH (a:Memory {{id: $pred}}), (b:Memory {{id: $succ}}) "
                f"CREATE (a)-[:{rel} {{confidence: $conf, reason: $reason, "
                f"shared_entities_json: $shared, hours_apart: $hours, created_at: $now}}]->(b)",
                {
                    "pred": predecessor_id,
                    "succ": successor_id,
                    "conf": conf,
                    "reason": reason[:500],
                    "shared": payload,
                    "hours": hours,
                    "now": now,
                },
            )
            return True
        except Exception as e:
            logger.debug(f"Chain relation creation ({rel}): {e}")
            return False

    def find_chain_related_memories(
        self,
        seed_memory_ids: List[str],
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Find memories related via PRECEDES/CAUSES edges around given seeds.
        Returns list of (memory_id, score) ranked by accumulated confidence.
        Optimized to batch query all seeds at once.
        """
        if not seed_memory_ids:
            return []
        conn = self._get_conn()
        seed_set = set(seed_memory_ids)
        scores: Dict[str, float] = {}
        # Kuzu might have limits on list size in IN clause, but 12-20 is fine.
        max_seed_scan = max(1, min(len(seed_memory_ids), 20))
        target_seeds = seed_memory_ids[:max_seed_scan]

        def _fetch_rows(query: str, params: Dict[str, Any]) -> List[Tuple[Any, ...]]:
            result = conn.execute(query, params)
            if hasattr(result, "fetchall"):
                try:
                    return result.fetchall()
                except Exception:
                    pass
            rows: List[Tuple[Any, ...]] = []
            while result.has_next():
                rows.append(result.get_next())
            return rows

        def _accumulate(rows, multiplier: float) -> None:
            for mem_id, conf in rows:
                if mem_id in seed_set:
                    continue
                confidence = float(conf) if conf is not None else 0.5
                score = max(0.0, min(1.0, confidence)) * multiplier
                prev = scores.get(mem_id, 0.0)
                scores[mem_id] = max(prev, score)

        try:
            # Outgoing PRECEDES
            out_pre = _fetch_rows(
                "MATCH (a:Memory)-[r:PRECEDES]->(b:Memory) "
                "WHERE list_contains($ids, a.id) "
                "RETURN b.id, r.confidence LIMIT $limit",
                {"ids": target_seeds, "limit": limit * 2},
            )
            _accumulate(out_pre, 1.0)

            # Outgoing CAUSES
            out_cau = _fetch_rows(
                "MATCH (a:Memory)-[r:CAUSES]->(b:Memory) "
                "WHERE list_contains($ids, a.id) "
                "RETURN b.id, r.confidence LIMIT $limit",
                {"ids": target_seeds, "limit": limit * 2},
            )
            _accumulate(out_cau, 1.15)

            # Incoming PRECEDES
            in_pre = _fetch_rows(
                "MATCH (a:Memory)-[r:PRECEDES]->(b:Memory) "
                "WHERE list_contains($ids, b.id) "
                "RETURN a.id, r.confidence LIMIT $limit",
                {"ids": target_seeds, "limit": limit * 2},
            )
            _accumulate(in_pre, 0.9)

            # Incoming CAUSES
            in_cau = _fetch_rows(
                "MATCH (a:Memory)-[r:CAUSES]->(b:Memory) "
                "WHERE list_contains($ids, b.id) "
                "RETURN a.id, r.confidence LIMIT $limit",
                {"ids": target_seeds, "limit": limit * 2},
            )
            _accumulate(in_cau, 1.05)

        except Exception as e:
            logger.debug(f"Chain traversal batch failed: {e}")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def delete_memory_references(self, memory_id: str) -> bool:
        """Remove all graph references for a memory."""
        conn = self._get_conn()
        try:
            conn.execute("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id})
            return True
        except Exception as e:
            logger.debug(f"Delete memory references: {e}")
            return False

    def close(self):
        # We cannot easily close all thread-local connections, but Kuzu handles cleanup
        # when the database object is destroyed or process exits.
        self._db = None
        # Clear current thread's connection if it exists
        if hasattr(self._thread_local, "conn"):
            del self._thread_local.conn