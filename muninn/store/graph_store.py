"""
Muninn Graph Store
------------------
Kuzu-based knowledge graph for entity relationships and graph-enhanced retrieval.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import kuzu

logger = logging.getLogger("Muninn.Graph")


class GraphStore:
    """Manages entity/relation knowledge graph in embedded Kuzu."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: Optional[kuzu.Database] = None
        self._conn: Optional[kuzu.Connection] = None
        self._initialize()

    def _get_conn(self) -> kuzu.Connection:
        if self._db is None:
            self._db = kuzu.Database(str(self.db_path))
        if self._conn is None:
            self._conn = kuzu.Connection(self._db)
        return self._conn

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
                # Normalize: assume max degree ~50 for practical purposes
                return min(1.0, degree / 50.0)
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
        self._conn = None
        self._db = None
