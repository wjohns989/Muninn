import multiprocessing
import time
import traceback
from pathlib import Path

import pytest

from muninn.core.types import MemoryRecord
from muninn.store.graph_store import GraphStore
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore


def perform_sqlite_writes(db_path, num_writes, process_id):
    try:
        store = SQLiteMetadataStore(db_path)
        for i in range(num_writes):
            mem_id = f"p{process_id}-m{i}"
            record = MemoryRecord(
                id=mem_id,
                content=f"Content from process {process_id}, write {i}",
                created_at=time.time(),
                importance=0.5
            )
            store.add(record)
            time.sleep(0.01)
    except Exception:
        print(f"Process {process_id} SQLite error:\n{traceback.format_exc()}")
        raise

def test_sqlite_concurrency(tmp_path):
    db_path = tmp_path / "metadata.db"
    num_processes = 4
    writes_per_process = 20

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=perform_sqlite_writes,
            args=(db_path, writes_per_process, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0

    # Verify all records were written
    store = SQLiteMetadataStore(db_path)
    all_memories = store.get_all(limit=num_processes * writes_per_process)
    assert len(all_memories) == num_processes * writes_per_process

def hold_store(kind, path, ready, release):
    """Open an embedded store, write once, and keep it open until released."""
    if kind == "vector":
        store = VectorStore(path, embedding_dims=4)
        store.upsert("owner", [0.1, 0.2, 0.3, 0.4], {})
    else:
        store = GraphStore(path)
        store.add_memory_node("owner", "owned", user_id="u", namespace="n")
    ready.set()
    release.wait(30)


def _open_and_write(kind, path):
    if kind == "vector":
        VectorStore(path, embedding_dims=4).upsert("intruder", [0.1, 0.2, 0.3, 0.4], {})
    else:
        GraphStore(path).add_memory_node("intruder", "x", user_id="u", namespace="n")


# Embedded Qdrant and Kuzu are single-process stores. Muninn relies on that:
# one machine-wide server owns them and every client goes through it. These
# tests pin the contract that a second process is refused immediately with a
# clear error instead of blocking or corrupting the owner's data.
@pytest.mark.parametrize(
    "kind, dirname, message",
    [("vector", "vector_store", "already accessed"), ("graph", "graph_db", "lock")],
)
def test_embedded_store_refuses_second_process(tmp_path, kind, dirname, message):
    path = tmp_path / dirname
    ready, release = multiprocessing.Event(), multiprocessing.Event()
    owner = multiprocessing.Process(target=hold_store, args=(kind, path, ready, release))
    owner.start()
    try:
        assert ready.wait(30), "owner process never opened the store"
        with pytest.raises(RuntimeError, match=message):
            _open_and_write(kind, path)
    finally:
        release.set()
        owner.join(30)
    assert owner.exitcode == 0

    # Once the owner has exited, the store opens normally.
    _open_and_write(kind, path)

if __name__ == "__main__":
    # Setup for manual run if needed
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_sqlite_concurrency(Path(tmp))
        print("SQLite concurrency test passed")
