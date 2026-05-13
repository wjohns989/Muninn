Let's review the code to ensure `asyncio.to_thread` works. `asyncio.to_thread` is available in Python 3.9+. The repository uses Python 3.12 (as seen from `pyenv` and `pyproject.toml`).

Let's check `muninn/mcp/handlers.py` and modify it.

We need to instantiate the global loop:
```python
_task_loop = asyncio.new_event_loop()
def _start_task_loop():
    asyncio.set_event_loop(_task_loop)
    _task_loop.run_forever()

threading.Thread(target=_start_task_loop, daemon=True, name="MuninnTaskLoop").start()
```

Where to put this? `muninn/mcp/handlers.py` has a global area:
```python
logger = logging.getLogger("Muninn.mcp.handlers")

_thread_local = threading.local()
```
I can place it right under `_thread_local = threading.local()`.

Let's do this modification.
