#!/usr/bin/env python3
"""
Muninn MCP Wrapper
------------------
Acts as a bridge between MCP clients (Claude Desktop, etc.) and the Muninn Memory Server.
Auto-starts the server if it's not running.
"""

import sys
import os
import time
import json
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from muninn.version import __version__

# Configure logging to file since stdout is used for MCP protocol
GLOBAL_MEMORY_DIR = Path(__file__).parent.resolve()
LOG_FILE = GLOBAL_MEMORY_DIR / "mcp_wrapper.log"

import functools

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_SESSION_STATE = {
    "negotiated": False,
    "initialized": False,
    "protocol_version": SUPPORTED_PROTOCOL_VERSIONS[0],
}

def get_git_info() -> Dict[str, str]:
    """Get project name and git branch in real-time."""
    info = {"project": "global", "branch": "none"}
    cwd = os.getcwd()
    try:
        # Find git root to avoid issues in subdirectories
        root_res = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=1)
        if root_res.returncode == 0:
            git_root = root_res.stdout.strip()
            info["project"] = os.path.basename(git_root)
            
            # Get branch
            branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=1)
            if branch_res.returncode == 0:
                info["branch"] = branch_res.stdout.strip()
        else:
            # Fallback to CWD basename if not a git repo
            info["project"] = os.path.basename(cwd)
    except Exception:
        info["project"] = os.path.basename(cwd)
    return info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(LOG_FILE),
    filemode='a'
)
logger = logging.getLogger("Muninn")

SERVER_SCRIPT = GLOBAL_MEMORY_DIR / "server.py"

# Critical: Ensure we use the neutral data directory env var if needed, 
# though server.py now hardcodes ~/.muninn/data. 
# We maintain the relative path for the script execution.
SERVER_URL = os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069")
HEALTH_URL = f"{SERVER_URL}/health"

def make_request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Make HTTP request with exponential backoff retry and server auto-restart."""
    max_retries = 3
    base_delay = 0.5
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            logger.warning(f"Connection failed (attempt {attempt+1}/{max_retries}): {e}")
            
            # If server might be down, ensure it's running
            if attempt < max_retries - 1:
                ensure_server_running()
                time.sleep(base_delay * (2 ** attempt))
            
    if last_error:
        raise last_error
    raise requests.RequestException("Unknown connection error after retries")

def is_server_running() -> bool:
    """Check if the Muninn server is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_and_start_ollama():
    """Check if Ollama is running, and start it if not."""
    from muninn.platform import spawn_detached_process, find_ollama_executable

    try:
        # Quick check if Ollama is responsive
        requests.get("http://localhost:11434", timeout=0.5)
        logger.info("Ollama is already running (responsive).")
        return True
    except requests.RequestException:
        logger.warning("Ollama is not responding. Attempting to start...")

    try:
        ollama_path = find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found on this system.")
            return False

        spawn_detached_process([ollama_path, "serve"])

        # Wait for Ollama to become responsive
        for _ in range(20):  # 10 seconds wait
            try:
                requests.get("http://localhost:11434", timeout=0.5)
                logger.info("Ollama started successfully.")
                return True
            except requests.RequestException:
                time.sleep(0.5)

        logger.error("Timed out waiting for Ollama to start.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch Ollama: {e}")
        return False

def start_server():
    """Start the Muninn server in a detached process (cross-platform)."""
    from muninn.platform import spawn_detached_process, find_python_executable

    logger.info("Starting Muninn server...")

    python_executable = find_python_executable()

    try:
        spawn_detached_process(
            [python_executable, str(SERVER_SCRIPT)],
            cwd=str(GLOBAL_MEMORY_DIR),
        )
        # Don't block waiting for full startup here.
        # The first tool call will trigger a retry loop if needed.
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def ensure_server_running():
    """Ensure server is up, starting it if necessary."""
    if not is_server_running():
        start_server()

# --- MCP Protocol Implementation ---

def send_json_rpc(message: Dict[str, Any]):
    """Send JSON-RPC message to stdout."""
    print(json.dumps(message))
    sys.stdout.flush()


def _send_json_rpc_error(msg_id: Any, code: int, message: str):
    """Send a JSON-RPC error response."""
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": code,
            "message": message,
        },
    })

def _negotiate_protocol_version(requested: Optional[str]) -> Optional[str]:
    """Return requested protocol version only when explicitly supported."""
    if requested and requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    if requested is None:
        return SUPPORTED_PROTOCOL_VERSIONS[0]
    return None


def handle_initialize(msg_id: Any, params: Optional[Dict[str, Any]] = None):
    """Handle the initialize request from the client."""
    requested = None
    if params:
        requested = params.get("protocolVersion")
    negotiated = _negotiate_protocol_version(requested)
    if negotiated is None:
        send_json_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32602,
                "message": (
                    f"Unsupported protocol version: {requested}. "
                    f"Supported versions: {', '.join(SUPPORTED_PROTOCOL_VERSIONS)}"
                ),
            },
        })
        return
    _SESSION_STATE["negotiated"] = True
    _SESSION_STATE["initialized"] = False
    _SESSION_STATE["protocol_version"] = negotiated

    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "muninn-mcp",
                "version": __version__
            },
            "instructions": (
                "Muninn MCP server. Set project goals, store/search memories, and use handoff tools "
                "for cross-assistant continuity."
            ),
        }
    })


def _dispatch_rpc_message(msg: Dict[str, Any]) -> None:
    """
    Handle a single parsed JSON-RPC message.

    Conformance notes:
    - Unknown request methods (with id) return -32601.
    - Unknown notifications (no id) are ignored.
    - notifications/initialized is only accepted after successful initialize.
    """
    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if not isinstance(method, str):
        if msg_id is not None:
            _send_json_rpc_error(msg_id, -32600, "Invalid Request: missing method")
        return

    if method == "initialize":
        if params is None:
            params = {}
        if not isinstance(params, dict):
            _send_json_rpc_error(msg_id, -32602, "Invalid params: initialize params must be an object")
            return
        handle_initialize(msg_id, params)
        return

    if method == "notifications/initialized":
        if _SESSION_STATE["negotiated"]:
            _SESSION_STATE["initialized"] = True
            logger.info("Client initialized connection")
        else:
            logger.warning("Ignored notifications/initialized before successful initialize")
        return

    if method == "ping":
        if msg_id is not None:
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            })
        return

    if method == "tools/list":
        if not _SESSION_STATE["initialized"]:
            if msg_id is not None:
                _send_json_rpc_error(
                    msg_id,
                    -32600,
                    "Server not initialized. Send initialize then notifications/initialized.",
                )
            return
        if msg_id is None:
            logger.debug("Ignoring tools/list notification without id")
            return
        handle_list_tools(msg_id)
        return

    if method == "tools/call":
        if not _SESSION_STATE["initialized"]:
            if msg_id is not None:
                _send_json_rpc_error(
                    msg_id,
                    -32600,
                    "Server not initialized. Send initialize then notifications/initialized.",
                )
            return
        if msg_id is None:
            logger.debug("Ignoring tools/call notification without id")
            return
        if params is None:
            params = {}
        if not isinstance(params, dict):
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call params must be an object")
            return
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call requires non-empty string name")
            return
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call arguments must be an object")
            return
        handle_call_tool(msg_id, {"name": name, "arguments": arguments})
        return

    if msg_id is not None:
        _send_json_rpc_error(msg_id, -32601, f"Method not found: {method}")
    else:
        logger.debug(f"Ignoring unknown notification method: {method}")

def handle_list_tools(msg_id: Any):
    """Return list of available tools."""
    tools = [
        {
            "name": "add_memory",
            "description": "Add a new memory to the global knowledge base. Use this to store facts, preferences, or important information that should be remembered across sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata tags (e.g., {'project': 'phoenix', 'category': 'api'})."
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search for memories relevant to a query. Uses hybrid search with optional reranking for precision.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 5)",
                        "default": 5
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable SOTA reranking for precision (default true)",
                        "default": True
                    },
                    "explain": {
                        "type": "boolean",
                        "description": "Include per-result recall trace explaining retrieval signals (v3.1.0)",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_all_memories",
            "description": "Retrieve all stored memories, optionally filtered by user/agent.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 100)",
                        "default": 100
                    }
                }
            }
        },
        {
            "name": "update_memory",
            "description": "Update an existing memory by ID. Use to correct or enhance stored information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to update."
                    },
                    "content": {
                        "type": "string",
                        "description": "New content to replace the existing memory."
                    }
                },
                "required": ["memory_id", "content"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a specific memory by ID. Cannot be undone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to delete."
                    }
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "delete_all_memories",
            "description": "Delete ALL memories for a user. DANGEROUS - use with extreme caution. Cannot be undone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm deletion of all memories.",
                        "default": False
                    }
                },
                "required": ["confirm"]
            }
        },
        {
            "name": "set_project_goal",
            "description": "Set or update project north-star goal/constraints for drift checks and goal-aware retrieval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal_statement": {
                        "type": "string",
                        "description": "Canonical objective statement for this project."
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional constraints/non-goals that must be preserved."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace scope (default: global).",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override; defaults to current git repo."
                    }
                },
                "required": ["goal_statement"]
            }
        },
        {
            "name": "get_project_goal",
            "description": "Fetch the active project goal for current repository/namespace scope.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    }
                }
            }
        },
        {
            "name": "export_handoff",
            "description": "Export deterministic cross-assistant handoff bundle for this project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Number of top memories to include."
                    }
                }
            }
        },
        {
            "name": "import_handoff",
            "description": "Import a handoff bundle idempotently using event ledger checks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bundle": {
                        "type": "object",
                        "description": "Handoff bundle produced by export_handoff."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "source": {
                        "type": "string",
                        "default": "mcp_import"
                    }
                },
                "required": ["bundle"]
            }
        },
        {
            "name": "record_retrieval_feedback",
            "description": "Record retrieval outcome feedback to improve adaptive signal weighting over time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that produced the retrieved memory."
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Identifier of the memory being rated."
                    },
                    "outcome": {
                        "type": "number",
                        "description": "Feedback score in [0,1] where 1 means helpful/accepted."
                    },
                    "rank": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based displayed rank position for counterfactual calibration."
                    },
                    "sampling_prob": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "description": "Optional probability that this result was shown/clicked under the logging policy."
                    },
                    "signals": {
                        "type": "object",
                        "description": "Optional signal contribution map, e.g. {\"vector\":0.8,\"bm25\":0.1}."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "source": {
                        "type": "string",
                        "default": "mcp_feedback"
                    }
                },
                "required": ["query", "memory_id", "outcome"]
            }
        },
        {
            "name": "ingest_sources",
            "description": "Ingest local files/directories into memory with fail-open parsing and per-source provenance metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of local file or directory paths to ingest."
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Recursively traverse directory sources."
                    },
                    "chronological_order": {
                        "type": "string",
                        "enum": ["none", "oldest_first", "newest_first"],
                        "default": "none",
                        "description": "Process source files in deterministic path order or by file modification time."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata merged into each ingested chunk."
                    },
                    "max_file_size_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional maximum source file size."
                    },
                    "chunk_size_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional chunk size override."
                    },
                    "chunk_overlap_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional chunk overlap override."
                    },
                    "min_chunk_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional minimum chunk length."
                    }
                },
                "required": ["sources"]
            }
        },
        {
            "name": "discover_legacy_sources",
            "description": "Discover local legacy assistant/MCP memory sources (Codex, Claude Code, Serena, Cursor/VS Code stores, etc.) available for import.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "roots": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional additional root directories to scan."
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional provider allowlist, e.g. ['codex_cli','serena_memory']."
                    },
                    "include_unsupported": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include files not currently supported by ingestion parsers."
                    },
                    "max_results_per_provider": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum files returned per provider."
                    }
                }
            }
        },
        {
            "name": "ingest_legacy_sources",
            "description": "Ingest user-selected legacy sources discovered from assistant logs and MCP memory programs with contextual metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selected_source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source IDs selected from discover_legacy_sources."
                    },
                    "selected_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit local paths to include."
                    },
                    "roots": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "include_unsupported": {
                        "type": "boolean",
                        "default": False
                    },
                    "max_results_per_provider": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False
                    },
                    "chronological_order": {
                        "type": "string",
                        "enum": ["none", "oldest_first", "newest_first"],
                        "default": "none"
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata merged into each ingested chunk."
                    },
                    "max_file_size_bytes": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "chunk_size_chars": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "chunk_overlap_chars": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "min_chunk_chars": {
                        "type": "integer",
                        "minimum": 1
                    }
                }
            }
        }
    ]
    
    read_only_tools = {"search_memory", "get_all_memories", "get_project_goal", "export_handoff", "discover_legacy_sources"}
    for tool in tools:
        schema = tool.get("inputSchema", {})
        if isinstance(schema, dict) and "$schema" not in schema:
            schema["$schema"] = JSON_SCHEMA_2020_12
        tool["annotations"] = {"readOnlyHint": tool.get("name") in read_only_tools}

    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "tools": tools
        }
    })

def handle_call_tool(msg_id: Any, params: Dict[str, Any]):
    """Handle tool execution requests."""
    name = params.get("name")
    arguments = params.get("arguments", {})
    
    ensure_server_running()
    
    try:
        if name == "add_memory":
            # SOTA: Inject current working directory as 'project' metadata
            # This allows the memory system to automatically anchor memories to the workspace
            metadata = arguments.get("metadata", {})
            git_info = get_git_info()
            if "project" not in metadata:
                metadata["project"] = git_info["project"]
            if "branch" not in metadata:
                metadata["branch"] = git_info["branch"]
            
            payload = {
                "content": arguments.get("content"),
                "metadata": metadata,
                "user_id": "global_user"
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/add", json=payload, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
            
        elif name == "search_memory":
            git_info = get_git_info()
            filters = arguments.get("filters", {})
            if "project" not in filters:
                filters["project"] = git_info["project"]

            explain = arguments.get("explain", False)
            payload = {
                "query": arguments.get("query"),
                "limit": arguments.get("limit", 5),
                "rerank": arguments.get("rerank", True),
                "user_id": "global_user",
                "filters": filters,
                "explain": explain,
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/search", json=payload, timeout=10)
            result = resp.json()

            formatted_results = []
            if result.get("success") and result.get("data"):
                for item in result["data"]:
                    content = str(item.get('content', item.get('memory', 'Unknown content')))
                    score = item.get('score', '')
                    mem_type = item.get('memory_type', '')
                    prefix = f"[{mem_type}:{score:.2f}] " if score and mem_type else ""
                    line = f"- {prefix}{content}"

                    # Append recall trace explanation if present (v3.1.0)
                    trace = item.get("trace")
                    if trace and explain:
                        explanation = trace.get("explanation", "")
                        dominant = trace.get("dominant_signal", "")
                        if explanation:
                            line += f"\n  Why: {explanation}"
                        elif dominant:
                            line += f"\n  Dominant signal: {dominant}"

                    formatted_results.append(line)
                text_response = "\n".join(formatted_results) if formatted_results else "No relevant memories found."
            else:
                text_response = f"Error or no data: {result}"

            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": text_response
                    }]
                }
            })
            
        elif name == "get_all_memories":
            limit = arguments.get("limit", 100)
            resp = make_request_with_retry("GET", f"{SERVER_URL}/get_all", params={"user_id": "global_user", "limit": limit}, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "update_memory":
            payload = {
                "memory_id": arguments.get("memory_id"),
                "data": arguments.get("content")
            }
            resp = make_request_with_retry("PUT", f"{SERVER_URL}/update", json=payload, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "delete_memory":
            memory_id = arguments.get("memory_id")
            resp = make_request_with_retry("DELETE", f"{SERVER_URL}/delete/{memory_id}", timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "delete_all_memories":
            confirm = arguments.get("confirm", False)
            if not confirm:
                send_json_rpc({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": "Must set 'confirm: true' to delete all memories"
                    }
                })
                return

            # Muninn v3 uses POST /delete_all with JSON body
            resp = make_request_with_retry("POST", f"{SERVER_URL}/delete_all", json={"user_id": "global_user"}, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "set_project_goal":
            git_info = get_git_info()
            payload = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "goal_statement": arguments.get("goal_statement"),
                "constraints": arguments.get("constraints", []),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/goal/set", json=payload, timeout=15)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "get_project_goal":
            git_info = get_git_info()
            params = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
            }
            resp = make_request_with_retry("GET", f"{SERVER_URL}/goal/get", params=params, timeout=10)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "export_handoff":
            git_info = get_git_info()
            payload = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "limit": arguments.get("limit", 25),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/export", json=payload, timeout=30)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "import_handoff":
            git_info = get_git_info()
            payload = {
                "bundle": arguments.get("bundle"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "source": arguments.get("source", "mcp_import"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/import", json=payload, timeout=30)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "record_retrieval_feedback":
            git_info = get_git_info()
            payload = {
                "query": arguments.get("query"),
                "memory_id": arguments.get("memory_id"),
                "outcome": arguments.get("outcome"),
                "rank": arguments.get("rank"),
                "sampling_prob": arguments.get("sampling_prob"),
                "signals": arguments.get("signals", {}),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "source": arguments.get("source", "mcp_feedback"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/feedback/retrieval", json=payload, timeout=15)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "ingest_sources":
            git_info = get_git_info()
            payload = {
                "sources": arguments.get("sources", []),
                "recursive": arguments.get("recursive", False),
                "chronological_order": arguments.get("chronological_order", "none"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "metadata": arguments.get("metadata", {}),
                "max_file_size_bytes": arguments.get("max_file_size_bytes"),
                "chunk_size_chars": arguments.get("chunk_size_chars"),
                "chunk_overlap_chars": arguments.get("chunk_overlap_chars"),
                "min_chunk_chars": arguments.get("min_chunk_chars"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest", json=payload, timeout=60)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "discover_legacy_sources":
            payload = {
                "roots": arguments.get("roots", []),
                "providers": arguments.get("providers", []),
                "include_unsupported": arguments.get("include_unsupported", False),
                "max_results_per_provider": arguments.get("max_results_per_provider", 100),
            }
            resp = make_request_with_retry(
                "POST",
                f"{SERVER_URL}/ingest/legacy/discover",
                json=payload,
                timeout=60,
            )
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "ingest_legacy_sources":
            git_info = get_git_info()
            payload = {
                "selected_source_ids": arguments.get("selected_source_ids", []),
                "selected_paths": arguments.get("selected_paths", []),
                "roots": arguments.get("roots", []),
                "providers": arguments.get("providers", []),
                "include_unsupported": arguments.get("include_unsupported", False),
                "max_results_per_provider": arguments.get("max_results_per_provider", 100),
                "recursive": arguments.get("recursive", False),
                "chronological_order": arguments.get("chronological_order", "none"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "metadata": arguments.get("metadata", {}),
                "max_file_size_bytes": arguments.get("max_file_size_bytes"),
                "chunk_size_chars": arguments.get("chunk_size_chars"),
                "chunk_overlap_chars": arguments.get("chunk_overlap_chars"),
                "min_chunk_chars": arguments.get("min_chunk_chars"),
            }
            resp = make_request_with_retry(
                "POST",
                f"{SERVER_URL}/ingest/legacy/import",
                json=payload,
                timeout=120,
            )
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        send_json_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        })

def main():
    logger.info("Muninn MCP Wrapper started")
    
    # Standard input loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            msg = json.loads(line)
            if not isinstance(msg, dict):
                continue
            _dispatch_rpc_message(msg)
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Loop error: {e}")

if __name__ == "__main__":
    main()
