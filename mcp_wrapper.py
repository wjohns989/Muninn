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

# Configure logging to file since stdout is used for MCP protocol
GLOBAL_MEMORY_DIR = Path(__file__).parent.resolve()
LOG_FILE = GLOBAL_MEMORY_DIR / "mcp_wrapper.log"

import functools

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
    try:
        # Quick check if Ollama is responsive
        requests.get("http://localhost:11434", timeout=0.5)
        logger.info("Ollama is already running (responsive).")
        return True
    except requests.RequestException:
        logger.warning("Ollama is not responding. Attempting to start...")

    try:
        # Attempt to start Ollama in the background (Windows specific)
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        
        # SOTA: Silent Launch (shell=False prevents cmd.exe wrapper)
        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            close_fds=True,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Wait for Ollama to become responsive
        for _ in range(20): # 10 seconds wait
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
    """Start the Muninn server in a detached process."""
    logger.info("Starting Muninn server...")
    
    # Windows specific flags for detached process
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000
    
    # Use pythonw.exe for windowless execution if available, else python.exe
    python_executable = sys.executable.replace("python.exe", "pythonw.exe") if "python.exe" in sys.executable else sys.executable

    try:
        subprocess.Popen(
            [python_executable, str(SERVER_SCRIPT)],
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            cwd=str(GLOBAL_MEMORY_DIR),
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # SOTA: Don't block waiting for full startup here. 
        # The first tool call will trigger a retry loop if needed.
        # Just give it a split second to spawn.
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

def handle_initialize(msg_id: Any):
    """Handle the initialize request from the client."""
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "muninn-mcp",
                "version": "3.0.0"
            }
        }
    })

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
        }
    ]
    
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

            payload = {
                "query": arguments.get("query"),
                "limit": arguments.get("limit", 5),
                "rerank": arguments.get("rerank", True),
                "user_id": "global_user",
                "filters": filters
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/search", json=payload, timeout=10)
            result = resp.json()

            formatted_results = []
            if result.get("success") and result.get("data"):
                for item in result["data"]:
                    # Muninn native returns: id, content, score, memory_type, metadata
                    content = str(item.get('content', item.get('memory', 'Unknown content')))
                    score = item.get('score', '')
                    mem_type = item.get('memory_type', '')
                    prefix = f"[{mem_type}:{score:.2f}] " if score and mem_type else ""
                    formatted_results.append(f"- {prefix}{content}")
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
            msg_id = msg.get("id")
            method = msg.get("method")
            
            if method == "initialize":
                handle_initialize(msg_id)
            elif method == "tools/list":
                handle_list_tools(msg_id)
            elif method == "tools/call":
                handle_call_tool(msg_id, msg.get("params", {}))
            elif method == "notifications/initialized":
                # Client acknowledging initialization
                logger.info("Client initialized connection")
            elif method == "ping":
                # Respond to ping
                send_json_rpc({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {}
                })
            else:
                # Basic initialization or unknown methods
                logger.debug(f"Ignoring unknown method: {method}")
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Loop error: {e}")

if __name__ == "__main__":
    main()
