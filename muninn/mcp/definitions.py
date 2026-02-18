from typing import List, Dict, Any

JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")

TOOLS_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "add_memory",
        "description": "Add a new memory to the global knowledge base. Use this to store facts, preferences, or important information that should be remembered across sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember."},
                "metadata": {"type": "object", "description": "Optional metadata tags (e.g., {'project': 'phoenix', 'category': 'api'})."}
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
                "query": {"type": "string", "description": "The search query."},
                "limit": {"type": "integer", "default": 5, "description": "Max number of results (default 5)"},
                "rerank": {"type": "boolean", "default": True, "description": "Enable SOTA reranking for precision (default true)"},
                "explain": {"type": "boolean", "default": False, "description": "Include per-result recall trace explaining retrieval signals (v3.1.0)"}
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
                "limit": {"type": "integer", "default": 100, "description": "Max number of results (default 100)"}
            }
        }
    },
    {
        "name": "update_memory",
        "description": "Update an existing memory by ID. Use to correct or enhance stored information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The unique ID of the memory to update."},
                "content": {"type": "string", "description": "New content to replace the existing memory."}
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
                "memory_id": {"type": "string", "description": "The unique ID of the memory to delete."}
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
                "goal_statement": {"type": "string", "description": "Canonical objective statement for this project."},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Optional constraints/non-goals that must be preserved."},
                "namespace": {"type": "string", "description": "Optional namespace scope (default: global).", "default": "global"},
                "project": {"type": "string", "description": "Optional project override; defaults to current git repo."}
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
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."}
            }
        }
    },
    {
        "name": "set_user_profile",
        "description": "Set or update editable user profile/global context (skills, environments, paths, hardware, preferences).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "description": "User profile patch/object to store. Can include skills, tools, paths, environment, hardware, and preferences."
                },
                "merge": {
                    "type": "boolean",
                    "default": True,
                    "description": "When true, deep-merge patch into existing profile; when false, replace profile."
                },
                "source": {
                    "type": "string",
                    "default": "mcp_tool",
                    "description": "Optional mutation source tag for auditability."
                }
            },
            "required": ["profile"]
        }
    },
    {
        "name": "get_user_profile",
        "description": "Fetch editable user profile/global context for the active user scope.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_model_profiles",
        "description": "Fetch active runtime extraction profile policy for helper/ingestion routing.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_model_profiles",
        "description": "Update runtime extraction profile policy without restarting the server.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_profile": {
                    "type": "string",
                    "enum": list(SUPPORTED_MODEL_PROFILES),
                    "description": "Default extraction profile fallback."
                },
                "runtime_model_profile": {
                    "type": "string",
                    "enum": list(SUPPORTED_MODEL_PROFILES),
                    "description": "Profile for add/update helper extraction."
                },
                "ingestion_model_profile": {
                    "type": "string",
                    "enum": list(SUPPORTED_MODEL_PROFILES),
                    "description": "Profile for source ingestion extraction."
                },
                "legacy_ingestion_model_profile": {
                    "type": "string",
                    "enum": list(SUPPORTED_MODEL_PROFILES),
                    "description": "Profile for legacy source ingestion extraction."
                },
                "source": {
                    "type": "string",
                    "description": "Optional mutation source tag for audit trail.",
                    "default": "mcp_tool"
                }
            }
        }
    },
    {
        "name": "get_model_profile_events",
        "description": "Fetch recent runtime model profile policy mutation events for auditability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 25,
                    "description": "Maximum number of recent events to return."
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
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."},
                "limit": {"type": "integer", "default": 25, "description": "Number of top memories to include."}
            }
        }
    },
    {
        "name": "import_handoff",
        "description": "Import a handoff bundle idempotently using event ledger checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bundle": {"type": "object", "description": "Handoff bundle produced by export_handoff."},
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."},
                "source": {"type": "string", "default": "mcp_import"}
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
                "query": {"type": "string", "description": "The query that produced the retrieved memory."},
                "memory_id": {"type": "string", "description": "Identifier of the memory being rated."},
                "outcome": {"type": "number", "description": "Feedback score in [0,1] where 1 means helpful/accepted."},
                "rank": {"type": "integer", "minimum": 1, "description": "Optional 1-based displayed rank position for counterfactual calibration."},
                "sampling_prob": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "description": "Optional probability that this result was shown/clicked under the logging policy."},
                "signals": {"type": "object", "description": "Optional signal contribution map, e.g. {\"vector\":0.8,\"bm25\":0.1}."},
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."},
                "source": {"type": "string", "default": "mcp_feedback"}
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
                "sources": {"type": "array", "items": {"type": "string"}, "description": "List of local file or directory paths to ingest."},
                "recursive": {"type": "boolean", "default": False, "description": "Recursively traverse directory sources."},
                "chronological_order": {"type": "string", "enum": ["none", "oldest_first", "newest_first"], "default": "none", "description": "Process source files in deterministic path order or by file modification time."},
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."},
                "metadata": {"type": "object", "description": "Optional metadata merged into each ingested chunk."},
                "max_file_size_bytes": {"type": "integer", "minimum": 1, "description": "Optional maximum source file size."},
                "chunk_size_chars": {"type": "integer", "minimum": 1, "description": "Optional chunk size override."},
                "chunk_overlap_chars": {"type": "integer", "minimum": 0, "description": "Optional chunk overlap override."},
                "min_chunk_chars": {"type": "integer", "minimum": 1, "description": "Optional minimum chunk length."}
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
                "roots": {"type": "array", "items": {"type": "string"}, "description": "Optional additional root directories to scan."},
                "providers": {"type": "array", "items": {"type": "string"}, "description": "Optional provider allowlist, e.g. ['codex_cli','serena_memory']."},
                "include_unsupported": {"type": "boolean", "default": False, "description": "Include files not currently supported by ingestion parsers."},
                "max_results_per_provider": {"type": "integer", "minimum": 1, "description": "Maximum files returned per provider."}
            }
        }
    },
    {
        "name": "ingest_legacy_sources",
        "description": "Ingest user-selected legacy sources discovered from assistant logs and MCP memory programs with contextual metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selected_source_ids": {"type": "array", "items": {"type": "string"}, "description": "Source IDs selected from discover_legacy_sources."},
                "selected_paths": {"type": "array", "items": {"type": "string"}, "description": "Optional explicit local paths to include."},
                "roots": {"type": "array", "items": {"type": "string"}},
                "providers": {"type": "array", "items": {"type": "string"}},
                "include_unsupported": {"type": "boolean", "default": False},
                "max_results_per_provider": {"type": "integer", "minimum": 1},
                "recursive": {"type": "boolean", "default": False},
                "chronological_order": {"type": "string", "enum": ["none", "oldest_first", "newest_first"], "default": "none"},
                "namespace": {"type": "string", "default": "global"},
                "project": {"type": "string", "description": "Optional project override."},
                "metadata": {"type": "object", "description": "Optional metadata merged into each ingested chunk."},
                "max_file_size_bytes": {"type": "integer", "minimum": 1},
                "chunk_size_chars": {"type": "integer", "minimum": 1},
                "chunk_overlap_chars": {"type": "integer", "minimum": 0},
                "min_chunk_chars": {"type": "integer", "minimum": 1}
            }
        }
    },
    {
        "name": "get_temporal_knowledge",
        "description": "Query the Temporal Knowledge Graph for facts valid at a specific time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "description": "Epoch timestamp to query validity at (defaults to now)."},
                "limit": {"type": "integer", "default": 50, "description": "Max facts to return."}
            }
        }
    },
    {
        "name": "create_federation_manifest",
        "description": "Generate a federation manifest for cross-agent synchronization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "default": "global", "description": "Project scope for the manifest."}
            }
        }
    },
    {
        "name": "calculate_federation_delta",
        "description": "Calculate memory differences between local and remote manifests.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "local": {"type": "object", "description": "Local manifest."},
                "remote": {"type": "object", "description": "Remote manifest."}
            },
            "required": ["local", "remote"]
        }
    },
    {
        "name": "create_federation_bundle",
        "description": "Create a portable sync bundle for specific memory IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "List of memory IDs to bundle."}
            },
            "required": ["memory_ids"]
        }
    },
    {
        "name": "apply_federation_bundle",
        "description": "Apply a sync bundle to the local memory store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bundle": {"type": "object", "description": "The sync bundle to apply."}
            },
            "required": ["bundle"]
        }
    }
]

# Mapping for tool categorized hints
READ_ONLY_TOOLS = {
    "search_memory", "get_all_memories", "get_project_goal", 
    "get_user_profile", "get_model_profiles", "get_model_profile_events",
    "export_handoff", "discover_legacy_sources",
    "get_temporal_knowledge", "create_federation_manifest", "calculate_federation_delta", "create_federation_bundle"
}

DESTRUCTIVE_TOOLS = {"delete_memory", "delete_all_memories"}

IDEMPOTENT_TOOLS = READ_ONLY_TOOLS.union({
    "update_memory", "delete_memory", "delete_all_memories",
    "set_project_goal", "set_user_profile", "set_model_profiles",
    "import_handoff", "apply_federation_bundle"
})