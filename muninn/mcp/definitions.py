from typing import List, Dict, Any

JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")

TOOLS_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "add_memory",
        "description": "Add a new memory to the knowledge base. Use this to store facts, preferences, or important information that should be remembered across sessions. Use scope='global' for universal rules/preferences that should always be visible; use scope='project' (default) for project-specific information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember."},
                "metadata": {"type": "object", "description": "Optional metadata tags (e.g., {'project': 'phoenix', 'category': 'api'})."},
                "scope": {
                    "type": "string",
                    "enum": ["project", "global"],
                    "default": "project",
                    "description": "Isolation scope. 'project' = visible only within this project (default). 'global' = always visible across all projects (use for user preferences, universal rules)."
                },
                "media_type": {
                    "type": "string",
                    "enum": ["text", "image", "audio", "video", "sensor"],
                    "default": "text",
                    "description": "Media type of the memory (Phase 20)."
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "set_project_instruction",
        "description": "Convenience tool to create a project-scoped instruction memory. The memory is tagged with the current git project and scope='project', ensuring it NEVER appears when working in a different repository. Use for project-specific coding conventions, constraints, or guidelines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "The project-specific instruction or convention to remember (e.g., 'Always use async/await for I/O operations in this codebase')."},
                "category": {"type": "string", "description": "Optional category tag for the instruction (e.g., 'coding_conventions', 'architecture', 'testing')."}
            },
            "required": ["instruction"]
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
                "explain": {"type": "boolean", "default": False, "description": "Include per-result recall trace explaining retrieval signals (v3.1.0)"},
                "media_type": {
                    "type": "string",
                    "enum": ["text", "image", "audio", "video", "sensor"],
                    "description": "Filter results by media type (Phase 20)."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "hunt_memory",
        "description": "Agentic proactive retrieval. Analyzes the query to identify key entities and relationships, then performs a multi-hop discovery pass across the knowledge graph and vector store to surface hidden context and 'forgotten' wisdom. Best for deep-diving into complex topics or resolving ambiguities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or topic to hunt for."},
                "limit": {"type": "integer", "default": 5, "description": "Max number of results (default 5)"},
                "depth": {"type": "integer", "default": 2, "description": "Search depth for graph traversal (default 2)"},
                "namespaces": {"type": "array", "items": {"type": "string"}, "description": "Optional namespaces to restrict the hunt."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "detect_information_gaps",
        "description": "Analyze a user query and current context to identify missing information (e.g. credentials, paths, IPs) that prevents successful execution. Uses cognitive reasoning (CoALA) to prevent hallucination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's intent or command (e.g. 'Deploy to production')."},
                "context": {"type": "string", "description": "Current conversation state or working memory."},
                "limit": {"type": "integer", "default": 10, "description": "Max memories to retrieve for grounding."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "trigger_distillation",
        "description": "Manually trigger the background knowledge distillation process. Identifies clusters of episodic memories and synthesizes them into clean semantic manuals. Useful for maintenance or after heavy interaction sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": True, "description": "Force run even if interval has not passed."}
            }
        }
    },
    {
        "name": "correct_fact",
        "description": "Correct an erroneous memory based on user feedback. Surgically rewrites the memory record to reflect the new truth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The ID of the memory to correct."},
                "correction": {"type": "string", "description": "The correct information provided by the user."}
            },
            "required": ["memory_id", "correction"]
        }
    },
    {
        "name": "forage_knowledge",
        "description": "Actively forage for related knowledge when an initial search is ambiguous or returns low-confidence results. Uses graph scent-following to discover hidden context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or topic to forage for."},
                "ambiguity_threshold": {"type": "number", "default": 0.7, "description": "Entropy threshold to trigger foraging (0.0 to 1.0)."}
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
        "name": "get_model_profile_alerts",
        "description": "Evaluate profile-policy mutation churn against alert thresholds to detect abnormal policy churn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_seconds": {
                    "type": "number",
                    "minimum": 60,
                    "description": "Optional alert lookback window in seconds."
                },
                "churn_threshold": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional total-event threshold within the window."
                },
                "source_churn_threshold": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional per-source event threshold within the window."
                },
                "distinct_sources_threshold": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional distinct-source threshold within the window."
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
        "name": "get_periodic_ingestion_status",
        "description": "Fetch periodic-ingestion scheduler runtime state, cadence, and last run result.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "run_periodic_ingestion",
        "description": "Trigger one immediate periodic-ingestion run using configured sources and options.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "start_periodic_ingestion",
        "description": "Start the periodic-ingestion scheduler loop without restarting the server.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "stop_periodic_ingestion",
        "description": "Stop the periodic-ingestion scheduler loop without restarting the server.",
        "inputSchema": {
            "type": "object",
            "properties": {}
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
    },
    {
        "name": "mimir_relay",
        "description": (
            "Relay an instruction to another AI agent (Claude Code, Codex CLI, Gemini CLI) "
            "via the IRP/1 Interop Relay Protocol. Muninn selects the best provider based on "
            "memory-aware routing, enforces policy (redaction, hop limits, consent), and returns "
            "a structured RelayResult. "
            "Modes: A=Advisory (observation only), B=Structured (single-provider execution), "
            "C=Reconcile (multi-provider consensus). "
            "Set provider='auto' to let Muninn choose. "
            "The run is persisted to the interop_runs audit table and all policy events are logged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The instruction or prompt to relay to the target agent."
                },
                "mode": {
                    "type": "string",
                    "enum": ["A", "B", "C"],
                    "default": "A",
                    "description": (
                        "IRP/1 relay mode. "
                        "A=Advisory (probe/observe, no write), "
                        "B=Structured (single-provider execution), "
                        "C=Reconcile (multi-provider consensus with conflict resolution)."
                    )
                },
                "provider": {
                    "type": "string",
                    "enum": ["claude_code", "codex_cli", "gemini_cli", "auto"],
                    "default": "auto",
                    "description": (
                        "Target provider. 'auto' triggers memory-aware scoring to select "
                        "the highest-confidence available provider."
                    )
                },
                "user_id": {
                    "type": "string",
                    "default": "global_user",
                    "description": "User namespace for memory scoping, audit logging, and consent checks."
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 131072,
                    "default": 4096,
                    "description": "Maximum output tokens to request from the target provider."
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional free-form context dictionary forwarded verbatim inside the "
                        "IRP envelope. Useful for passing project metadata, conversation history "
                        "summaries, or structured task parameters to the remote agent."
                    )
                },
                "policy": {
                    "type": "object",
                    "description": "Optional IRP/1 policy overrides for this relay call.",
                    "properties": {
                        "max_hops": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3,
                            "description": "Maximum number of provider hops before the relay aborts."
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 300.0,
                            "default": 30.0,
                            "description": "Wall-clock timeout for the entire relay operation."
                        },
                        "allow_redaction": {
                            "type": "boolean",
                            "default": True,
                            "description": "When true, PII/secret patterns are redacted from the instruction before forwarding."
                        },
                        "require_consent": {
                            "type": "boolean",
                            "default": False,
                            "description": "When true, the relay is blocked unless the user has an active consent record."
                        },
                        "network": {
                            "type": "string",
                            "enum": ["allow_all", "local_only", "deny_all"],
                            "default": "allow_all",
                            "description": (
                                "Network policy. 'local_only' restricts relay to providers "
                                "reachable without external network calls. 'deny_all' blocks "
                                "all outbound relay (useful for dry-run/policy testing)."
                            )
                        }
                    }
                }
            },
            "required": ["instruction"]
        }
    }
]

# Mapping for tool categorized hints
READ_ONLY_TOOLS = {
    "search_memory", "hunt_memory", "get_all_memories", "get_project_goal",
    "get_user_profile", "get_model_profiles", "get_model_profile_events", "get_model_profile_alerts",
    "export_handoff", "discover_legacy_sources",
    "get_periodic_ingestion_status",
    "get_temporal_knowledge", "create_federation_manifest", "calculate_federation_delta", "create_federation_bundle",
    "detect_information_gaps", "forage_knowledge"
}

DESTRUCTIVE_TOOLS = {"delete_memory", "delete_all_memories"}

IDEMPOTENT_TOOLS = READ_ONLY_TOOLS.union({
    "update_memory", "delete_memory", "delete_all_memories",
    "set_project_goal", "set_user_profile", "set_model_profiles",
    "import_handoff", "apply_federation_bundle",
    "set_project_instruction",
    "run_periodic_ingestion",
    "start_periodic_ingestion",
    "stop_periodic_ingestion",
    "trigger_distillation",
    "correct_fact",
    "forage_knowledge"
})

# mimir_relay creates a new interop_runs record on every call and may have
# side-effects on the remote agent — it is therefore not read-only or idempotent.
MIMIR_TOOLS = {"mimir_relay"}