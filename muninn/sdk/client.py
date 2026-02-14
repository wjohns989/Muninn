"""
Muninn Python SDK clients (sync + async).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
import requests

from muninn.sdk.errors import MuninnAPIError, MuninnConnectionError

DEFAULT_BASE_URL = os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069")


def _normalize_base_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid Muninn base URL: {base_url!r}")
    return value


def _coerce_error_detail(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            try:
                return json.dumps(detail, sort_keys=True)
            except Exception:
                return str(detail)
    if isinstance(payload, str):
        return payload
    return fallback


class _BaseMuninnClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0):
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _validate_add_inputs(
        self,
        content: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> None:
        if not content and not messages:
            raise ValueError("Either 'content' or 'messages' must be provided.")

    def _unwrap_api_payload(self, response: Any, *, path: str, status_code: int) -> Any:
        if status_code >= 400:
            detail = _coerce_error_detail(response, f"HTTP {status_code} error")
            raise MuninnAPIError(detail, status_code=status_code, path=path, payload=response)

        if path == "/health":
            if isinstance(response, dict):
                return response
            raise MuninnAPIError(
                "Invalid health response payload",
                status_code=status_code,
                path=path,
                payload=response,
            )

        if isinstance(response, dict) and "success" in response:
            if response.get("success") is False:
                detail = _coerce_error_detail(response, "Muninn API returned success=false")
                raise MuninnAPIError(detail, status_code=status_code, path=path, payload=response)
            return response.get("data")

        return response

    def _search_payload(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: Optional[str],
        limit: int,
        rerank: bool,
        filters: Optional[Dict[str, Any]],
        namespaces: Optional[List[str]],
        explain: bool,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "user_id": user_id,
            "agent_id": agent_id,
            "limit": limit,
            "rerank": rerank,
            "filters": filters,
            "namespaces": namespaces,
            "explain": explain,
        }


class MuninnClient(_BaseMuninnClient):
    """
    Synchronous SDK for Muninn REST APIs.

    Usage:
        from muninn.sdk import MuninnClient
        client = MuninnClient()
        result = client.search("roadmap status")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout)
        self._owns_session = session is None
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "MuninnClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = self._url(path)
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise MuninnConnectionError(f"Failed to connect to Muninn server at {self.base_url}: {exc}") from exc

        payload: Any
        if response.content:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
        else:
            payload = {}

        return self._unwrap_api_payload(payload, path=path, status_code=response.status_code)

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def add(
        self,
        *,
        content: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "global",
        infer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._validate_add_inputs(content=content, messages=messages)
        return self._request(
            "POST",
            "/add",
            json_body={
                "content": content,
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": metadata,
                "namespace": namespace,
                "infer": infer,
            },
        )

    def search(
        self,
        query: str,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = self._search_payload(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            rerank=rerank,
            filters=filters,
            namespaces=namespaces,
            explain=explain,
        )
        return self._request("POST", "/search", json_body=payload)

    def set_project_goal(
        self,
        *,
        goal_statement: str,
        project: str,
        constraints: Optional[List[str]] = None,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/goal/set",
            json_body={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "goal_statement": goal_statement,
                "constraints": constraints or [],
            },
        )

    def get_project_goal(
        self,
        *,
        project: str,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Optional[Dict[str, Any]]:
        return self._request(
            "GET",
            "/goal/get",
            params={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
            },
        )

    def export_handoff(
        self,
        *,
        project: str,
        limit: int = 25,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/handoff/export",
            json_body={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "limit": limit,
            },
        )

    def import_handoff(
        self,
        *,
        bundle: Dict[str, Any],
        project: str,
        source: str = "handoff_import",
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/handoff/import",
            json_body={
                "bundle": bundle,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "source": source,
            },
        )

    def record_retrieval_feedback(
        self,
        *,
        query: str,
        memory_id: str,
        outcome: float,
        project: str = "global",
        rank: Optional[int] = None,
        sampling_prob: Optional[float] = None,
        user_id: str = "global_user",
        namespace: str = "global",
        signals: Optional[Dict[str, float]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/feedback/retrieval",
            json_body={
                "query": query,
                "memory_id": memory_id,
                "outcome": outcome,
                "rank": rank,
                "sampling_prob": sampling_prob,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "signals": signals or {},
                "source": source,
            },
        )

    def ingest_sources(
        self,
        *,
        sources: List[str],
        project: str = "global",
        user_id: str = "global_user",
        namespace: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not sources:
            raise ValueError("sources must be a non-empty list")
        return self._request(
            "POST",
            "/ingest",
            json_body={
                "sources": sources,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "metadata": metadata or {},
                "recursive": recursive,
                "max_file_size_bytes": max_file_size_bytes,
                "chunk_size_chars": chunk_size_chars,
                "chunk_overlap_chars": chunk_overlap_chars,
                "min_chunk_chars": min_chunk_chars,
            },
        )

    def discover_legacy_sources(
        self,
        *,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/ingest/legacy/discover",
            json_body={
                "roots": roots or [],
                "providers": providers or [],
                "include_unsupported": include_unsupported,
                "max_results_per_provider": max_results_per_provider,
            },
        )

    def ingest_legacy_sources(
        self,
        *,
        selected_source_ids: Optional[List[str]] = None,
        selected_paths: Optional[List[str]] = None,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
        project: str = "global",
        user_id: str = "global_user",
        namespace: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not (selected_source_ids or selected_paths):
            raise ValueError("Provide selected_source_ids and/or selected_paths")
        return self._request(
            "POST",
            "/ingest/legacy/import",
            json_body={
                "selected_source_ids": selected_source_ids or [],
                "selected_paths": selected_paths or [],
                "roots": roots or [],
                "providers": providers or [],
                "include_unsupported": include_unsupported,
                "max_results_per_provider": max_results_per_provider,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "metadata": metadata or {},
                "recursive": recursive,
                "max_file_size_bytes": max_file_size_bytes,
                "chunk_size_chars": chunk_size_chars,
                "chunk_overlap_chars": chunk_overlap_chars,
                "min_chunk_chars": min_chunk_chars,
            },
        )

    def get_all(
        self,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self._request(
            "GET",
            "/get_all",
            params={
                "user_id": user_id,
                "agent_id": agent_id,
                "limit": limit,
            },
        )

    def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        return self._request("PUT", "/update", json_body={"memory_id": memory_id, "data": data})

    def delete(self, memory_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/delete/{memory_id}")

    def delete_all(
        self,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/delete_all",
            json_body={"user_id": user_id, "agent_id": agent_id},
        )

    def delete_batch(self, memory_ids: Iterable[str]) -> List[Dict[str, Any]]:
        return self._request("POST", "/delete_batch", json_body={"memory_ids": list(memory_ids)})

    def get_graph(self, *, user_id: str = "global_user") -> Dict[str, Any]:
        return self._request("GET", "/graph", params={"user_id": user_id})

    def handover(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        memory_id: str,
        reason: str = "Handover for context alignment",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/handover",
            json_body={
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "memory_id": memory_id,
                "reason": reason,
            },
        )

    def federated_search(
        self,
        query: str,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = self._search_payload(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            rerank=rerank,
            filters=filters,
            namespaces=namespaces,
            explain=explain,
        )
        return self._request("POST", "/federated/search", json_body=payload)

    def run_consolidation(self) -> Dict[str, Any]:
        return self._request("POST", "/consolidation/run")

    def consolidation_status(self) -> Dict[str, Any]:
        return self._request("GET", "/consolidation/status")


class AsyncMuninnClient(_BaseMuninnClient):
    """
    Async SDK for Muninn REST APIs.

    Usage:
        from muninn.sdk import AsyncMuninnClient
        async with AsyncMuninnClient() as client:
            result = await client.search("roadmap status")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(headers={"Accept": "application/json"})

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncMuninnClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = self._url(path)
        try:
            response = await self._client.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise MuninnConnectionError(f"Failed to connect to Muninn server at {self.base_url}: {exc}") from exc

        if response.content:
            try:
                payload: Any = response.json()
            except ValueError:
                payload = response.text
        else:
            payload = {}
        return self._unwrap_api_payload(payload, path=path, status_code=response.status_code)

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def add(
        self,
        *,
        content: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "global",
        infer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._validate_add_inputs(content=content, messages=messages)
        return await self._request(
            "POST",
            "/add",
            json_body={
                "content": content,
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": metadata,
                "namespace": namespace,
                "infer": infer,
            },
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = self._search_payload(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            rerank=rerank,
            filters=filters,
            namespaces=namespaces,
            explain=explain,
        )
        return await self._request("POST", "/search", json_body=payload)

    async def set_project_goal(
        self,
        *,
        goal_statement: str,
        project: str,
        constraints: Optional[List[str]] = None,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/goal/set",
            json_body={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "goal_statement": goal_statement,
                "constraints": constraints or [],
            },
        )

    async def get_project_goal(
        self,
        *,
        project: str,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Optional[Dict[str, Any]]:
        return await self._request(
            "GET",
            "/goal/get",
            params={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
            },
        )

    async def export_handoff(
        self,
        *,
        project: str,
        limit: int = 25,
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/handoff/export",
            json_body={
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "limit": limit,
            },
        )

    async def import_handoff(
        self,
        *,
        bundle: Dict[str, Any],
        project: str,
        source: str = "handoff_import",
        user_id: str = "global_user",
        namespace: str = "global",
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/handoff/import",
            json_body={
                "bundle": bundle,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "source": source,
            },
        )

    async def record_retrieval_feedback(
        self,
        *,
        query: str,
        memory_id: str,
        outcome: float,
        project: str = "global",
        rank: Optional[int] = None,
        sampling_prob: Optional[float] = None,
        user_id: str = "global_user",
        namespace: str = "global",
        signals: Optional[Dict[str, float]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/feedback/retrieval",
            json_body={
                "query": query,
                "memory_id": memory_id,
                "outcome": outcome,
                "rank": rank,
                "sampling_prob": sampling_prob,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "signals": signals or {},
                "source": source,
            },
        )

    async def ingest_sources(
        self,
        *,
        sources: List[str],
        project: str = "global",
        user_id: str = "global_user",
        namespace: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not sources:
            raise ValueError("sources must be a non-empty list")
        return await self._request(
            "POST",
            "/ingest",
            json_body={
                "sources": sources,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "metadata": metadata or {},
                "recursive": recursive,
                "max_file_size_bytes": max_file_size_bytes,
                "chunk_size_chars": chunk_size_chars,
                "chunk_overlap_chars": chunk_overlap_chars,
                "min_chunk_chars": min_chunk_chars,
            },
        )

    async def discover_legacy_sources(
        self,
        *,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/ingest/legacy/discover",
            json_body={
                "roots": roots or [],
                "providers": providers or [],
                "include_unsupported": include_unsupported,
                "max_results_per_provider": max_results_per_provider,
            },
        )

    async def ingest_legacy_sources(
        self,
        *,
        selected_source_ids: Optional[List[str]] = None,
        selected_paths: Optional[List[str]] = None,
        roots: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        include_unsupported: bool = False,
        max_results_per_provider: int = 100,
        project: str = "global",
        user_id: str = "global_user",
        namespace: str = "global",
        metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        max_file_size_bytes: Optional[int] = None,
        chunk_size_chars: Optional[int] = None,
        chunk_overlap_chars: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not (selected_source_ids or selected_paths):
            raise ValueError("Provide selected_source_ids and/or selected_paths")
        return await self._request(
            "POST",
            "/ingest/legacy/import",
            json_body={
                "selected_source_ids": selected_source_ids or [],
                "selected_paths": selected_paths or [],
                "roots": roots or [],
                "providers": providers or [],
                "include_unsupported": include_unsupported,
                "max_results_per_provider": max_results_per_provider,
                "user_id": user_id,
                "namespace": namespace,
                "project": project,
                "metadata": metadata or {},
                "recursive": recursive,
                "max_file_size_bytes": max_file_size_bytes,
                "chunk_size_chars": chunk_size_chars,
                "chunk_overlap_chars": chunk_overlap_chars,
                "min_chunk_chars": min_chunk_chars,
            },
        )

    async def get_all(
        self,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await self._request(
            "GET",
            "/get_all",
            params={
                "user_id": user_id,
                "agent_id": agent_id,
                "limit": limit,
            },
        )

    async def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        return await self._request("PUT", "/update", json_body={"memory_id": memory_id, "data": data})

    async def delete(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/delete/{memory_id}")

    async def delete_all(
        self,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/delete_all",
            json_body={"user_id": user_id, "agent_id": agent_id},
        )

    async def delete_batch(self, memory_ids: Iterable[str]) -> List[Dict[str, Any]]:
        return await self._request("POST", "/delete_batch", json_body={"memory_ids": list(memory_ids)})

    async def get_graph(self, *, user_id: str = "global_user") -> Dict[str, Any]:
        return await self._request("GET", "/graph", params={"user_id": user_id})

    async def handover(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        memory_id: str,
        reason: str = "Handover for context alignment",
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/handover",
            json_body={
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "memory_id": memory_id,
                "reason": reason,
            },
        )

    async def federated_search(
        self,
        query: str,
        *,
        user_id: str = "global_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        namespaces: Optional[List[str]] = None,
        explain: bool = False,
    ) -> List[Dict[str, Any]]:
        payload = self._search_payload(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            rerank=rerank,
            filters=filters,
            namespaces=namespaces,
            explain=explain,
        )
        return await self._request("POST", "/federated/search", json_body=payload)

    async def run_consolidation(self) -> Dict[str, Any]:
        return await self._request("POST", "/consolidation/run")

    async def consolidation_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/consolidation/status")


class Memory(MuninnClient):
    """Mem0-style sync alias."""


class AsyncMemory(AsyncMuninnClient):
    """Mem0-style async alias."""
