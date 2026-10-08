"""Synthetic graph transport proof; no server import, live store or model calls."""
import asyncio
import ast
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from muninn.core.credential_boundary import (
    CredentialMemoryError, project_credentials,
)
from muninn.store.graph_store import GraphStore

FAKE = 'OnlySyntheticGraph!42'


class Rows:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.next = None

    def has_next(self):
        self.next = next(self.rows, None)
        return self.next is not None

    def get_next(self):
        return self.next


def graph_with_rows(rows):
    store = object.__new__(GraphStore)
    connection = MagicMock()
    connection.execute.side_effect = lambda *_: Rows(rows)
    store._get_conn = MagicMock(return_value=connection)
    return store, connection


def test_graph_entities_project_legacy_values_without_mutation():
    original = [
        [f'PASSWORD={FAKE}', 'configuration', 2, 'personal'],
        ['sk-only-synthetic-graph-key', 'concept', 3, 'global'],
        ['OPENROUTER_API_KEY in .env', 'credential_location', 1, 'project'],
    ]
    before = copy.deepcopy(original)
    store, connection = graph_with_rows(original)
    result = store.get_all_entities(user_id='synthetic-user')
    assert FAKE not in json.dumps(result)
    assert 'sk-only-synthetic-graph-key' not in json.dumps(result)
    assert result[-1]['name'] == original[-1][0]
    assert [row['mention_count'] for row in result] == [2, 3, 1]
    assert original == before
    assert connection.execute.call_args.args[1]['uid'] == 'synthetic-user'


def test_graph_search_projects_summary_and_entity_match():
    rows = [['synthetic-id', f'PASSWORD={FAKE}', 'sk-only-synthetic-match']]
    before = copy.deepcopy(rows)
    store, _ = graph_with_rows(rows)
    result = store.search_memories('configuration', limit=1, user_id='synthetic-user')
    assert result and result[0]['id'] == 'synthetic-id'
    assert result[0]['score'] == 1.0
    assert FAKE not in json.dumps(result)
    assert 'sk-only-synthetic-match' not in json.dumps(result)
    assert rows == before


@pytest.mark.parametrize('operation', [
    lambda graph: graph.add_entity(f'PASSWORD={FAKE}', 'configuration'),
    lambda graph: graph.create_relation('safe', f'PASSWORD={FAKE}', 'safe-other'),
    lambda graph: graph.create_relation('safe', 'relates', f'PASSWORD={FAKE}'),
    lambda graph: graph.add_memory_node('safe-id', 'normal ' * 100 + f'PASSWORD={FAKE}'),
    lambda graph: graph.link_memory_to_entity('safe-id', 'safe-name', role=f'PASSWORD={FAKE}'),
    lambda graph: graph.add_chain_link('a', 'b', reason='normal ' * 100 + f'PASSWORD={FAKE}'),
    lambda graph: graph.add_chain_link('a', 'b', shared_entities=[f'PASSWORD={FAKE}']),
    lambda graph: graph.create_relation('safe', 'relates', 'safe-other', confidence=f'PASSWORD={FAKE}'),
    lambda graph: graph.add_chain_link('a', 'b', confidence=f'PASSWORD={FAKE}'),
    lambda graph: graph.add_chain_link('a', 'b', hours_apart=f'PASSWORD={FAKE}'),
])
def test_graph_writes_reject_before_connection_or_partial_side_effect(operation):
    store, connection = graph_with_rows([])
    with pytest.raises(CredentialMemoryError) as error:
        operation(store)
    assert FAKE not in str(error.value)
    store._get_conn.assert_not_called()
    connection.execute.assert_not_called()


def test_graph_read_error_diagnostics_do_not_release_exception_content(caplog):
    store, connection = graph_with_rows([])
    connection.execute.side_effect = RuntimeError(f'PASSWORD={FAKE}')
    with caplog.at_level(logging.DEBUG, logger='Muninn.Graph'):
        assert store.get_all_entities() == []
        assert store.search_memories('configuration') == []
    assert FAKE not in caplog.text
    assert 'RuntimeError' in caplog.text


def graph_handler(graph):
    source = Path(__file__).resolve().parents[1] / 'server.py'
    handler = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == 'get_graph_endpoint')
    handler.decorator_list = []
    namespace = dict(memory=SimpleNamespace(_graph=graph), logger=MagicMock(),
                     HTTPException=HTTPException, Optional=__import__('typing').Optional,
                     project_credentials=project_credentials, JSONResponse=JSONResponse,
                     NO_STORE={'Cache-Control':'no-store'})
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['get_graph_endpoint'], namespace['logger']


def test_actual_graph_http_handler_forwards_scope_and_projects_alternate_store():
    original = [{'name':f'PASSWORD={FAKE}', 'mention_count':2}]
    graph = MagicMock()
    graph.get_all_entities.return_value = original
    handler, _ = graph_handler(graph)
    response = asyncio.run(handler(user_id='synthetic-user'))
    graph.get_all_entities.assert_called_once_with(user_id='synthetic-user')
    assert response.headers['cache-control'] == 'no-store'
    result = json.loads(response.body)
    assert result['data']['entity_count'] == 1
    assert FAKE not in json.dumps(result)
    assert FAKE in original[0]['name']


def test_actual_graph_http_handler_failure_is_static():
    graph = MagicMock()
    graph.get_all_entities.side_effect = RuntimeError(f'PASSWORD={FAKE}')
    handler, logger = graph_handler(graph)
    with pytest.raises(HTTPException) as error:
        asyncio.run(handler())
    assert error.value.status_code == 500
    assert FAKE not in error.value.detail
    assert FAKE not in str(logger.error.call_args)


@pytest.mark.parametrize('operation', [
    lambda graph: graph.search_memories(f'PASSWORD={FAKE}'),
    lambda graph: graph.find_related_memories([f'PASSWORD={FAKE}']),
    lambda graph: graph.get_entity_centrality(f'PASSWORD={FAKE}'),
    lambda graph: graph.get_all_entities(namespace=f'PASSWORD={FAKE}'),
])
def test_graph_text_query_guards_precede_query_or_diagnostics(operation):
    store, connection = graph_with_rows([])
    with pytest.raises(CredentialMemoryError):
        operation(store)
    store._get_conn.assert_not_called()
    connection.execute.assert_not_called()


def test_normal_graph_metadata_writes_keep_exact_parameters():
    store, connection = graph_with_rows([])
    assert store.add_entity('OPENROUTER_API_KEY in .env', 'credential_location',
                            user_id='synthetic-user', namespace='project')
    params = connection.execute.call_args.args[1]
    assert params['name'] == 'OPENROUTER_API_KEY in .env'
    assert params['uid'] == 'synthetic-user' and params['ns'] == 'project'
    assert store.add_memory_node('synthetic-id', 'OpenRouter key is configured in .env')
    assert connection.execute.call_args.args[1]['summary'] == 'OpenRouter key is configured in .env'


def test_unsafe_legacy_identifier_is_not_released_or_aliased_to_another_memory():
    rows = [['sk-only-synthetic-legacy-id', 'normal summary', 'safe-name'],
            ['safe-id', f'PASSWORD={FAKE}', 'safe-name']]
    before = copy.deepcopy(rows)
    store, _ = graph_with_rows(rows)
    result = store.search_memories('configuration', limit=2)
    assert [record['id'] for record in result] == ['safe-id']
    assert result[0]['score'] == 1.0
    assert 'sk-only-synthetic-legacy-id' not in json.dumps(result)
    assert FAKE not in json.dumps(result)
    assert rows == before


@pytest.mark.parametrize('operation', [
    lambda graph: graph.add_entity('safe-name', 'configuration'),
    lambda graph: graph.create_relation('safe', 'relates', 'safe-other'),
    lambda graph: graph.add_memory_node('safe-id', 'safe summary'),
    lambda graph: graph.link_memory_to_entity('safe-id', 'safe-name'),
    lambda graph: graph.add_chain_link('a', 'b', reason='safe reason'),
    lambda graph: graph.find_related_memories(['safe-name']),
    lambda graph: graph.get_memory_node_degrees_batch(['safe-id']),
    lambda graph: graph.find_chain_related_memories(['safe-id']),
    lambda graph: graph.delete_memory_references('safe-id'),
    lambda graph: graph._initialize(),
])
def test_other_graph_diagnostics_do_not_echo_legacy_driver_data(operation, caplog):
    store, connection = graph_with_rows([])
    store.db_path = Path('synthetic-graph')
    connection.execute.side_effect = RuntimeError(f'PASSWORD={FAKE}')
    with caplog.at_level(logging.DEBUG, logger='Muninn.Graph'):
        operation(store)
    assert FAKE not in caplog.text
    assert 'RuntimeError' in caplog.text
