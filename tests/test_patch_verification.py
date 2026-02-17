import sys
import pytest
from muninn.mcp import handlers, utils

def test_patch_handlers_git_info(monkeypatch):
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "PATCHED"})
    assert handlers.get_git_info()["project"] == "PATCHED"

def test_patch_utils_git_info(monkeypatch):
    monkeypatch.setattr(utils, "get_git_info", lambda: {"project": "PATCHED_UTILS"})
    assert utils.get_git_info()["project"] == "PATCHED_UTILS"
    # Handlers imported from utils before this patch, so it should retain original
    assert handlers.get_git_info()["project"] != "PATCHED_UTILS" 

def test_patch_string_handlers(monkeypatch):
    monkeypatch.setattr("muninn.mcp.handlers.get_git_info", lambda: {"project": "STRING_PATCH"})
    assert handlers.get_git_info()["project"] == "STRING_PATCH"
