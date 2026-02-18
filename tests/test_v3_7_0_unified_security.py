import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch
from muninn.core.security import verify_token, initialize_security, get_token

def test_unified_verify_token_logic():
    """Verify core verify_token logic with environment setup."""
    # Patch environment and reset security module state
    with patch.dict(os.environ, {"MUNINN_AUTH_TOKEN": "test_secret_123"}):
        import muninn.core.security
        muninn.core.security._GLOBAL_AUTH_TOKEN = None
        
        token = get_token()
        assert token == "test_secret_123"
        assert verify_token("test_secret_123") is True
        assert verify_token("wrong_token") is False
        assert verify_token(None) is False

def test_mcp_request_token_injection():
    """Verify that MCP requests automatically inject the Authorization header."""
    # Reset security module state
    import muninn.core.security
    muninn.core.security._GLOBAL_AUTH_TOKEN = None
    
    with patch.dict(os.environ, {"MUNINN_AUTH_TOKEN": "proxy_token_456"}):
        from muninn.mcp.requests import make_request_with_retry
        
        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            # Make a request
            make_request_with_retry("GET", "http://localhost:8000/health")
            
            # Verify headers
            args, kwargs = mock_request.call_args
            assert "headers" in kwargs
            assert kwargs["headers"]["Authorization"] == "Bearer proxy_token_456"

@pytest.mark.asyncio
async def test_server_dependency_parity():
    """Verify that server.py uses the centralized core validation logic."""
    from fastapi.security import HTTPAuthorizationCredentials
    import server
    
    mock_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
    
    # Patch the function where it is USED (in server.py)
    with patch("server.core_verify_token") as mock_core_verify:
        mock_core_verify.return_value = True
        
        # Should succeed
        await server.verify_token(mock_creds)
        mock_core_verify.assert_called_with("valid_token")

    with patch("server.core_verify_token") as mock_core_verify:
        mock_core_verify.return_value = False
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await server.verify_token(mock_creds)
        assert excinfo.value.status_code == 401

def test_security_initialization_generation():
    """Verify that a token is generated if none is configured."""
    import muninn.core.security
    muninn.core.security._GLOBAL_AUTH_TOKEN = None
    
    with patch.dict(os.environ, {}, clear=True):
        token = initialize_security(None)
        assert token is not None
        assert len(token) > 20
        assert verify_token(token) is True
