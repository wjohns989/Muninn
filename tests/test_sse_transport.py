import asyncio
import json
import httpx
import pytest

@pytest.mark.asyncio
async def test_sse_flow():
    base_url = "http://localhost:8000"

    # Skip if no live server is running
    try:
        async with httpx.AsyncClient() as probe:
            r = await probe.get(f"{base_url}/health", timeout=1.0)
    except Exception:
        pytest.skip("No live Muninn server on port 8000")

    
    # 1. Connect to SSE
    print(f"Connecting to {base_url}/mcp/sse ...")
    async with httpx.AsyncClient() as client:
        # We need to stream the response
        async with client.stream("GET", f"{base_url}/mcp/sse", timeout=10.0) as response:
            post_uri = None
            async for line in response.aiter_lines():
                if not line:
                    continue
                print(f"RAW: {line}")
                if line.startswith("data:"):
                    # Basic SSE parsing
                    data_str = line[5:].strip()
                    if post_uri is None:
                        # The first event is endpoint
                        post_uri = data_str
                        print(f"Found post_uri: {post_uri}")
                        
                        # Now that we have the URI, trigger the POST in the background
                        async def _send_init():
                            await asyncio.sleep(0.5)
                            print(f"Sending POST to {post_uri}")
                            payload = {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "initialize",
                                "params": {
                                    "protocolVersion": "2024-11-05",
                                    "capabilities": {},
                                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                                }
                            }
                            res = await client.post(post_uri, json=payload)
                            print(f"POST response: {res.status_code}")
                        
                        asyncio.create_task(_send_init())
                    else:
                        try:
                            msg = json.loads(data_str)
                            print(f"Message received: {msg.get('method') or msg.get('id')}")
                            if msg.get("id") == 1:
                                print("Test passed! Initialize response received.")
                                return
                        except Exception:
                            pass

if __name__ == "__main__":
    asyncio.run(test_sse_flow())

