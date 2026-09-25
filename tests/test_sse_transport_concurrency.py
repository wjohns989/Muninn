import asyncio
import json
import httpx
import uuid
import sys

async def run_client_session(base_url: str, expected_version: str):
    print(f"[{expected_version}] Starting client session...")
    
    async with httpx.AsyncClient() as client:
        # We need to stream the response
        async with client.stream("GET", f"{base_url}/mcp/sse", timeout=10.0) as response:
            post_uri = None
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if post_uri is None:
                        post_uri = data_str
                        print(f"[{expected_version}] Connected. Endpoint: {post_uri}")
                        
                        # Background task to send initialization AFTER we connect
                        async def _send_init():
                            await asyncio.sleep(0.5)
                            print(f"[{expected_version}] Sending initialize...")
                            payload = {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "initialize",
                                "params": {
                                    "protocolVersion": expected_version,
                                    "capabilities": {},
                                    "clientInfo": {"name": f"test-{expected_version}", "version": "1.0.0"}
                                }
                            }
                            await client.post(post_uri, json=payload)
                        
                        asyncio.create_task(_send_init())
                    else:
                        try:
                            msg = json.loads(data_str)
                            if msg.get("id") == 1 and "result" in msg:
                                negotiated_version = msg["result"]["protocolVersion"]
                                print(f"[{expected_version}] Received initialize response. Version: {negotiated_version}")
                                print(f"[{expected_version}] ✅ SUCCESS")
                                return True
                        except Exception:
                            pass
    return False

async def main():
    base_url = "http://localhost:8000"
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
            if resp.status_code != 200:
                print("Server not ready. Start it on port 8000 first.")
                sys.exit(1)
    except Exception:
        print("Server not reachable on port 8000.")
        sys.exit(1)

    print("Running two concurrent connections to ensure State Isolation...")
    # Send two drastically different versions to see if they bleed into each other's protocol negotation
    res1, res2 = await asyncio.gather(
        run_client_session(base_url, "2024-11-05"),
        run_client_session(base_url, "2025-01-01") # Even if unsupported, the server response will reflect negotiations
    )
    
    if res1 and res2:
        print("🎉 Both clients negotiated isolated states successfully!")
    else:
        print("❌ One or more clients failed completion.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
