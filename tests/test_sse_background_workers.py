import asyncio
import json
import httpx
import sys

async def run_tool_test(base_url: str):
    async with httpx.AsyncClient() as client:
        # 1. Connect and initialize
        async with client.stream("GET", f"{base_url}/mcp/sse", timeout=15.0) as response:
            post_uri = None
            session_established = False
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if post_uri is None:
                        post_uri = data_str
                        # Send init
                        async def _send_init():
                            await asyncio.sleep(0.5)
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
                            await client.post(post_uri, json=payload)
                        asyncio.create_task(_send_init())
                    else:
                        try:
                            msg = json.loads(data_str)
                            if msg.get("id") == 1 and not session_established:
                                session_established = True
                                print("SUCCESS: Session Initialized.")
                                
                                # MCP spec requires this before sending requests
                                async def _send_initialized_notif():
                                    await client.post(post_uri, json={
                                        "jsonrpc": "2.0",
                                        "method": "notifications/initialized"
                                    })
                                asyncio.create_task(_send_initialized_notif())
                                
                                # Trigger a background task tool call
                                async def _send_tool():
                                    await asyncio.sleep(0.5) # Give the initialized notif a moment to process
                                    payload = {
                                        "jsonrpc": "2.0",
                                        "id": 2,
                                        "method": "tools/call",
                                        "params": {
                                            "name": "search_memory",
                                            "arguments": {
                                                "query": "hello"
                                            },
                                            "task": {"ttl": 30000}
                                        }
                                    }
                                    await client.post(post_uri, json=payload)
                                asyncio.create_task(_send_tool())
                            
                            elif msg.get("id") == 2:
                                # We got the immediate accepted response. Wait for the notification.
                                print("Raw Task Accepted Payload:", json.dumps(msg, indent=2))
                                meta = msg.get("result", {}).get("_meta", {})
                                related = meta.get("io.modelcontextprotocol/related-task", {})
                                task_id = related.get("taskId")
                                print(f"ACCEPTED: Tool call accepted as Task ID: {task_id}")
                                
                            elif msg.get("method") == "notifications/tasks/status":
                                task_info = msg.get("params", {}).get("task", {})
                                status = task_info.get("status")
                                print(f"STATUS NOTIFICATION: {status}")
                                if status in ("completed", "failed"):
                                    if status == "failed":
                                        print("FAILED: Worker failed. Wait, why did it fail?")
                                        print(json.dumps(task_info.get("error"), indent=2))
                                        return False
                                    else:
                                        print("SUCCESS: Correctly completed task in background thread!")
                                        return True
                        except Exception as e:
                            print(f"Error parsing: {e}")
    return False

if __name__ == "__main__":
    success = asyncio.run(run_tool_test("http://localhost:8001"))
    sys.exit(0 if success else 1)
