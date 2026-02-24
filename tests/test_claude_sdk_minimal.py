
import asyncio
from claude_agent_sdk import query

async def main():
    print("Sending query to Claude via Agent SDK...")
    try:
        # Stateless query, similar to 'claude -p'
        async for message in query(prompt="Respond with only 'SDK_OK'"):
            # The SDK yields Message objects. We want to see the text result.
            if hasattr(message, 'content'):
                print(f"Message content: {message.content}")
            else:
                print(f"Message: {message}")
    except Exception as e:
        print(f"Error during SDK query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
