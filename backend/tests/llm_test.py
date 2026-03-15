import asyncio
from app.core.generation.llm_client import HuggingFaceClient

async def main():
    client = HuggingFaceClient()
    messages = [{"role": "user", "content": "Say hello in one sentence."}]
    response = await client.generate(messages)
    print(response)

asyncio.run(main())