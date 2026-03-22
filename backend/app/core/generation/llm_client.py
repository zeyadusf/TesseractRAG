import httpx
import asyncio
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceClient:

    def __init__(self):
        # get settings, build the API url, set headers
        self.settings = get_settings()
        self.url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.settings.HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

    async def generate(self, messages: list[dict]) -> str:
        
        # Step 1: build the request body
        payload = {
            "model": self.settings.LLM_MODEL_1,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p" : 0.9,
        }

        # Step 2: retry loop — try up to 3 times
        for attempt in range(3):

            # Step 3: make async POST request using httpx.AsyncClient
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.url,
                    headers=self.headers,
                    json=payload
                )

            # Step 4: if 503, log it, wait 20 seconds, then continue to next attempt
            if response.status_code == 503:
                logger.warning("Model is loading on HuggingFace. Waiting 20 seconds...")
                await asyncio.sleep(20)
                continue

            # Step 5: if not 200, raise an exception with the status code
            if response.status_code != 200:
                raise Exception(
                    f"HuggingFace API error: {response.status_code} - {response.text}"
                )

            # Step 6: parse the response JSON and extract the assistant's reply text
            # HF response shape: {"choices": [{"message": {"content": "..."}}]}
            data = response.json()
            return data["choices"][0]["message"]["content"]

        # Step 7: if all retries exhausted, raise an exception
        raise Exception("HuggingFace model failed after 3 retries")