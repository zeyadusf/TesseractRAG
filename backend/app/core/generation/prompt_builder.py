from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are TesseractRAG, a precise document-grounded assistant.

You are given retrieved context from the user's documents. Your job is to answer the user's question using ONLY that context.

Rules:
- Answer strictly from the context below. Do not use outside knowledge.
- If the context does not contain enough information, respond exactly: "I don't have enough information in the provided documents to answer this."
- Never fabricate facts, names, numbers, or citations.
- If the context partially answers the question, give what you can and state what is missing.
- Be concise and direct. No filler phrases.

Context:
{context}"""  


class PromptBuilder:

    def build(self, question: str, context: str, history: list[dict]) -> list[dict]:
        
        # Step 1: start messages list with system prompt
        messages = [{"role":"system","content": SYSTEM_PROMPT.format(context=context)}]

        # Step 2: slice last 6 messages from history
        recent_history = history[-6:]

        # Step 3: extend messages with recent history
        messages.extend(recent_history)

        # Step 4: build the final user message combining context and question
        # Format it clearly so the LLM knows what is context and what is the question

        # Step 5: append final user message to messages
        messages.append({"role":'user',"content" : question})

        return messages