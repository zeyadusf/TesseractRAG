from enum import Enum

class QueryReWritePrompt(str, Enum):
    BASE = """Rewrite the user query into a single, highly effective search query.

<rules>
- Be specific, unambiguous, and keyword-focused
- Expand implicit meaning if needed
- Keep it concise (max 10 words)
- Return ONLY the rewritten query. NO quotes, NO explanations, NO markdown.
- If "RAG" appears, it means Retrieval-Augmented Generation.
</rules>

User query: {query}
Rewritten query:"""
    
    EXPAND = """Expand the query with relevant synonyms and related technical terms for better document retrieval.

<rules>
- Output ONLY a space-separated list of keywords.
- NO explanations, NO markdown, NO boolean operators (OR/AND).
- DO NOT repeat words from the original query.
- DO NOT add unrelated acronyms (e.g., never add "RAG" unless it's in the query).
- Maximum 10 keywords.
</rules>

Query: {query}
Keywords:"""