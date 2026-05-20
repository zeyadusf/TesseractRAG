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
<conversation_history>
{history}
</conversation_history>
Query: {query}
Keywords:"""

    CONTEXTUAL = """You are a query rewriter for a RAG (Retrieval-Augmented Generation) system.

<task>
Analyze the current message and conversation history, then decide what to do.
</task>

<rules>
CASE 1 — GREETING OR SMALL TALK
If the message is a greeting, farewell, thanks, or social phrase with no information need
(e.g. hi, hello, bye, thanks, thank you, great, awesome, perfect, well done, who are you,
      مرحبا, أهلاً, شكراً, شكرا, تمام, عظيم, ممتاز, وداعاً, أحسنت, برافو, مين انت)
→ Rewrite into a short, natural conversational phrase that captures the intent.
   Examples: "hi" → "hello greeting", "شكراً" → "thank you appreciation", "bye" → "farewell goodbye"

CASE 2 — FOLLOW-UP or VAGUE
If the message refers back to the previous answer or is too vague to retrieve alone
(e.g. "tell me more", "explain further", "continue", "what do you think",
      "قولي أكتر", "وضّح أكتر", "أكمل", "ما رأيك", "give an example")
→ Rewrite into a standalone search query using the last assistant answer as context.
→ Be specific, keyword-focused, max 12 words.

CASE 3 — CLEAR QUESTION
If the message is already a clear, standalone question
→ Return it exactly as-is, word for word.

CRITICAL:
- Output ONE line only. Nothing else.
- No labels, no explanations, no markdown.
- Never output "SMALLTALK" or any case label.
</rules>

<conversation_history>
{history}
</conversation_history>

Current message: {query}
Output:"""