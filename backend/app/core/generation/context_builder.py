import hashlib
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = get_settings().MAX_CONTEXT_CHARS


class ContextBuilder:

    def build(self, chunks: list[dict]) -> str:
        seen_hashes = set()
        context_parts = []
        total_chars = 0

        for chunk in chunks:
            # Step 1: get the text content from the chunk dict
            content = chunk['content']

            # Step 2: generate MD5 hash of the content string
            # hashlib.md5(content.encode()).hexdigest() gives you the hash
            chunk_hash = hashlib.md5(content.encode()).hexdigest()

            # Step 3: if this hash is already in seen_hashes, skip this chunk
            if chunk_hash in seen_hashes :
                continue

            # Step 4: check if adding this chunk would exceed MAX_CONTEXT_CHARS
            if total_chars + len(content) > MAX_CONTEXT_CHARS:
                break

            # Step 5: add the hash to seen_hashes
            seen_hashes.add(chunk_hash)

            # Step 6: update total_chars
            total_chars += len(content)

            # Step 7: format the chunk and append to context_parts
            # Format: [Source: filename, Chunk N]\ntext
            formatted = f"[Source : {chunk['document_name']}, Chunk {chunk['chunk_index']}]\n{content}"
            context_parts.append(formatted)
        logger.info(f'ContextBuilder return {total_chars} char..')
        return "\n\n".join(context_parts)

