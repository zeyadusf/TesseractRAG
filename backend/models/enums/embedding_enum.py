from enum import Enum

class JinaEmbedTasks(str, Enum):
    EmbedDoc = "retrieval.passage"
    EmbedQuery = "retrieval.query"

