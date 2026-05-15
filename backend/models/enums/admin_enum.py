from enum import Enum


class OrderDir(str, Enum):
    asc = "asc"
    desc = "desc"


class SessionOrderBy(str, Enum):
    created_at = "created_at"
    document_count = "document_count"
    message_count = "message_count"


class MessageOrderBy(str, Enum):
    created_at = "created_at"
    faithfulness = "faithfulness"
    answer_relevancy = "answer_relevancy"
    context_precision = "context_precision"
    context_recall = "context_recall"