"""RAG retriever scaffold."""


def retrieve_similar(query: str, k: int = 2) -> list[str]:
    # TODO: connect FAISS index in optional Phase 8.
    _ = query
    return [] if k <= 0 else []
