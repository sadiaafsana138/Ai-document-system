from app.embeddings import get_embeddings
from app.llm import generate_draft


def build_context_with_evidence(chunks):
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[EXCERPT {i + 1}]:\n{chunk.strip()}")
    return "\n\n".join(parts)


def run_rag_pipeline(query, vector_store, style_instructions=""):
    query_vec = get_embeddings([query])[0]

    raw_results = vector_store.search_with_scores(query_vec, k=5)

    filtered = [r for r in raw_results if r["score"] < 500]
    if not filtered:
        filtered = raw_results

    chunks = [r["text"] for r in filtered]
    scores = [r["score"] for r in filtered]

    context = build_context_with_evidence(chunks)

    draft = generate_draft(context, query, style_instructions)

    return {
        "query": query,
        "draft": draft,
        "evidence": [
            {"excerpt_number": i + 1, "text": chunk, "relevance_score": round(score, 4)}
            for i, (chunk, score) in enumerate(zip(chunks, scores))
        ],
    }