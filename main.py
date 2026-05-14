from app.ocr import extract_text_from_pdf
from app.chunking import chunk_text
from app.embeddings import get_embeddings
from app.vector_store import VectorStore

pdf_text = extract_text_from_pdf("data/raw/sample.pdf")

chunks = chunk_text(pdf_text)

vectors = get_embeddings(chunks)

store = VectorStore(dim=len(vectors[0]))
store.add(vectors, chunks)

query = "what is this document about?"

query_vec = get_embeddings([query])[0]

results = store.search(query_vec)

print("\n--- TOP MATCHES ---\n")
for r in results:
    print(r)