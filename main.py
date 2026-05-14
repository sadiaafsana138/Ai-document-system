from app.ocr import extract_text_from_pdf
from app.chunking import chunk_text
from app.embeddings import get_embeddings

text = extract_text_from_pdf("data/raw/sample.pdf")

chunks = chunk_text(text)

vectors = get_embeddings(chunks)

print("Chunks:", len(chunks))
print("Vectors shape:", len(vectors), len(vectors[0]))