import os
from dotenv import load_dotenv

from app.ocr import extract_text_from_pdf
from app.chunking import chunk_text
from app.embeddings import get_embeddings
from app.vector_store import VectorStore
from app.rag import run_rag_pipeline
from app.feedback import save_edit, extract_style_patterns, get_style_instructions

load_dotenv()

DIVIDER = "=" * 65


def print_section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    print_section("PEARSON SPECTER LITT — AI Document System")

    print_section("STEP 1: Document Processing")

    pdf_path = "data/raw/sample.pdf"
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at {pdf_path}")
        return

    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pdf_text):,} characters from document.")
    print(f"Preview:\n{pdf_text[:300]}...")

    print_section("STEP 2: Chunking & Embedding")

    chunks = chunk_text(pdf_text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks.")

    vectors = get_embeddings(chunks)
    print(f"Embedding dimension: {len(vectors[0])}")

    store = VectorStore(dim=len(vectors[0]))
    store.add(vectors, chunks)
    print("Vector store built.")

    print_section("STEP 3: Grounded Draft Generation")

    query = (
        "Summarise the key facts, parties involved, dates, and legal issues "
        "present in this document. Produce a first-pass internal memo."
    )

    style_v1 = get_style_instructions()
    if style_v1:
        print(f"Found existing style preferences:\n{style_v1}\n")
    else:
        print("No previous style preferences found. Using defaults.")

    result_v1 = run_rag_pipeline(query, store, style_instructions=style_v1)

    print("\n--- DRAFT v1 ---\n")
    print(result_v1["draft"])

    print("\n--- SUPPORTING EVIDENCE ---")
    for ev in result_v1["evidence"]:
        print(f"\n[EXCERPT {ev['excerpt_number']}] (score={ev['relevance_score']}):")
        print(ev["text"][:250].strip() + "...")

    print_section("STEP 4: Operator Edit")

    edited_draft = (
        result_v1["draft"]
        + "\n\n"
        + "[OPERATOR EDITS APPLIED]\n"
        + "- Added a Risk Assessment section at the end.\n"
        + "- Replaced long paragraphs with bullet points.\n"
        + "- Removed hedging phrases like 'it appears' or 'it seems'.\n"
        + "- Every factual claim ends with its excerpt citation.\n"
        + "- Added a one-line Executive Summary at the top.\n"
    )

    save_edit(
        original_draft=result_v1["draft"],
        edited_draft=edited_draft,
        query=query,
    )
    print("Operator edit saved.")

    print_section("STEP 5: Learning Style Patterns")

    patterns = extract_style_patterns()
    print(f"\nLearned patterns:\n{patterns}")

    print_section("STEP 6: Improved Draft")

    result_v2 = run_rag_pipeline(query, store, style_instructions=patterns)

    print("\n--- DRAFT v2 (Improved) ---\n")
    print(result_v2["draft"])

    print_section("PIPELINE COMPLETE")
    print("Draft v1 generated without style history.")
    print("Operator edit captured and style patterns extracted.")
    print("Draft v2 generated using learned operator preferences.")


if __name__ == "__main__":
    main()