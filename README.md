# Pearson Specter Litt -- AI Document System
End-to-end pipeline: ingest messy legal documents, extract text,
retrieve relevant evidence, generate grounded draft memos, and
improve over time from operator edits.
## Project Structure
  app/
    ocr.py           PDF text extraction (PyMuPDF + Tesseract fallback)
    chunking.py      Sliding-window text chunker
    embeddings.py    Sentence-transformers vector embeddings
    vector_store.py  FAISS similarity search
    rag.py           Retrieval + context building + generation
    llm.py           Groq LLM integration
    feedback.py      Edit capture and style pattern learning
  main.py            Full pipeline entry point
  data/raw/          Place input PDFs here
  data/feedback/     Auto-generated: edit history + learned patterns
  .env               Your GROQ_API_KEY (never commit this)
## Setup
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  Create .env in project root:
    GROQ_API_KEY=your_key_here
  Get a free key at: console.groq.com
## Run
  python main.py
## How It Works
  1. OCR       -- extract text from PDF (typed or scanned)
  2. Chunking  -- split into 500-char overlapping pieces
  3. Embed     -- convert chunks to 384-dim vectors
  4. FAISS     -- index vectors for similarity search
  5. RAG       -- embed query, retrieve top chunks, label them
  6. LLM       -- Groq Llama 3.3 70B writes grounded draft
  7. Feedback  -- save operator edit, extract style patterns
  8. Improved  -- re-run with learned style injected
## Key Design Decisions
  Grounding: LLM only sees retrieved excerpts. Every claim must
  cite [EXCERPT N]. Evidence Gaps section is mandatory.
  Feedback: No fine-tuning. Edits stored as (original, edited,
  query) triples. LLM extracts bullet-point style rules. These
  are injected into the system prompt on the next run.
  Messy docs: ocr.py handles each page independently.
  Typed pages use direct extraction.
  Scanned pages fall back to Tesseract OCR
