# Pearson Specter Litt — AI Document System

An end-to-end AI pipeline that ingests messy legal documents, extracts structured content, generates grounded draft memos, and improves over time from operator edits.

---

## Architecture

```
data/raw/          ← input PDFs / images
data/feedback/     ← saved operator edits + learned style patterns

app/
  ocr.py           ← PDF text extraction (PyMuPDF + Tesseract OCR fallback)
  chunking.py      ← sliding-window text chunker
  embeddings.py    ← sentence-transformers embeddings (all-MiniLM-L6-v2)
  vector_store.py  ← FAISS flat-L2 index with score retrieval
  rag.py           ← retrieval + context building + draft generation
  llm.py           ← Groq (Llama 3.3 70B) draft generation with style injection
  feedback.py      ← edit capture, pattern extraction, style persistence

main.py            ← full pipeline demo (run this)
create_sample_pdf.py ← generates a synthetic legal PDF for testing
```

### How It Works

```
PDF/Image
   │
   ▼
[OCR Layer]  PyMuPDF for text PDFs; Tesseract for scanned/image pages
   │
   ▼
[Chunker]    500-char sliding window, 50-char overlap
   │
   ▼
[Embeddings] all-MiniLM-L6-v2 sentence-transformers
   │
   ▼
[FAISS Store] flat L2 index, returns text + relevance score
   │
   ▼
[RAG Layer]  top-5 chunks → labelled context [EXCERPT N]
   │
   ▼
[LLM Draft]  Groq Llama 3.3 70B → grounded memo with citations
   │
   ▼
[Operator Edit] captured as (original, edited, query) in JSON
   │
   ▼
[Pattern Extraction] LLM analyses edits → bullet-point style rules
   │
   ▼
[Improved Draft]  same query, same retrieval + injected style rules
```

---

## Setup

### Prerequisites
- Python 3.11+
- Tesseract OCR installed on your system
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Mac: `brew install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`

### Install dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### Set your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at https://console.groq.com

---

## Running

### 1. Create a sample document (first time only)

```bash
pip install fpdf2
python create_sample_pdf.py
```

This creates `data/raw/sample.pdf` — a synthetic legal case file.

### 2. Run the full pipeline

```bash
python main.py
```

**What you will see:**
1. Text extracted from the PDF
2. Chunks and embeddings created
3. **Draft v1** — grounded internal memo with `[EXCERPT N]` citations
4. Supporting evidence listed with relevance scores
5. Simulated operator edit saved to `data/feedback/`
6. Style patterns extracted by LLM
7. **Draft v2** — improved memo using learned operator preferences

---

## Key Design Decisions

### Grounding over generation
The LLM is given numbered excerpts and instructed to cite every claim. An "Evidence Gaps" section is required at the end of every draft, explicitly surfacing what the source material does not cover.

### Feedback loop
Operator edits are stored as `(original, edited, query)` triples. A separate LLM call analyses the diffs and extracts bullet-point style rules. These rules are injected into the system prompt on the next run — no fine-tuning required, no model retraining.

### Messy document handling
`ocr.py` first attempts direct text extraction (PyMuPDF). If a page returns no text (scanned or image-only), it renders that page to an image and runs Tesseract OCR. Mixed documents (some typed pages, some scanned) are handled page by page.

### Retrieval transparency
`vector_store.py` returns L2 distances alongside texts. `rag.py` exposes the full evidence list with scores in its return value so any caller can inspect what supported each draft.

---

## Tradeoffs & Limitations

| Decision | Tradeoff |
|---|---|
| FAISS flat L2 index | Fast and simple; not scalable to millions of docs |
| all-MiniLM-L6-v2 | Lightweight and fast; larger models give better retrieval |
| Simulated operator edit in demo | Real system needs a UI diff or API endpoint |
| Style patterns stored as plain text | Simple and inspectable; a structured schema would be more robust |
| Groq Llama 3.3 70B | Free tier, fast; GPT-4o would give higher draft quality |

---

## Sample Output

```
INTERNAL MEMO — PEARSON SPECTER LITT LLP
Re: Harrington v. Nexcore Pharmaceuticals

EXECUTIVE SUMMARY
Ms. Harrington was terminated on January 15, 2024, 65 days after filing
an internal whistleblower complaint [EXCERPT 2], raising a strong
retaliation claim under Sarbanes-Oxley Section 806.

KEY FACTS
- Plaintiff employed at Nexcore January 2018 – January 2024 [EXCERPT 1]
- Whistleblower complaint filed November 12, 2023 [EXCERPT 3]
...

EVIDENCE GAPS
- Full text of Nexcore Data Protection Policy not available in excerpts
- IT log timestamps not included in source material
```

---

## Evaluation

| Criterion | Approach |
|---|---|
| Extraction quality | Manual review of extracted text vs. source PDF |
| Retrieval relevance | L2 scores logged; top-5 chunks inspected per run |
| Draft grounding | Every claim must carry [EXCERPT N]; un-cited claims = failure |
| Improvement | Draft v2 checked for presence of operator-requested elements |
| Code quality | Modular app/ package, single responsibility per file |
