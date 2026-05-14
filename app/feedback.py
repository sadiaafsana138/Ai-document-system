import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EDITS_FILE = "data/feedback/edit_history.json"
PATTERNS_FILE = "data/feedback/style_patterns.txt"


def _load_edits():
    if os.path.exists(EDITS_FILE):
        with open(EDITS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_edits(edits):
    os.makedirs(os.path.dirname(EDITS_FILE), exist_ok=True)
    with open(EDITS_FILE, "w", encoding="utf-8") as f:
        json.dump(edits, f, indent=2, ensure_ascii=False)


def save_edit(original_draft, edited_draft, query):
    edits = _load_edits()
    edits.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "original": original_draft,
            "edited": edited_draft,
        }
    )
    _save_edits(edits)
    print(f"[Feedback] Edit saved. Total edits on record: {len(edits)}")


def extract_style_patterns():
    edits = _load_edits()
    if not edits:
        print("[Feedback] No edits found. Nothing to learn from.")
        return ""

    recent = edits[-5:]

    examples_text = ""
    for i, edit in enumerate(recent):
        examples_text += f"""
--- Edit {i + 1} ---
TASK: {edit['query']}

ORIGINAL DRAFT:
{edit['original'][:600]}

EDITED VERSION:
{edit['edited'][:600]}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Analyse how a legal operator edits AI-generated drafts.
Extract 4-6 concrete reusable style preferences the operator consistently applies.
Return ONLY a bullet-point list.

{examples_text}""",
            }
        ],
        temperature=0.1,
        max_tokens=500,
    )

    patterns = response.choices[0].message.content.strip()

    os.makedirs(os.path.dirname(PATTERNS_FILE), exist_ok=True)
    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        f.write(patterns)

    print(f"[Feedback] Style patterns saved to {PATTERNS_FILE}")
    return patterns


def get_style_instructions():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""