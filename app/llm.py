import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a legal document analyst at Pearson Specter Litt law firm.
Generate grounded draft summaries based ONLY on the provided document excerpts.
Rules:
- Never add facts not present in the source excerpts
- Reference every claim with its excerpt number e.g. [EXCERPT 2]
- If source material is insufficient, say so explicitly
- Write in professional legal tone"""


def generate_draft(context: str, query: str, style_instructions: str = "") -> str:

    system = SYSTEM_PROMPT

    if style_instructions:
        system += f"\n\nOperator preferences from previous edits:\n{style_instructions}"

    user_message = f"""Using ONLY the document excerpts below, generate a first-pass internal memo.

=== DOCUMENT EXCERPTS ===
{context}

=== TASK ===
{query}

=== INSTRUCTIONS ===
- Cite each excerpt you use with [EXCERPT N]
- Use headers to structure the memo
- End with an "Evidence Gaps" section listing what the excerpts do NOT cover
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return response.choices[0].message.content