import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from data import DOCUMENTS

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    return model.encode(text).tolist()


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks, current = [], ""
    for s in sentences:
        if not s.endswith("."):
            s += "."
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


# Start clean (removes old test rows; chunks are deleted via cascade)
supabase.table("documents").delete().neq(
    "id", "00000000-0000-0000-0000-000000000000"
).execute()

for d in DOCUMENTS:
    doc = supabase.table("documents").insert(
        {"title": d["title"], "source": "demo"}
    ).execute().data[0]

    for i, chunk in enumerate(chunk_text(d["text"])):
        supabase.table("chunks").insert({
            "document_id": doc["id"],
            "content": chunk,
            "chunk_index": i,
            "embedding": embed(chunk),
        }).execute()

    print(f"Ingested: {d['title']}")

print("Done!")