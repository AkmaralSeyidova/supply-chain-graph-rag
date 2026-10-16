import os
import re
import sys
import time
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

question = " ".join(sys.argv[1:]) or (
    "Which companies are upstream suppliers of the company "
    "that had a brake safety incident?"
)
print(f"\nQUESTION: {question}")

# 1. Vector search: find the best starting chunk
q_emb = model.encode(question).tolist()
top = supabase.rpc(
    "match_chunks", {"query_embedding": q_emb, "match_count": 3}
).execute().data
start_chunk = top[0]
print(f"\n[1] Best chunk ({start_chunk['similarity']:.2f}): {start_chunk['content']}")

# 2. Chunk -> companies (via entity_mentions)
rows = supabase.table("entity_mentions").select(
    "entities(id, name, type)"
).eq("chunk_id", start_chunk["id"]).execute().data
seeds = [r["entities"] for r in rows if r["entities"]["type"] == "Company"]
print(f"[2] Starting entities: {[s['name'] for s in seeds]}")
named = [s for s in seeds if s["name"].lower() in question.lower()]
if named:
    seeds = named

# 3. Graph traversal
direction = "down" if re.search(r"downstream|affected|impact", question, re.I) else "up"
best = {}
for s in seeds:
    edges = supabase.rpc(
        "walk_supplies",
        {"start_id": s["id"], "direction": direction, "max_depth": 3},
    ).execute().data
    for e in edges:
        key = (e["supplier"], e["buyer"])
        if key not in best or e["depth"] < best[key]["depth"]:
            best[key] = e

facts, chunk_ids = [], set()
for e in sorted(best.values(), key=lambda x: x["depth"]):
    facts.append(f"(hop {e['depth']}) {e['supplier']} supplies {e['products']} to {e['buyer']}")
    chunk_ids.add(e["chunk_id"])

print(f"[3] Graph facts (direction={direction}):")
for f in facts:
    print("   ", f)
# 4. Source text for the facts, plus the starting chunk
chunk_ids.add(start_chunk["id"])
texts = supabase.table("chunks").select("content").in_(
    "id", list(chunk_ids)
).execute().data
sources = "\n".join(f"- {t['content']}" for t in texts)

prompt = f"""Answer the question using ONLY the graph facts and source text below.
Explain the chain (who supplies whom). Distinguish direct suppliers (hop 1)
from indirect ones (hop 2 and beyond). If the information is not there, say so.

QUESTION: {question}

GRAPH FACTS:
{chr(10).join(facts)}

SOURCE TEXT:
{sources}
"""

for attempt in range(5):
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        break
    except Exception as ex:
        if attempt < 4 and any(x in str(ex) for x in ("503", "429", "UNAVAILABLE")):
            wait = 10 * 2 ** attempt
            print(f"  Server busy, retrying in {wait}s...")
            time.sleep(wait)
        else:
            raise

print("\nANSWER:\n", resp.text)