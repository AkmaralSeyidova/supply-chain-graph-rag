import os
import sys
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

question = " ".join(sys.argv[1:]) or "Who supplies Zenith Motors?"
q_emb = model.encode(question).tolist()

results = supabase.rpc(
    "match_chunks", {"query_embedding": q_emb, "match_count": 3}
).execute().data

print(f"\nQuestion: {question}\n")
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content']}\n")