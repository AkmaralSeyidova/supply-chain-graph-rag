import os
import re
import json
from collections import Counter
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

SUFFIXES = {"inc", "corp", "corporation", "ltd", "llc", "co", "company", "gmbh", "plc"}
KEY_ALIASES = {
    "affected_units": "vehicles_recalled",
    "vehicles": "vehicles_recalled",
    "units_recalled": "vehicles_recalled",
    "injured": "injuries",
}
SNAKE_VALUE_KEYS = {"kind", "cause"}


def snake(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s.strip().lower()).strip("_")


def match_key(name: str) -> str:
    """'BuildCo Inc.' and 'buildco' -> 'buildco'"""
    s = re.sub(r"[^\w\s]", " ", name.lower())
    return " ".join(w for w in s.split() if w not in SUFFIXES)


def clean_props(props: dict) -> dict:
    out = {}
    for k, v in (props or {}).items():
        k = KEY_ALIASES.get(snake(k), snake(k))
        if k in SNAKE_VALUE_KEYS and isinstance(v, str):
            v = snake(v)
        out[k] = v
    return out


with open("extraction_results.json", encoding="utf-8") as f:
    results = json.load(f)

# ---------- 1. Entity resolution ----------
nodes = {}      # (type, key) -> {names, props, chunks}


def add_node(name, etype, props, chunk_id):
    k = (etype, match_key(name))
    n = nodes.setdefault(k, {"names": Counter(), "props": {}, "chunks": set()})
    n["names"][name] += 1
    n["props"].update(clean_props(props))
    n["chunks"].add(chunk_id)


for item in results:
    cid = item["chunk_id"]
    for e in item["entities"]:
        add_node(e["name"], e["type"], e["properties"], cid)
    for r in item["relationships"]:      # endpoints may be missing from entities
        add_node(r["source"], r["source_type"], {}, cid)
        add_node(r["target"], r["target_type"], {}, cid)

# canonical name = most frequent spelling; keep other spellings as aliases
for (etype, key), n in nodes.items():
    n["name"] = n["names"].most_common(1)[0][0]
    aliases = sorted(set(n["names"]) - {n["name"]})
    if aliases:
        n["props"]["aliases"] = aliases

print(f"Resolved to {len(nodes)} unique entities")

# ---------- 2. Write to Supabase ----------
# Start clean. Deleting entities cascades to relationships and entity_mentions.
supabase.table("entities").delete().neq(
    "id", "00000000-0000-0000-0000-000000000000"
).execute()

ids = {}   # (type, key) -> uuid
for (etype, key), n in nodes.items():
    row = supabase.table("entities").insert({
        "name": n["name"],
        "type": etype,
        "properties": n["props"],
        "embedding": model.encode(n["name"]).tolist(),
    }).execute().data[0]
    ids[(etype, key)] = row["id"]

# ---------- 3. Relationships (merged) ----------
# One edge per (source, type, target). Products are collected into a list.
merged = {}
for item in results:
    for r in item["relationships"]:
        s = ids[(r["source_type"], match_key(r["source"]))]
        t = ids[(r["target_type"], match_key(r["target"]))]
        m = merged.setdefault(
            (s, r["type"], t),
            {"props": {}, "chunk_id": item["chunk_id"]},
        )
        for k, v in clean_props(r["properties"]).items():
            if k == "product":
                products = m["props"].setdefault("product", [])
                if v not in products:
                    products.append(v)
            else:
                m["props"].setdefault(k, v)

rel_count = 0
for (s, rtype, t), m in merged.items():
    supabase.table("relationships").insert({
        "source_id": s,
        "target_id": t,
        "type": rtype,
        "properties": m["props"],
        "chunk_id": m["chunk_id"],   # provenance (first chunk that stated it)
    }).execute()
    rel_count += 1

# ---------- 4. Mentions (the bridge to the text) ----------
mentions = [
    {"entity_id": ids[k], "chunk_id": cid}
    for k, n in nodes.items() for cid in n["chunks"]
]
supabase.table("entity_mentions").insert(mentions).execute()

print(f"Wrote {len(ids)} entities, {rel_count} relationships, {len(mentions)} mentions")