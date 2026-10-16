import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types
from pydantic import ValidationError
from ontology import Entity, Relationship

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.8-flash" 

PROMPT = """You extract a knowledge graph from supply chain text.

ALLOWED ENTITY TYPES: Company, Product, Location, Incident

ALLOWED RELATIONSHIPS (source type -> target type):
- SUPPLIES: Company -> Company (source provides goods to target)
- MANUFACTURES: Company -> Product
- LOCATED_IN: Company -> Location
- HAD_INCIDENT: Company -> Incident
- SHIPS_FOR: Company -> Company (logistics company ships goods for target)

RULES:
1. Use ONLY the types above. If a fact does not fit, skip it.
2. SUPPLIES always goes from the supplier to the buyer. Never create BUYS_FROM.
   "X buys steel from Y" means (Y) SUPPLIES (X).
3. Put the supplied product in the SUPPLIES properties, e.g. {"product": "steel plates"}.
   Do not create Product entities for supplied goods.
4. Resolve pronouns ("It", "Its") to the company they refer to.
5. Every Incident needs a short, specific name, e.g. "Pacific Rubber factory closure 2024".
   Properties use snake_case keys and clean values: {"date": "2024-03", "kind": "recall",
   "cause": "brake failure"}. Numbers are numbers.
6. Every relationship must include source, source_type, type, target, target_type, properties.
7. Only extract what the text states. Do not guess.

Return ONLY JSON in this shape:
{"entities": [{"name": "...", "type": "...", "properties": {}}],
 "relationships": [{"source": "...", "source_type": "...", "type": "...",
                    "target": "...", "target_type": "...", "properties": {}}]}

TEXT:
{TEXT}
"""

def call_llm(text: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.replace("{TEXT}", text),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            return json.loads(response.text)
        except Exception as ex:
            msg = str(ex)
            temporary = "503" in msg or "429" in msg or "UNAVAILABLE" in msg
            if temporary and attempt < retries - 1:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s
                print(f"  Server busy, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def extract(text: str):
    """Call the LLM, then validate every item against the ontology."""
    raw = call_llm(text)
    entities, relationships, rejected = [], [], []

    for e in raw.get("entities", []):
        try:
            entities.append(Entity(**e))
        except ValidationError as err:
            rejected.append({"item": e, "reason": err.errors()[0]["msg"]})

    for r in raw.get("relationships", []):
        try:
            relationships.append(Relationship(**r))
        except ValidationError as err:
            rejected.append({"item": r, "reason": err.errors()[0]["msg"]})

    return entities, relationships, rejected


chunks = supabase.table("chunks").select("id, content").execute().data
results = []

for c in chunks:
    print(f"\n=== {c['content'][:70]}...")
    try:
        entities, relationships, rejected = extract(c["content"])
    except Exception as ex:
        print(f"  FAILED: {ex}")
        continue

    for r in relationships:
        props = f" {r.properties}" if r.properties else ""
        print(f"  ({r.source}:{r.source_type.value}) -[{r.type.value}{props}]-> "
              f"({r.target}:{r.target_type.value})")
    for e in entities:
        if e.type.value == "Incident":
            print(f"  Incident node: {e.name} {e.properties}")
    for bad in rejected:
        print(f"  REJECTED: {bad['reason']} <- {bad['item']}")

    results.append({
        "chunk_id": c["id"],
        "entities": [e.model_dump(mode="json") for e in entities],
        "relationships": [r.model_dump(mode="json") for r in relationships],
        "rejected": rejected,
    })
    time.sleep(5)  # stay under the free-tier rate limit

with open("extraction_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nDone! Saved extraction_results.json")