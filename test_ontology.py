from pydantic import ValidationError
from ontology import Relationship, Entity, EntityType, RelType

# Valid: Acme Steel supplies BuildCo
ok = Relationship(
    source="Acme Steel", source_type="Company",
    type="SUPPLIES",
    target="BuildCo", target_type="Company",
    properties={"product": "steel plates"},
)
print("VALID:", ok.source, ok.type.value, ok.target)

# Invalid type name (LLM invented a label)
try:
    Relationship(
        source="Acme Steel", source_type="Company",
        type="WORKS_WITH",
        target="BuildCo", target_type="Company",
    )
except ValidationError as e:
    print("\nREJECTED (unknown relationship type):")
    print(e.errors()[0]["msg"])

# Invalid direction/types (a Product cannot supply a Company)
try:
    Relationship(
        source="steel plates", source_type="Product",
        type="SUPPLIES",
        target="BuildCo", target_type="Company",
    )
except ValidationError as e:
    print("\nREJECTED (wrong entity types):")
    print(e.errors()[0]["msg"])


