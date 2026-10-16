from enum import Enum
from pydantic import BaseModel, Field, model_validator

class EntityType(str, Enum):
               COMPANY = "Company"
               PRODUCT = "Product"
               LOCATION = "Location"
               INCIDENT = "Incident"

class RelType(str, Enum):
        SUPPLIES = "SUPPLIES"
        MANUFACTURES = "MANUFACTURES"
        LOCATED_IN = "LOCATED_IN"
        HAD_INCIDENT = "HAD_INCIDENT"
        SHIPS_FOR = "SHIPS_FOR"
 # relationship type -> (allowed source type, allowed target type)
ALLOWED = {
    RelType.SUPPLIES: (EntityType.COMPANY, EntityType.COMPANY),
    RelType.MANUFACTURES: (EntityType.COMPANY, EntityType.PRODUCT),
    RelType.LOCATED_IN: (EntityType.COMPANY, EntityType.LOCATION),
    RelType.HAD_INCIDENT: (EntityType.COMPANY, EntityType.INCIDENT),
    RelType.SHIPS_FOR: (EntityType.COMPANY, EntityType.COMPANY),
}


class Entity(BaseModel):
    name: str = Field(description="Canonical name, e.g. 'Zenith Motors'")
    type: EntityType
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str
    source_type: EntityType
    type: RelType
    target: str
    target_type: EntityType
    properties: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_ontology_rules(self):
        expected_source, expected_target = ALLOWED[self.type]
        if self.source_type != expected_source or self.target_type != expected_target:
            raise ValueError(
                f"{self.type.value} must go {expected_source.value} -> "
                f"{expected_target.value}, got {self.source_type.value} -> "
                f"{self.target_type.value}"
            )
        return self


class Extraction(BaseModel):
    """What the LLM must return for each chunk (used on Day 4)."""
    entities: list[Entity]
    relationships: list[Relationship]                      