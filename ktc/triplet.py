from dataclasses import dataclass


@dataclass(frozen=True)
class Triplet:
    head: str
    relation: str
    tail: str

    def as_text(self) -> str:
        return f"{self.head} {self.relation} {self.tail}".strip()

    def to_dict(self) -> dict:
        return {"head": self.head, "relation": self.relation, "tail": self.tail}
