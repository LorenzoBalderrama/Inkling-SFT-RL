"""Task data and strict response parsing. No service dependencies."""

import json
import re
from dataclasses import asdict, dataclass, field

FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    documents: tuple[Passage, ...]
    gold_answer: str | None
    supporting_ids: tuple[str, ...]
    domain: str = ""
    level: int = 1
    traps: tuple[str, ...] = ()
    # Grader-only: the year each trap plants ("" means any answer falls for it).
    trap_answers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        ids = [doc.id for doc in self.documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Document IDs must be nonempty and unique")
        if not set(self.supporting_ids) <= set(ids):
            raise ValueError("Supporting IDs must refer to documents")
        if (self.gold_answer is None) != (not self.supporting_ids):
            raise ValueError("Answerable examples need an answer and supporting evidence")
        if self.gold_answer in self.trap_answers.values():
            raise ValueError("A trap cannot plant the gold answer")

    @classmethod
    def from_dict(cls, value: dict) -> "Example":
        return cls(
            id=value["id"],
            question=value["question"],
            documents=tuple(Passage(**doc) for doc in value["documents"]),
            gold_answer=value["gold_answer"],
            supporting_ids=tuple(value["supporting_ids"]),
            domain=value.get("domain", ""),
            level=value.get("level", 1),
            traps=tuple(value.get("traps", ())),
            trap_answers=value.get("trap_answers", {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def prompt(self) -> str:
        documents = "\n".join(f"[{doc.id}] {doc.text}" for doc in self.documents)
        return (
            "You are a clerk at the Meridian Bureau of Records. Answer the request using only "
            "the records in this case file. Return exactly one JSON object with keys "
            '"answer" (string), "citations" (list of record IDs), and "abstain" (boolean). '
            "Give only the four-digit year in answer. Cite only the records that establish "
            "the answer. If the file does not establish the answer, return "
            '{"answer":"","citations":[],"abstain":true}.\n\n'
            f"{self.question}\nCase file:\n{documents}"
        )


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    abstain: bool

    @classmethod
    def parse(cls, text: str) -> "Answer | None":
        text = text.strip()
        if fenced := FENCED.fullmatch(text):
            text = fenced.group(1)
        try:
            value = json.loads(text, object_pairs_hook=_unique_keys)
        except (ValueError, RecursionError):
            return None
        if not isinstance(value, dict) or set(value) != {"answer", "citations", "abstain"}:
            return None
        if not isinstance(value["answer"], str) or type(value["abstain"]) is not bool:
            return None
        citations = value["citations"]
        if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            return None
        if len(citations) != len(set(citations)):
            return None
        if value["abstain"] and (value["answer"] or citations):
            return None
        return cls(value["answer"].strip(), tuple(citations), value["abstain"])
