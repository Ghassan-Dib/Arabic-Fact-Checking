import json
import logging
from datetime import datetime

from agent.agent import Agent
from core.exceptions import JSONParsingError, LLMClientError
from models.verification import QAPair
from prompts.qa import EVIDENCE_QA_PROMPT, GOLD_QA_PROMPT

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise JSONParsingError("No JSON object found in LLM response", raw_response=text)

    try:
        result: dict[str, object] = json.loads(text[start:end])
        return result
    except json.JSONDecodeError as exc:
        raise JSONParsingError(str(exc), raw_response=text[start:end]) from exc


class QAGenerator:
    def __init__(self, *, agent: Agent) -> None:
        self._agent = agent

    def _call(self, prompt: str) -> dict[str, object]:
        try:
            text = self._agent.run(prompt, max_tokens=3000, temperature=0.1)
            return _extract_json(text)
        except (JSONParsingError, LLMClientError):
            raise
        except Exception as exc:
            raise LLMClientError(f"LLM call failed: {exc}") from exc

    def generate_from_evidence(
        self,
        claim: str,
        evidence_text: str,
        claim_date: datetime | None = None,
    ) -> list[QAPair]:
        date_str = claim_date.strftime("%Y-%m-%d") if claim_date else "unknown"
        prompt = EVIDENCE_QA_PROMPT.format(claim=claim, claim_date=date_str, evidence=evidence_text)
        data = self._call(prompt)
        raw = data.get("qa_pairs", [])
        raw_pairs = raw if isinstance(raw, list) else []
        return [
            QAPair(question=p["question"], answer=p["answer"])
            for p in raw_pairs
            if isinstance(p, dict) and p.get("question") and p.get("answer")
        ]

    def generate_from_gold_evidence(
        self,
        claim: str,
        fact_check_content: str,
        source_content: str,
    ) -> list[QAPair]:
        prompt = GOLD_QA_PROMPT.format(
            claim=claim,
            fact_check_content=fact_check_content,
            source_content=source_content,
        )
        data = self._call(prompt)
        raw = data.get("qa_pairs", [])
        raw_pairs = raw if isinstance(raw, list) else []
        return [
            QAPair(question=p["question"], answer=p["answer"])
            for p in raw_pairs
            if isinstance(p, dict) and p.get("question") and p.get("answer")
        ]
