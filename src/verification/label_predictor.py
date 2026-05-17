import json
import logging

from agent.agent import Agent
from core.exceptions import LLMClientError
from models.claim import ClaimLabel
from prompts.label import LABEL_PROMPT

logger = logging.getLogger(__name__)

_LABEL_MAP: dict[str, ClaimLabel] = {
    "SUPPORTED": ClaimLabel.SUPPORTED,
    "REFUTED": ClaimLabel.REFUTED,
    "NOT_ENOUGH_EVIDENCE": ClaimLabel.NOT_ENOUGH_EVIDENCE,
    "CONFLICTING_EVIDENCE": ClaimLabel.CONFLICTING_EVIDENCE,
}


class LabelPredictor:
    def __init__(self, *, agent: Agent) -> None:
        self._agent = agent

    def predict(self, claim: str, evidence: str) -> ClaimLabel:
        prompt = LABEL_PROMPT.format(claim=claim, evidence=evidence)
        try:
            text = self._agent.run(prompt, max_tokens=100, temperature=0.0)
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"Label prediction failed: {exc}") from exc

        try:
            data = json.loads(text)
            raw_label = data.get("predicted_label", "")
        except json.JSONDecodeError:
            logger.warning("Could not parse label response: %s", text[:200])
            return ClaimLabel.NOT_ENOUGH_EVIDENCE

        label = _LABEL_MAP.get(raw_label)
        if label is None:
            logger.warning("Unknown label '%s', defaulting to NOT_ENOUGH_EVIDENCE", raw_label)
            return ClaimLabel.NOT_ENOUGH_EVIDENCE
        return label
