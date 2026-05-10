import json
import logging

import anthropic
from anthropic.types import TextBlock

from core.exceptions import LLMClientError
from models.claim import ClaimLabel

logger = logging.getLogger(__name__)

_LABEL_PROMPT = """
سوف تحصل على ادعاء ومجموعة من الأدلة. حدد الحكم على الادعاء بناءً على الأدلة فقط، دون الاعتماد على أي معرفة خارجية.

التصنيفات المتاحة:
1. SUPPORTED - الأدلة تدعم الادعاء بشكل واضح.
2. REFUTED - الأدلة تناقض الادعاء بشكل مباشر أو تجعله غير مرجح.
3. NOT_ENOUGH_EVIDENCE - لا توجد أدلة كافية لدعم أو نفي الادعاء.
4. CONFLICTING_EVIDENCE - توجد أدلة متناقضة أو انتقائية.

تعليمات:
- اعتمد فقط على الأدلة المقدمة.
- أعد فقط كائن JSON يحتوي على المفتاح "predicted_label".

الادعاء: {claim}
الأدلة: {evidence}

أعد الإجابة بهذا الشكل فقط:
{{ "predicted_label": "SUPPORTED" }}
"""

_LABEL_MAP: dict[str, ClaimLabel] = {
    "SUPPORTED": ClaimLabel.SUPPORTED,
    "REFUTED": ClaimLabel.REFUTED,
    "NOT_ENOUGH_EVIDENCE": ClaimLabel.NOT_ENOUGH_EVIDENCE,
    "CONFLICTING_EVIDENCE": ClaimLabel.CONFLICTING_EVIDENCE,
}


class LabelPredictor:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def predict(self, claim: str, evidence: str) -> ClaimLabel:
        prompt = _LABEL_PROMPT.format(claim=claim, evidence=evidence)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            block = next((b for b in resp.content if isinstance(b, TextBlock)), None)
            if block is None:
                raise LLMClientError("No text block in LLM response")
            text = block.text.strip()
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
