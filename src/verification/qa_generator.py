import json
import logging
from datetime import datetime

import anthropic
from anthropic.types import TextBlock

from core.exceptions import JSONParsingError, LLMClientError
from models.verification import QAPair

logger = logging.getLogger(__name__)

_EVIDENCE_QA_PROMPT = """
أنت مساعد للتحقق من الأخبار. لديك **ادعاء**، و**أدلة مستردة** من مصادر متعددة. مهمتك هي توليد أزواج سؤال وجواب دقيقة ومباشرة.

قواعد مهمة:
1. يجب أن تكون الأسئلة والأجوبة قابلة للإجابة استنادًا حصريًا إلى الأدلة المعطاة.
2. لا تخترع أي معلومة غير موجودة في الأدلة.
3. اربط كل سؤال مباشرة بالادعاء.
4. وضّح في الإجابة كيف يمكن استخدام الدليل للتحقق من الادعاء، مع مراعاة الصلاحية الزمنية للمصدر.
5. الإخراج يجب أن يكون بصيغة JSON صحيحة فقط.

المدخلات:
الادعاء: {claim}
تاريخ الادعاء: {claim_date}
الأدلة المسترجعة: {evidence}

الإخراج المطلوب (JSON صحيح فقط):
{{
    "qa_pairs": [
        {{
            "question": "سؤال مرتبط بالتحقق من الادعاء بناءً على الدليل",
            "answer": "إجابة مستندة إلى الدليل"
        }}
    ]
}}
"""

_GOLD_QA_PROMPT = """
أنت خبير في تحليل مقالات التحقق من الحقائق. ستتلقى نص الادعاء، مقال التحقق من الحقائق، والمصادر المرتبطة به.
مهمتك هي تحليل هذه المواد لإنتاج أزواج سؤال-جواب تشرح كيف تم استخدام المصادر للتحقق من الادعاء.

الادعاء: {claim}
مقال التحقق من الحقائق: {fact_check_content}
المصادر المستشهد بها: {source_content}

**مهم جداً: قدم إجابتك كـ JSON صحيح فقط.**
{{
    "qa_pairs": [
        {{
            "question": "السؤال المستخرج",
            "answer": "الجواب المستند إلى المصدر"
        }}
    ]
}}
"""


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
    def __init__(self, api_key: str, model: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call(self, prompt: str) -> dict[str, object]:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            block = next((b for b in resp.content if isinstance(b, TextBlock)), None)
            if block is None:
                raise LLMClientError("No text block in LLM response")
            return _extract_json(block.text)
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
        prompt = _EVIDENCE_QA_PROMPT.format(
            claim=claim, claim_date=date_str, evidence=evidence_text
        )
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
        prompt = _GOLD_QA_PROMPT.format(
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
