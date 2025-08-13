import os
import sys
import json
import anthropic
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.config.settings import CLAUDE_SONNET_4

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@dataclass
class QAPair:
    question: str
    answer: str
    source_evidence: str
    confidence_score: float
    question_type: (
        str  # "factual", "numerical", "source_verification", "timeline", etc.
    )


class FactCheckQAGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = CLAUDE_SONNET_4

    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract JSON from Claude's response, handling markdown code blocks"""
        response_text = response_text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()
        elif response_text.startswith("```"):
            # Handle plain ``` blocks
            response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        # Find JSON boundaries
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            return json.loads(json_text)
        else:
            raise ValueError(
                f"No valid JSON found in response: {response_text[:200]}..."
            )

    def generate_qa_pairs(
        self, fact_check_content: str, source_content: str, source_url: str = ""
    ) -> Dict:
        """Generate Q&A pairs by analyzing fact-check article and its source"""

        prompt = f"""
        أنت خبير في تحليل مقالات التحقق من الحقائق. قم بتحليل مقال التحقق من الحقائق والمصدر المرتبط به لإنتاج أزواج سؤال-جواب.

        مقال التحقق من الحقائق:
        {fact_check_content}

        المصدر المستشهد به:
        {source_content}

        رابط المصدر: {source_url}

        المطلوب:
        1. ما هي الأسئلة التي استخدم مقال التحقق هذا المصدر للإجابة عليها؟
        2. ما هي المعلومات المحددة من المصدر التي تم استخدامها؟
        3. كيف تم استخدام هذه المعلومات في التحقق؟

        **مهم جداً: قدم إجابتك كـ JSON صحيح فقط، بدون أي نص إضافي قبل أو بعد JSON.**
        {{
            "qa_pairs": [
                {{
                    "question": "السؤال المستخرج",
                    "answer": "الجواب من المصدر",
                    
                }}
            ]
        }}
        """

        # "source_evidence": "النص المحدد من المصدر المستخدم كدليل",
        # "confidence_score": 0.95,
        # "question_type": "factual",
        # "verification_method": "كيف تم التحقق من هذه المعلومة"

        # "source_usage_analysis": {{
        #     "main_purpose": "الغرض الرئيسي من استخدام هذا المصدر",
        #     "information_extracted": ["قائمة بالمعلومات المستخرجة"],
        #     "credibility_factors": ["العوامل التي تدعم مصداقية المصدر"],
        #     "limitations": ["قيود أو نقاط ضعف في المصدر"]
        # }},
        # "relationship_mapping": {{
        #     "direct_citations": ["الاقتباسات المباشرة"],
        #     "implicit_references": ["المراجع الضمنية"],
        #     "contradictions": ["أي تناقضات محتملة"]
        # }}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return self._extract_json_from_response(response.content[0].text)

        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response.content[0].text[:500] + "..."
                if "response" in locals() and len(response.content[0].text) > 500
                else response.content[0].text,
            }
        except Exception as e:
            return {"error": f"Failed to generate Q&A pairs: {str(e)}"}

    def generate_evidence_qa_pairs(
        self, claim: str, evidence_content: str, evidence_url: str = ""
    ) -> Dict:
        """Generate Q&A pairs for the retrieved evidence given the claim and a retrieved evidence"""

        prompt = f"""
        أنت خبير في تحليل مقالات التحقق من الحقائق. قم بتحليل الادعاء والمصدر المرتبط به لإنتاج أزواج سؤال-جواب.

        الادعاء:
        {claim}

        المصدر المستشهد به:
        {evidence_content}

        رابط المصدر: {evidence_url}

        المطلوب:
        1. ما هي المعلومات المحددة من المصدر التي يمكن استخدامها؟
        2. كيف يمكن استخدام هذه المعلومات في التحقق؟

        **مهم جداً: قدم إجابتك كـ JSON صحيح فقط، بدون أي نص إضافي قبل أو بعد JSON.**
        {{
            "qa_pairs": [
                {{
                    "question": "السؤال المستخرج",
                    "answer": "الجواب من المصدر",
                }}
            ]
        }}
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return self._extract_json_from_response(response.content[0].text)

        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response.content[0].text[:500] + "..."
                if "response" in locals() and len(response.content[0].text) > 500
                else response.content[0].text,
            }
        except Exception as e:
            return {"error": f"Failed to generate Q&A pairs: {str(e)}"}

    def extract_specific_questions(
        self, fact_check_content: str, source_content: str
    ) -> List[str]:
        """Extract specific questions that the source was used to answer"""

        prompt = f"""
        بناءً على مقال التحقق من الحقائق والمصدر التالي، استخرج الأسئلة المحددة التي تم استخدام المصدر للإجابة عليها:

        مقال التحقق:
        {fact_check_content}

        المصدر:
        {source_content}

        استخرج فقط الأسئلة التي يمكن الإجابة عليها مباشرة من المصدر. قدم الإجابة كقائمة JSON:
        {{
            "questions": [
                "السؤال الأول",
                "السؤال الثاني"
            ]
        }}
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            result = json.loads(response.content[0].text)
            return result.get("questions", [])

        except Exception as e:
            return [f"Error extracting questions: {str(e)}"]

    def identify_source_information_usage(
        self, fact_check_content: str, source_content: str
    ) -> Dict:
        """Identify what specific information from source was used in fact-checking"""

        prompt = f"""
        قم بتجدبد المعلومات المحددة من المصدر التي تم استخدامها في مقال التحقق:

        مقال التحقق:
        {fact_check_content}

        المصدر:
        {source_content}

        حدد:
        1. المعلومات المستخدمة مباشرة
        2. المعلومات المشار إليها ضمنياً
        3. الأرقام والإحصائيات المستخدمة
        4. التواريخ والأحداث المرجعية
        5. أي معلومات من المصدر لم يتم استخدامها

        النتيجة بصيغة JSON:
        {{
            "used_information": {{
                "direct_quotes": ["الاقتباسات المباشرة"],
                "statistics": ["الأرقام والإحصائيات"],
                "dates_events": ["التواريخ والأحداث"],
                "implicit_references": ["المراجع الضمنية"]
            }},
            "unused_information": ["المعلومات غير المستخدمة من المصدر"],
            "information_transformation": [
                {{
                    "original": "النص الأصلي من المصدر",
                    "used_as": "كيف تم استخدامه في مقال التحقق",
                    "purpose": "الغرض من الاستخدام"
                }}
            ]
        }}
        """

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return json.loads(response.content[0].text)

        except Exception as e:
            return {"error": f"Failed to identify information usage: {str(e)}"}

    def batch_process_fact_check_sources(
        self, fact_check_source_pairs: List[Tuple[str, str, str]]
    ) -> List[Dict]:
        """Process multiple fact-check and source pairs in batch"""
        results = []

        for i, (fact_check, source, url) in enumerate(fact_check_source_pairs):
            print(f"Processing pair {i + 1}/{len(fact_check_source_pairs)}")

            # Generate comprehensive analysis
            qa_analysis = self.generate_qa_pairs(fact_check, source, url)

            # Extract specific questions
            questions = self.extract_specific_questions(fact_check, source)

            # Identify information usage
            info_usage = self.identify_source_information_usage(fact_check, source)

            result = {
                "pair_id": i,
                "source_url": url,
                "qa_analysis": qa_analysis,
                "extracted_questions": questions,
                "information_usage": info_usage,
                "processing_timestamp": "2025-07-23",  # You'd use actual timestamp
            }

            results.append(result)

        return results

    def generate_training_dataset(
        self, results: List[Dict], output_file: str = "qa_training_data.json"
    ):
        """Convert results into training dataset format"""

        training_data = []

        for result in results:
            if "qa_analysis" in result and "qa_pairs" in result["qa_analysis"]:
                for qa_pair in result["qa_analysis"]["qa_pairs"]:
                    training_example = {
                        "instruction": "استخرج السؤال والجواب من مقال التحقق والمصدر المرتبط",
                        "input": {
                            "fact_check_excerpt": "جزء من مقال التحقق",  # You'd extract relevant parts
                            "source_excerpt": qa_pair.get("source_evidence", ""),
                            "source_url": result.get("source_url", ""),
                        },
                        "output": {
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"],
                            "confidence": qa_pair.get("confidence_score", 0.0),
                            "type": qa_pair.get("question_type", ""),
                        },
                    }
                    training_data.append(training_example)

        # Save training data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        return training_data


class FactCheckDatasetBuilder:
    def __init__(self, api_key: str):
        self.qa_generator = FactCheckQAGenerator(api_key)

    def process_single_pair(
        self, fact_check_text: str, source_text: str, source_url: str = ""
    ):
        """Process a single fact-check/source pair"""

        print("\nGenerating Q&A pairs...\n")
        qa_result = self.qa_generator.generate_qa_pairs(
            fact_check_text, source_text, source_url
        )

        # print("Extracting specific questions...")
        # questions = self.qa_generator.extract_specific_questions(
        #     fact_check_text, source_text
        # )

        # print("Analyzing information usage...")
        # info_usage = self.qa_generator.identify_source_information_usage(
        #     fact_check_text, source_text
        # )

        # return {
        #     "qa_pairs": qa_result,
        #     "questions": questions,
        #     "information_usage": info_usage,
        # }

        return qa_result
