import tqdm
import copy
import json
import time
import numpy as np
import anthropic

from src.config.settings import ANTHROPIC_API_KEY, CLAUDE_SONNET_4


class EV2REvaluator:
    # Config
    MAX_RETRIES = 10
    ev2r_reporting_levels = [0.5]
    # LLM
    MAX_TOKENS = 3000
    TEMPERATURE = 0  # or 0.1

    def __init__(self, properties=None):
        self.properties = properties
        self.prompt_type = properties.PromptTypes("atomic_reference_prec_recall")
        self.prompt_type1 = properties.PromptTypes(
            "atomic_question_reference_prec_recall"
        )
        self.model = CLAUDE_SONNET_4
        self.api_key = ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def prepare_dataset(self, srcs, tgts):
        pred_questions = []
        ref_questions = []
        pred_qa_pairs = []
        ref_qa_pairs = []

        for i in range(len(srcs)):
            # ------------------------- extract questions and QA pairs from src files
            src_qa_pairs = srcs.iloc[i]["retrieved_qa_pairs"]

            src_q_evidence = []
            for _qa_pair in src_qa_pairs:
                _ques = _qa_pair["question"]
                if _ques:
                    src_q_evidence.append(_ques)

            pred_questions.append(
                self.properties.AveritecEntry(
                    claim=srcs.iloc[i]["claim"],
                    label=srcs.iloc[i]["predicted_label"],
                    evidence=" ".join(src_q_evidence),
                    id=srcs.iloc[i]["ClaimID"] or i,  # Use index if id is not available
                )
            )
            pred_qa_pairs.append(
                self.properties.AveritecEntry(
                    claim=srcs.iloc[i]["claim"],
                    label=srcs.iloc[i]["predicted_label"],
                    evidence=src_qa_pairs,
                    id=srcs.iloc[i]["ClaimID"] or i,  # Use index if id is not available
                )
            )

            # ------------------------- extract questions and QA pairs from tgt files
            tgt_qa_pairs = tgts.iloc[i]["qa_pairs"]

            tgt_q_evidence = []
            for _qa_pair in tgt_qa_pairs:
                _ques = _qa_pair["question"]
                if _ques:
                    tgt_q_evidence.append(_ques)

            ref_questions.append(
                self.properties.AveritecEntry(
                    claim=tgts.iloc[i]["claim"],
                    label=tgts.iloc[i]["normalized_label"],
                    evidence=" ".join(tgt_q_evidence),
                    id=tgts.iloc[i]["ClaimID"] or i,  # Use index if id is not available
                )
            )
            ref_qa_pairs.append(
                self.properties.AveritecEntry(
                    claim=tgts.iloc[i]["claim"],
                    label=tgts.iloc[i]["normalized_label"],
                    evidence=tgt_qa_pairs,
                    id=tgts.iloc[i]["ClaimID"] or i,  # Use index if id is not available
                )
            )

        return pred_questions, ref_questions, pred_qa_pairs, ref_qa_pairs

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

    def query_claude_sonnet_api(self, prompt):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
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

    def prepare_prompt(self, tgt_sample, pred_sample, input_type):
        """Formats prompt using dataset sample as input."""
        if input_type == "qa_pair":
            prompt = self.properties.PROMPT_MAPPING[self.prompt_type].format(
                tgt_sample.claim, tgt_sample.evidence, pred_sample.evidence
            )
        if input_type == "question":
            prompt = self.properties.PROMPT_MAPPING[self.prompt_type1].format(
                tgt_sample.claim, tgt_sample.evidence, pred_sample.evidence
            )
        return prompt

    def get_response_text(self, response_text):
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

    def process_output(self, sample, response):
        logprob_inp = None
        return self.properties.OpenAIResponse(
            claim=sample.claim,
            evidence=sample.evidence,
            response=response,  # self.get_response_text(response),
            gold=sample.label.lower(),
            id=sample.id,
            logprobs=logprob_inp,
        )

    def calculate_question_score_prec_recall_claude_response(self, response_llm):
        response_openai_copy = copy.deepcopy(response_llm)
        try:
            if isinstance(response_llm.response, str):
                response = json.loads(response_llm.response)
            else:
                response = response_llm.response

            response_openai_copy.response = response
            response_openai_copy.response["precision"] = (
                response["support predicted questions"]
                / response["facts count predicted questions"]
            )
            response_openai_copy.response["recall"] = (
                response["support reference questions"]
                / response["facts count reference questions"]
            )
        except Exception as e:
            print("Following exception occurred: {}".format(e))
            return None

        return response_openai_copy

    def calculate_atomic_score_prec_recall_openai_response(self, response_llm):
        response_openai_copy = copy.deepcopy(response_llm)
        try:
            if isinstance(response_llm.response, str):
                response = json.loads(response_llm.response)
            else:
                response = response_llm.response

            response_openai_copy.response = response
            response_openai_copy.response["precision"] = (
                response["support predicted evidence"]
                / response["facts count predicted evidence"]
            )
            response_openai_copy.response["recall"] = (
                response["support reference evidence"]
                / response["facts count reference evidence"]
            )
        except Exception as e:
            print("Following exception occurred: {}".format(e))
            return None

        return response_openai_copy

    def calculate_question_scores(self, responses):
        predictions_q_scores = []

        for i, res in enumerate(responses):
            pred_q_scores = self.calculate_question_score_prec_recall_claude_response(
                res
            )
            # if pred_w_scores:
            predictions_q_scores.append(pred_q_scores)

        return predictions_q_scores

    def calculate_prediction_scores(self, responses):
        predictions_w_scores = []

        for i, res in enumerate(responses):
            pred_w_scores = self.calculate_atomic_score_prec_recall_openai_response(res)
            # if pred_w_scores:
            predictions_w_scores.append(pred_w_scores)

        return predictions_w_scores

    def prompt_api_model(self, srcs, tgts, input_type):
        responses = []

        for i, tgt_sample in tqdm.tqdm(enumerate(tgts), total=len(tgts)):
            pred_sample = srcs[i]
            #
            prompt = self.prepare_prompt(tgt_sample, pred_sample, input_type)
            #
            attempt = 0
            while attempt < self.MAX_RETRIES:
                try:
                    response = self.query_claude_sonnet_api(prompt)
                    responses.append(self.process_output(tgt_sample, response))
                    # print("✓ One request successfully processed.")
                    break
                except Exception as e:
                    attempt += 1
                    wait_time = 5  # **attempt  # Exponential backoff
                    print(
                        f"Request timed out due to error: {e}. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

        return responses

    def extract_ev2r_score(self, srcs, tgts, qa_evi_scores):
        scores = []
        ev2r_qa_recall = []

        for i in tqdm.tqdm(range(len(srcs))):
            this_example_scores = [0.0 for _ in self.ev2r_reporting_levels]

            for _, ev2r_score in enumerate(qa_evi_scores):
                if ev2r_score and ev2r_score.id == i:
                    _, recall = (
                        ev2r_score.response["precision"],
                        ev2r_score.response["recall"],
                    )

                    for j, level in enumerate(self.ev2r_reporting_levels):
                        if recall > level:
                            this_example_scores[j] = (
                                1.0
                                if srcs.iloc[i]["predicted_label"]
                                == tgts.iloc[i]["normalized_label"]
                                else 0.0
                            )

                    scores.append(this_example_scores)
                    ev2r_qa_recall.append(recall)
                    break

                if ev2r_score and ev2r_score.id > i:
                    break

            if len(scores) != (i + 1):
                scores.append(this_example_scores)
                ev2r_qa_recall.append(0.0)

        return (
            np.mean(np.array(scores), axis=0),
            scores,
            np.mean(np.array(ev2r_qa_recall), axis=0),
            ev2r_qa_recall,
        )

    def extract_recall_score(self, evi_scores):
        evi_recall = []

        for score in evi_scores:
            if score:
                _, recall = (
                    score.response["precision"],
                    score.response["recall"],
                )
                evi_recall.append(recall)
            else:
                evi_recall.append(0.0)

        return np.mean(np.array(evi_recall), axis=0), evi_recall
