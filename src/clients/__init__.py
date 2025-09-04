import os
import sys
import json
import anthropic

from src.config.settings import ANTHROPIC_API_KEY, CLAUDE_SONNET_4

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class ClaudeSonnet4Client:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("API key must be provided!")
        self.model = CLAUDE_SONNET_4
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(self, prompt: str, max_tokens=3000) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except (json.JSONDecodeError, ValueError) as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response.content[0].text[:500] + "..."
                if "response" in locals() and len(response.content[0].text) > 500
                else response.content[0].text,
            }
        except Exception as e:
            return {"error": f"Failed to generate Q&A pairs: {str(e)}"}

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
