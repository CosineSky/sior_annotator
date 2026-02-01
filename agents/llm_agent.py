import json
import requests
from configs.semantic_map import SEMANTIC_MAP


class LLMAgent:
    """
    Real LLM Agent:
    - Judge if a given mask is valid.
    - Provide a semantic type.
    """

    SEMANTIC_CANDIDATES = list(SEMANTIC_MAP.keys())

    def __init__(self, endpoint="", api_key=""):
        self.endpoint = endpoint
        self.api_key = api_key

    def judge(self, image, mask):
        """
        Args:
            image: Original image (path or base64)
            mask: Mask image (path or base64)
        Returns:
            dict: {
                "decision": "keep" / "discard",
                "semantic": "road" / "forest" / ...,
                "confidence": float,
                "reason": str
            }
        """
        prompt = f"""
            You are a remote sensing image analysis expert. Given an image and its mask:
            - Image: {image}
            - Mask: {mask}
            
            Please determine if the mask is valid ("keep") or invalid ("discard"),
            identify the most likely semantic type (choose from: {', '.join(self.SEMANTIC_CANDIDATES)}),
            provide a confidence score between 0 and 1,
            and explain your reasoning in one sentence.
            
            Return your answer in JSON format:
            {{ "decision": "...", "semantic": "...", "confidence": 0.0, "reason": "..." }}
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "You are a strict visual quality control and semantic classification assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        response = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM output is not valid JSON: {content}")

        return {
            "decision": parsed.get("decision", "discard"),
            "semantic": parsed.get("semantic", "background"),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": parsed.get("reason", "")
        }
