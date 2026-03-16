import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class ClaimExtractor:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)

    def extract_claims(self, text: str):

        prompt = f"""
Extract factual claims from the following text.

Rules:
- Each claim must contain ONE fact.
- Split sentences containing multiple facts.
- Ignore opinions.
- Return ONLY JSON.

Format:
{{
 "claims": ["claim1", "claim2"]
}}

Text:
{text}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract factual claims."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)
            return data.get("claims", [])
        except Exception:
            return [content]