import json
import re

from deepeval.models import DeepEvalBaseLLM

from src.llm import get_llm


class JudgeLLM(DeepEvalBaseLLM):
    """Wraps whatever get_llm() returns so DeepEval can use it as the judge."""

    def __init__(self, provider="groq"):
        self.provider = provider
        self.model = get_llm(provider)

    def load_model(self):
        return self.model

    def _parse(self, text, schema):
        # strip code fences, grab the first {...} block, validate against the schema
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return schema(**json.loads(match.group(0)))

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self.model.invoke(prompt).content
        try:
            return self.model.with_structured_output(schema).invoke(prompt)
        except Exception:
            return self._parse(self.model.invoke(prompt).content, schema)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"{self.provider}-judge"