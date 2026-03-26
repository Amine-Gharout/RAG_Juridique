from __future__ import annotations

import importlib
from typing import Iterable

from .config import Settings
from .models import Candidate


class GroqLegalAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _get_client(self):
        if not self.settings.groq_api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Set the environment variable before running the chat pipeline."
            )

        try:
            groq_module = importlib.import_module("groq")
            Groq = getattr(groq_module, "Groq")
        except ImportError as exc:
            raise RuntimeError(
                "Groq SDK is not installed. Install dependencies from requirements.txt."
            ) from exc

        return Groq(api_key=self.settings.groq_api_key)

    def rewrite_query_for_retrieval(self, query: str) -> str:
        if not self.settings.groq_api_key:
            return query

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.settings.groq_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user query into a concise legal retrieval query in French. "
                            "Keep legal keywords, article hints, infractions, sanctions, and entities. "
                            "Return only one short line, no explanation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten or query
        except Exception:
            return query

    def _format_context(self, candidates: Iterable[Candidate]) -> str:
        chunks: list[str] = []
        for idx, item in enumerate(candidates, start=1):
            text = item["text"]
            if len(text) > 900:
                text = text[:900].rstrip() + "..."
            chunks.append(
                "\n".join(
                    [
                        f"[{idx}] Art. {item['article_number']} | tier={item['tier']} | score={item['fused_score']}",
                        text,
                    ]
                )
            )
        return "\n\n".join(chunks)

    def generate_answer(
        self,
        query: str,
        candidates: list[Candidate],
        conversation_history: list[dict[str, str]],
    ) -> str:
        client = self._get_client()

        system_prompt = (
            "You are a legal RAG assistant for Algerian penal code data. "
            "Answer only from the provided sources. "
            "Never invent article numbers or legal facts. "
            "If the context is insufficient, say so clearly and ask for clarification. "
            "Always cite legal references in this exact format: Art. <number>."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}]
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        context = self._format_context(candidates)
        user_prompt = (
            f"User query:\n{query}\n\n"
            "Retrieved sources:\n"
            f"{context}\n\n"
            "Instructions:\n"
            "1) Use only these sources.\n"
            "2) Provide a concise legal answer.\n"
            "3) Cite one or more relevant articles in format Art. X.\n"
            "4) If some sources are weak, mention uncertainty."
        )
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=self.settings.temperature,
        )
        return response.choices[0].message.content or ""
