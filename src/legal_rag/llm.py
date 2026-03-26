from __future__ import annotations

import importlib
from typing import Iterable, Protocol

from .config import Settings
from .models import Candidate


class LegalAssistantInterface(Protocol):
    def rewrite_query_for_retrieval(self, query: str) -> str:
        ...

    def generate_answer(
        self,
        query: str,
        candidates: list[Candidate],
        conversation_history: list[dict[str, str]],
    ) -> str:
        ...


def format_context(candidates: Iterable[Candidate]) -> str:
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


def get_system_prompt() -> str:
    return (
        "You are a legal RAG assistant for Algerian penal code data. "
        "You MUST converse and provide all answers exclusively in French. "
        "Answer only from the provided sources. "
        "Never invent article numbers or legal facts. "
        "Always cite legal references in this exact format: Art. <number>."
    )


def get_user_prompt(query: str, context: str) -> str:
    return (
        f"User query:\n{query}\n\n"
        "Retrieved sources:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "1) Use only these sources.\n"
        "2) Provide a concise legal answer IN FRENCH.\n"
        "3) Cite one or more relevant articles in format Art. X.\n"
        "4) If some sources are weak, mention uncertainty."
    )


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
                "Groq SDK is not installed. Install dependencies from requirements-rag.txt."
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

    def generate_answer(
        self,
        query: str,
        candidates: list[Candidate],
        conversation_history: list[dict[str, str]],
    ) -> str:
        client = self._get_client()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": get_system_prompt()}]
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        context = format_context(candidates)
        messages.append(
            {"role": "user", "content": get_user_prompt(query, context)})

        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=self.settings.temperature,
        )
        return response.choices[0].message.content or ""


class GeminiLegalAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set the environment variable before running the chat pipeline."
            )

        try:
            self.genai = importlib.import_module("google.generativeai")
            self.genai.configure(api_key=self.settings.gemini_api_key)
        except ImportError as exc:
            raise RuntimeError(
                "Google Generative AI SDK is not installed. Append 'google-generativeai' to requirements-rag.txt."
            ) from exc

    def rewrite_query_for_retrieval(self, query: str) -> str:
        try:
            model = self.genai.GenerativeModel(self.settings.gemini_model)
            prompt = (
                "Rewrite the user query into a concise legal retrieval query in French. "
                "Keep legal keywords, article hints, infractions, sanctions, and entities. "
                "Return only one short line, no explanation.\n\n"
                f"Query: {query}"
            )
            response = model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=0.0)
            )
            rewritten = (response.text or "").strip()
            return rewritten or query
        except Exception:
            return query

    def generate_answer(
        self,
        query: str,
        candidates: list[Candidate],
        conversation_history: list[dict[str, str]],
    ) -> str:
        model = self.genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            system_instruction=get_system_prompt(),
        )

        history_messages = []
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                gemini_role = "model" if role == "assistant" else "user"
                history_messages.append(
                    {"role": gemini_role, "parts": [content]})

        context = format_context(candidates)
        user_prompt_text = get_user_prompt(query, context)
        history_messages.append({"role": "user", "parts": [user_prompt_text]})

        response = model.generate_content(
            history_messages,
            generation_config=self.genai.types.GenerationConfig(
                temperature=self.settings.temperature
            )
        )
        return response.text or ""


def get_llm_assistant(settings: Settings) -> LegalAssistantInterface:
    if settings.llm_provider == "gemini":
        return GeminiLegalAssistant(settings)
    return GroqLegalAssistant(settings)
