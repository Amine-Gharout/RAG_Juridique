from __future__ import annotations

from dataclasses import dataclass
import importlib
import re

from .config import Settings, load_settings
from .corpus import load_tiered_corpus
from .llm import GroqLegalAssistant
from .models import Candidate, LegalRAGState
from .retrieval import (
    extract_article_reference,
    merge_candidates,
    normalize_article_number,
    retrieve_for_tier,
)


_CITATION_RE = re.compile(
    r"\b(?:art(?:icle)?\.?)\s*([0-9]+(?:\s*(?:bis|ter|quater))?)\b",
    re.IGNORECASE,
)


@dataclass
class Runtime:
    settings: Settings
    corpus_by_tier: dict[str, list[dict[str, object]]]
    assistant: GroqLegalAssistant


def _extract_citations(answer: str) -> list[str]:
    citations: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        citations.append(normalize_article_number(match.group(1)))
    return citations


def _confidence_level(candidates: list[Candidate]) -> str:
    tiers = {candidate["tier"] for candidate in candidates}
    if "C" in tiers:
        return "LOW"
    if "B" in tiers:
        return "MEDIUM"
    return "HIGH"


def _build_graph(runtime: Runtime):
    langgraph_graph = importlib.import_module("langgraph.graph")
    START = getattr(langgraph_graph, "START")
    END = getattr(langgraph_graph, "END")
    StateGraph = getattr(langgraph_graph, "StateGraph")

    cfg = runtime.settings.retrieval

    def parse_query(state: LegalRAGState) -> dict[str, object]:
        query = state.get("user_query", "")
        query_lower = query.lower()
        article_reference = extract_article_reference(query)
        retrieval_query = query
        if not article_reference:
            retrieval_query = runtime.assistant.rewrite_query_for_retrieval(
                query)

        wants_low = any(
            keyword in query_lower
            for keyword in (
                "tier c",
                "quarantine",
                "low confidence",
                "uncertain",
                "expert mode",
            )
        )
        return {
            "query_metadata": {
                "article_reference": article_reference,
                "wants_low_confidence": wants_low,
                "retrieval_query": retrieval_query,
            }
        }

    def retrieve_candidates(state: LegalRAGState) -> dict[str, object]:
        query_metadata = state["query_metadata"]
        query = query_metadata["retrieval_query"]
        allow_low = bool(state.get("allow_low_confidence", False)
                         or query_metadata["wants_low_confidence"])

        tier_a = retrieve_for_tier(
            query=query,
            rows=runtime.corpus_by_tier["A"],
            tier="A",
            cfg=cfg,
            article_reference=query_metadata["article_reference"],
        )

        selected: list[Candidate] = list(tier_a)
        warnings: list[str] = []

        strongest_a = tier_a[0]["fused_score"] if tier_a else 0.0
        has_exact_in_a = any(item["exact_match_score"] > 0 for item in tier_a)
        needs_b = (not has_exact_in_a) and (
            len(tier_a) < cfg.min_docs_for_confident_answer
            or strongest_a < cfg.min_score_for_no_fallback
        )

        tier_b: list[Candidate] = []
        if needs_b:
            tier_b = retrieve_for_tier(
                query=query,
                rows=runtime.corpus_by_tier["B"],
                tier="B",
                cfg=cfg,
                article_reference=query_metadata["article_reference"],
            )
            if tier_b:
                selected.extend(tier_b)

        tier_c: list[Candidate] = []
        strongest_after_ab = selected[0]["fused_score"] if selected else 0.0
        has_exact_after_ab = any(
            item["exact_match_score"] > 0 for item in selected)
        should_try_c = allow_low and (not has_exact_after_ab) and (
            not selected or strongest_after_ab < cfg.min_score_for_no_fallback
        )
        if should_try_c:
            tier_c = retrieve_for_tier(
                query=query,
                rows=runtime.corpus_by_tier["C"],
                tier="C",
                cfg=cfg,
                article_reference=query_metadata["article_reference"],
            )
            if tier_c:
                selected.extend(tier_c)

        selected = merge_candidates(selected, cfg.final_context_k)
        selected_tiers = {item["tier"] for item in selected}
        tier_path = [tier for tier in (
            "A", "B", "C") if tier in selected_tiers]

        if "B" in selected_tiers:
            warnings.append(
                "Fallback to tier B was needed because tier A confidence was limited."
            )
        if "C" in selected_tiers:
            warnings.append(
                "Tier C was included. Treat the answer as low confidence.")

        needs_clarification = len(selected) == 0

        if needs_clarification:
            warnings.append("No relevant legal context was retrieved.")

        return {
            "tier_a_candidates": tier_a,
            "tier_b_candidates": tier_b,
            "tier_c_candidates": tier_c,
            "selected_candidates": selected,
            "tier_path": tier_path,
            "warnings": warnings,
            "needs_clarification": needs_clarification,
        }

    def route_after_retrieval(state: LegalRAGState) -> str:
        return "clarify" if state.get("needs_clarification", False) else "generate"

    def generate_answer(state: LegalRAGState) -> dict[str, object]:
        selected = state.get("selected_candidates", [])
        conversation_history = state.get("conversation_history", [])
        warnings = list(state.get("warnings", []))

        try:
            draft = runtime.assistant.generate_answer(
                query=state["user_query"],
                candidates=selected,
                conversation_history=conversation_history,
            )
        except RuntimeError as exc:
            draft = (
                "The pipeline retrieved relevant legal context, but Groq generation is not available yet. "
                f"Reason: {exc}"
            )
            warnings.append(
                "Generation fallback activated due to Groq configuration issue.")

        return {
            "draft_answer": draft,
            "warnings": warnings,
        }

    def verify_and_format(state: LegalRAGState) -> dict[str, object]:
        selected = state.get("selected_candidates", [])
        answer = state.get("draft_answer", "").strip()
        warnings = list(state.get("warnings", []))

        available = {
            normalize_article_number(candidate["article_number"]): candidate for candidate in selected
        }
        extracted = _extract_citations(answer)

        cited: list[str] = []
        for item in extracted:
            if item in available and item not in cited:
                cited.append(item)

        if not cited and selected:
            for candidate in selected[:2]:
                normalized = normalize_article_number(
                    candidate["article_number"])
                if normalized not in cited:
                    cited.append(normalized)
            warnings.append(
                "No verifiable citation in model output; attached top retrieved articles.")

        if not answer:
            answer = "No answer was generated. Please retry with a more explicit legal question."

        citation_lines: list[str] = []
        for normalized in cited:
            candidate = available.get(normalized)
            if not candidate:
                continue
            citation_lines.append(
                f"- Art. {candidate['article_number']} (tier {candidate['tier']}, score {candidate['fused_score']})"
            )

        final_answer = answer
        if citation_lines:
            final_answer = f"{answer}\n\nSources:\n" + \
                "\n".join(citation_lines)

        return {
            "cited_articles": [f"Art. {available[item]['article_number']}" for item in cited if item in available],
            "confidence_level": _confidence_level(selected),
            "final_answer": final_answer,
            "warnings": warnings,
        }

    def clarify(state: LegalRAGState) -> dict[str, object]:
        message = (
            "I could not find enough reliable legal context for this question. "
            "Please rephrase with an article reference (for example: Art. 15 bis) "
            "or specify the legal concept you need."
        )
        return {
            "final_answer": message,
            "cited_articles": [],
            "confidence_level": "LOW",
        }

    graph = StateGraph(LegalRAGState)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve_candidates", retrieve_candidates)
    graph.add_node("generate", generate_answer)
    graph.add_node("verify", verify_and_format)
    graph.add_node("clarify", clarify)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "retrieve_candidates")
    graph.add_conditional_edges(
        "retrieve_candidates",
        route_after_retrieval,
        {
            "clarify": "clarify",
            "generate": "generate",
        },
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("clarify", END)

    return graph.compile()


class LegalRAGApp:
    def __init__(self, settings: Settings | None = None) -> None:
        resolved_settings = settings or load_settings()
        self.runtime = Runtime(
            settings=resolved_settings,
            corpus_by_tier=load_tiered_corpus(resolved_settings),
            assistant=GroqLegalAssistant(resolved_settings),
        )
        self.graph = _build_graph(self.runtime)
        self.conversation_history: list[dict[str, str]] = []

    def ask(self, query: str, allow_low_confidence: bool = False) -> LegalRAGState:
        state: LegalRAGState = {
            "user_query": query,
            "allow_low_confidence": allow_low_confidence,
            "conversation_history": self.conversation_history,
        }
        result = self.graph.invoke(state)

        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append(
            {"role": "assistant", "content": str(
                result.get('final_answer', ''))}
        )
        self.conversation_history = self.conversation_history[-12:]

        return result
