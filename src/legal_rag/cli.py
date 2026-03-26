from __future__ import annotations

import argparse

from .graph import LegalRAGApp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LangGraph + Groq legal RAG chat")
    parser.add_argument("--query", type=str, help="Single query mode.")
    parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow tier C retrieval when confidence is low.",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Show routing metadata and warnings.",
    )
    return parser


def _print_result(result: dict[str, object], show_debug: bool) -> None:
    print("\n=== Answer ===")
    print(result.get("final_answer", ""))

    if show_debug:
        print("\n=== Debug ===")
        print(f"Confidence level: {result.get('confidence_level', 'UNKNOWN')}")
        print(f"Tier path: {result.get('tier_path', [])}")
        warnings = result.get("warnings", [])
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"- {warning}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    app = LegalRAGApp()

    if args.query:
        result = app.ask(
            query=args.query,
            allow_low_confidence=args.allow_low_confidence,
        )
        _print_result(result, args.show_debug)
        return

    print("Legal RAG interactive chat. Type 'exit' to stop.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = app.ask(
            query=query,
            allow_low_confidence=args.allow_low_confidence,
        )
        _print_result(result, args.show_debug)


if __name__ == "__main__":
    main()
