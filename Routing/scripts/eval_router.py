"""Run the labeled tickets through the real router and report how routing performs.

Uses the real Groq API (costs calls), so it is NOT part of pytest.

    python -m scripts.eval_router                    # full run (classifier + handlers)
    python -m scripts.eval_router --classify-only    # skip handler calls: cheaper, faster
    python -m scripts.eval_router --out results.json # also save per-ticket results

Final route accuracy is what the customer experiences (after the guard and
threshold). Classifier accuracy only covers tickets the guard let through, to
show how the LLM label itself performs.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import ProviderError
from app.providers.groq_provider import GroqJSONProvider
from app.schemas import LLMAnswer, Route
from app.services.router import route_ticket

TICKETS_FILE = Path(__file__).parents[1] / "tests" / "data" / "tickets.jsonl"
ROUTES = list(Route)
SHORT = {
    Route.ACCOUNT_ACCESS: "acct",
    Route.TROUBLESHOOTING: "trbl",
    Route.HOW_TO: "how",
    Route.HUMAN_ESCALATION: "human",
}


class ClassifyOnlyProvider:
    """Real classifier calls, stubbed handler calls (for --classify-only)."""

    def __init__(self, inner: GroqJSONProvider) -> None:
        self._inner = inner

    def complete_json(self, *, response_model: type[BaseModel], **kwargs: Any) -> BaseModel:
        if response_model is LLMAnswer:
            return LLMAnswer(summary="(skipped)", steps=["(skipped)"], follow_up="(skipped)")
        return self._inner.complete_json(response_model=response_model, **kwargs)


class RetryCounter(logging.Handler):
    """Counts the provider's retry warnings by reason, so retries show up per ticket."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.reasons: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("provider_retry"):
            reason = message.split("reason=", 1)[1].split()[0]
            self.reasons[reason] += 1

    @property
    def count(self) -> int:
        return sum(self.reasons.values())


def load_tickets() -> list[dict]:
    lines = TICKETS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run(classify_only: bool) -> list[dict]:
    settings = Settings()
    provider: Any = GroqJSONProvider(settings)
    if classify_only:
        provider = ClassifyOnlyProvider(provider)

    retries = RetryCounter()
    provider_logger = logging.getLogger("app.providers.groq_provider")
    provider_logger.setLevel(logging.WARNING)  # root is ERROR; warnings must still reach us
    provider_logger.propagate = False  # count them, don't print them
    provider_logger.addHandler(retries)

    results = []
    for ticket in load_tickets():
        retries.reasons.clear()
        start = time.perf_counter()
        row: dict[str, Any] = {
            "id": ticket["id"],
            "expected": ticket["expected_route"],
            "note": ticket.get("note"),
        }
        try:
            response = route_ticket(provider, settings, ticket["message"])
            row |= {
                "route": response.route.value,
                "decided_by": response.decided_by,
                "classified_route": (
                    response.classified_route.value if response.classified_route else None
                ),
                "confidence": response.confidence,
                "error": None,
            }
        except ProviderError as exc:
            row |= {
                "route": None,
                "decided_by": None,
                "classified_route": None,
                "confidence": None,
                "error": type(exc).__name__,
            }
        row["ms"] = round((time.perf_counter() - start) * 1000)
        row["retries"] = retries.count
        row["retry_reasons"] = dict(retries.reasons)
        row["ok"] = row["route"] == row["expected"]
        results.append(row)

        mark = "OK " if row["ok"] else "BAD"
        conf = f"{row['confidence']:.2f}" if row["confidence"] is not None else "  - "
        got = row["route"] or f"ERROR {row['error']}"
        print(
            f"{mark} {row['id']}  expected={row['expected']:<16} got={got:<16} "
            f"by={row['decided_by'] or '-':<10} conf={conf} {row['ms']:>5}ms "
            f"retries={row['retries']}"
        )
    return results


def report(results: list[dict], classify_only: bool) -> None:
    n = len(results)
    ok = sum(r["ok"] for r in results)
    print("\n" + "=" * 72)
    print(f"Mode: {'classify-only' if classify_only else 'full'}   Tickets: {n}")
    print(f"Final route accuracy: {ok}/{n} = {ok / n:.0%}")

    classified = [r for r in results if r["classified_route"] is not None]
    if classified:
        c_ok = sum(r["classified_route"] == r["expected"] for r in classified)
        print(
            f"Classifier accuracy (tickets past the guard): "
            f"{c_ok}/{len(classified)} = {c_ok / len(classified):.0%}"
        )

    # Confusion matrix: rows = expected, columns = final route.
    print("\nConfusion matrix (rows = expected, cols = got)")
    header = "".join(f"{SHORT[r]:>7}" for r in ROUTES) + "    err"
    print(f"{'':>8}{header}")
    for expected in ROUTES:
        cells = Counter(r["route"] for r in results if r["expected"] == expected.value)
        row = "".join(f"{cells.get(got.value, 0):>7}" for got in ROUTES)
        print(f"{SHORT[expected]:>8}{row}{cells.get(None, 0):>7}")

    decided = Counter(r["decided_by"] or "error" for r in results)
    escalated = sum(r["route"] == Route.HUMAN_ESCALATION.value for r in results)
    expected_human = sum(r["expected"] == Route.HUMAN_ESCALATION.value for r in results)
    print(f"\nEscalation rate: {escalated}/{n} = {escalated / n:.0%} (expected {expected_human / n:.0%})")
    print("Decided by: " + ", ".join(f"{k}={v}" for k, v in sorted(decided.items())))

    confidences = [r["confidence"] for r in results if r["confidence"] is not None]
    if confidences:
        print(
            f"Confidence: min={min(confidences):.2f} median={statistics.median(confidences):.2f} "
            f"max={max(confidences):.2f}  distinct values={len(set(confidences))}"
        )

    latencies = [r["ms"] for r in results]
    print(
        f"Latency: mean={statistics.mean(latencies):.0f}ms "
        f"median={statistics.median(latencies):.0f}ms max={max(latencies)}ms"
    )
    total_retries = sum(r["retries"] for r in results)
    with_retry = sum(r["retries"] > 0 for r in results)
    by_reason: Counter[str] = Counter()
    for r in results:
        by_reason.update(r["retry_reasons"])
    reasons = ", ".join(f"{k}={v}" for k, v in by_reason.most_common()) or "none"
    print(
        f"Retries: {total_retries} total, {with_retry}/{n} tickets needed at least one "
        f"({reasons})"
    )
    errors = [r for r in results if r["error"]]
    print(f"Errors: {len(errors)}" + "".join(f"\n  {r['id']}: {r['error']}" for r in errors))

    wrong = [r for r in results if not r["ok"]]
    if wrong:
        print("\nMisrouted:")
        for r in wrong:
            note = f"  ({r['note']})" if r["note"] else ""
            print(
                f"  {r['id']}: expected {r['expected']}, got {r['route']} "
                f"by {r['decided_by']}{note}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classify-only", action="store_true", help="skip handler LLM calls")
    parser.add_argument("--out", type=Path, help="write per-ticket results as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)  # keep the report readable
    results = run(args.classify_only)
    report(results, args.classify_only)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved per-ticket results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
