import json
from collections import Counter
from pathlib import Path

from app.schemas import Route, SupportRequest

TICKETS_FILE = Path(__file__).parents[1] / "data" / "tickets.jsonl"


def load_tickets() -> list[dict]:
    lines = TICKETS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_ticket_file_loads_and_is_balanced():
    tickets = load_tickets()

    assert len(tickets) == 24
    counts = Counter(Route(t["expected_route"]) for t in tickets)
    assert counts == {route: 6 for route in Route}


def test_tickets_have_unique_ids_and_valid_messages():
    tickets = load_tickets()

    assert len({t["id"] for t in tickets}) == len(tickets)
    for ticket in tickets:
        assert set(ticket) <= {"id", "message", "expected_route", "note"}
        SupportRequest.model_validate(
            {"message": ticket["message"]}, context={"max_message_chars": 5000}
        )
