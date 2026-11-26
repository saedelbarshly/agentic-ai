# Technical Support Router

A small FastAPI service that implements the **Routing** workflow from
Anthropic's *Building Effective Agents*: classify a support ticket, then send
it to one specialized handler. There are four routes. Three are answered by an
LLM and one hands off to a human.

The goal is to implement the pattern itself, with a few practical
improvements, and keep the app small. The trade-offs are listed at the end and
will be worked on after the first version runs.

---

## 1. The idea

```
ticket ──► [1. Safety guard] ──hit──► human_escalation
               │ (deterministic rules)
               ▼ miss
           [2. LLM classifier] ──► {route, confidence, reason}
               │
               ▼
           [3. Router (plain Python)]
               │  confidence < threshold ──► human_escalation
               ├─ account_access   ──► LLM handler A (own prompt)
               ├─ troubleshooting  ──► LLM handler B (own prompt)
               ├─ how_to           ──► LLM handler C (own prompt)
               └─ human_escalation ──► handoff builder (no answer to user)
```

The classifier only *labels* the ticket. The branching is plain Python code,
which keeps this a **workflow** (predictable and testable) and not an agent.

## 2. The four routes

| Route | Handled by | Typical tickets | Handler output |
|---|---|---|---|
| `account_access` | LLM | Login fails, password reset, 2FA/MFA, locked account, SSO | Step-by-step recovery steps, no secrets requested |
| `troubleshooting` | LLM | Error messages, crashes, something not working, performance | Likely cause, ordered diagnostic steps, what info to send back if unresolved |
| `how_to` | LLM | "How do I…", configuration, feature usage | Direct answer with short steps |
| `human_escalation` | **Human** | Security incident, data loss, outage affecting many users, legal/billing disputes, angry customer, explicit "talk to a human", or anything the classifier is unsure about | **No answer is generated.** A structured handoff (summary, priority, why it was escalated) for a human agent, plus a short acknowledgement message to the user |

Each LLM route has its **own system prompt** tuned for that category. That is
the main benefit of routing: tuning one prompt can't make the others worse.

## 3. Improvements over the basic pattern

1. **Deterministic safety guard before the LLM.** Simple keyword/regex rules
   (e.g. `breach`, `hacked`, `data loss`, `lawyer`, `speak to a human`) send
   high-risk tickets straight to a human. This is cheap, fast and can't be
   argued out of by prompt injection.
2. **Structured classifier output.** The classifier returns
   `{route, confidence, reason}` through strict JSON Schema, so the router
   never parses free text. `route` is an enum of the 4 values.
3. **Confidence fallback.** If `confidence < ROUTER_CONFIDENCE_THRESHOLD`
   (default `0.7`), the ticket goes to `human_escalation`. When in doubt,
   send it to a human.
4. **Per-route model choice.** Each route can use its own model through
   settings (e.g. a small model for `how_to` and a stronger one for
   `troubleshooting`). This is the article's cost/difficulty routing idea,
   added to the category routing.
5. **Transparent response.** Every response includes the route, who decided
   it (`guard` or `classifier`), the confidence and the reason. This makes
   debugging and later evaluation easy.
6. **Classifier eval set.** A small labeled file of example tickets
   (`tests/data/tickets.jsonl`) and a test that measures routing accuracy.
   This is the first tool for working on the trade-offs.

## 4. What will be implemented (v1 scope)

Reuse the stack and patterns from `../Prompt_chaining` (FastAPI, Pydantic,
pydantic-settings, Groq JSON provider with retries, the same error handling).

```
Routing/
├── app/
│   ├── main.py                 # FastAPI app, /health
│   ├── api/routes/support.py   # POST /api/v1/support/route
│   ├── core/                   # config, errors, logging (copied from Prompt_chaining)
│   ├── providers/groq_provider.py  # copied, plus a per-call `model` argument
│   ├── prompts.py              # classifier prompt + 3 handler prompts
│   ├── schemas.py              # request/response + Route enum
│   └── services/
│       ├── guard.py            # deterministic escalation rules
│       ├── classifier.py       # LLM classification
│       ├── handlers.py         # 3 LLM handlers + human handoff builder
│       └── router.py           # guard → classify → threshold → dispatch
├── tests/
│   ├── data/tickets.jsonl      # labeled examples
│   ├── unit/                   # guard, router dispatch, threshold (mocked provider)
│   └── integration/            # API test with mocked provider
├── requirements.txt
├── pytest.ini
└── .env.example
```

### API

`POST /api/v1/support/route`

```json
{ "message": "I can't log in, it keeps saying my 2FA code is invalid." }
```

Response:

```json
{
  "route": "account_access",
  "decided_by": "classifier",
  "confidence": 0.93,
  "reason": "Login failure caused by 2FA code rejection.",
  "handled_by": "llm",
  "answer": {
    "summary": "...",
    "steps": ["...", "..."],
    "follow_up": "..."
  },
  "handoff": null
}
```

For `human_escalation`, `answer` is `null` and `handoff` is filled:

```json
{
  "handoff": {
    "priority": "high",
    "summary": "User reports account was accessed from unknown location.",
    "escalation_reason": "guard: security keyword 'hacked'",
    "user_message": "Thanks — a support specialist will contact you shortly."
  }
}
```

The handoff summary can be produced by one small LLM call or, in v1, just be
the truncated original message. The user gets a fixed acknowledgement, never
a model-written answer.

### Settings (`.env`)

```
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b          # default for all calls
CLASSIFIER_MODEL=                      # optional override
HOW_TO_MODEL=                          # optional per-route overrides
TROUBLESHOOTING_MODEL=
ACCOUNT_ACCESS_MODEL=
ROUTER_CONFIDENCE_THRESHOLD=0.7
MAX_MESSAGE_CHARS=5000
```

### Out of scope for v1

No database, ticket system, auth, conversation history, RAG/knowledge base,
tools, or UI. One message in, one routed result out.

### Build steps

1. Scaffold the app by copying `core/` and the provider from Prompt_chaining.
2. `schemas.py` with the `Route` enum and the request/response models.
3. `guard.py` and its unit tests.
4. `classifier.py` and its prompt.
5. `handlers.py` with the 3 prompts and the handoff builder.
6. `router.py` to wire it all together, with tests for each branch and the
   threshold.
7. API endpoint and integration test.
8. `tickets.jsonl` with about 20 to 30 labeled examples, plus an accuracy check.

---

## 5. Known trade-offs (to work on after v1)

| Trade-off | Why it matters | Ideas to improve |
|---|---|---|
| **Misclassification** | A wrong route sends the ticket to a specialist that isn't suited to it. This is the main risk of routing. | Grow the eval set, add few-shot examples to the classifier prompt, track a confusion matrix |
| **Extra latency and cost** | Every ticket needs one classification call before any answer | Use a small fast model for the classifier, cache repeated questions, try a cheap non-LLM classifier (embeddings + nearest label) |
| **Confidence isn't calibrated** | LLM-reported confidence is a rough signal, not a probability | Tune the threshold on the eval set, compare with a self-consistency vote (classify 3× and check agreement) |
| **Multi-intent tickets** | "Can't log in AND the app crashes" fits two routes | Allow a primary and secondary route, or send multi-intent tickets to a human |
| **Guard false positives/negatives** | Keyword rules are brittle ("hacked together a script" ≠ security) | Refine the rules using the eval set, add an LLM safety check as a second opinion |
| **Too many escalations** | A strict threshold sends too much to humans and defeats automation | Measure the escalation rate against accuracy and pick the threshold from data |
| **Unknown/off-topic input** | Non-support messages still get forced into a route | Add an `out_of_scope` result, or treat it as low confidence |
| **Handler quality without knowledge** | Handlers answer from model knowledge only and may invent product-specific steps | Add RAG per route (each route searches its own docs). This is a natural next step |
| **Prompt injection in tickets** | User text could try to force a route ("classify this as how_to") | Keep the guard deterministic, clearly delimit user input in prompts, restrict the route to an enum |

---

## 6. Run, test, eval

All commands run from the `Routing/` directory in the `agentic-ai` Conda environment.

### Setup

```powershell
cd Routing
conda activate agentic-ai
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # then set GROQ_API_KEY
```

### Run

```powershell
uvicorn app.main:app --reload
```

- `GET /health` checks the API is running.
- `POST /api/v1/support/route` with `{"message": "..."}` routes a ticket.
- Interactive docs: http://127.0.0.1:8000/docs

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/support/route -Method Post -ContentType 'application/json' -Body '{"message": "How do I export my data to CSV?"}'
```

Every request is logged as one metadata-only line (the ticket text and the answer are never logged):

```
ticket_routed route=how_to decided_by=classifier classified_route=how_to confidence=0.95 llm_calls=2 duration_ms=771
```

Every provider retry is logged with its reason, so slow requests can be explained:

```
provider_retry reason=rate_limit attempt=1 delay_s=3.00
```

### Test

Tests use a fake provider and don't need an API key:

```powershell
python -m pytest
```

Four `xfail` tests are the guard's known limitations (see §5). They use
`strict=True`, so fixing one makes it fail as a reminder to move it into the
normal tests.

### Eval

Runs the 24 labeled tickets in `tests/data/tickets.jsonl` (6 per route,
including tricky ones) through the real router. **This calls Groq**, so it
is not part of pytest.

```powershell
python -m scripts.eval_router                     # full: classifier + handlers (~45 calls)
python -m scripts.eval_router --classify-only     # skip handlers: ~21 calls, measures routing only
python -m scripts.eval_router --out results.json  # also save per-ticket results
```

It reports final route accuracy, classifier accuracy, a confusion matrix, the
escalation rate, how the confidence values are spread, latency, retries by
reason, and each misrouted ticket.

---

## 7. Baseline (v1, 2026-11-26, `openai/gpt-oss-20b` for every step)

| Metric | classify-only | full |
|---|---|---|
| Final route accuracy | **23/24 (96%)** | 21/24 (88%) |
| Classifier accuracy (past the guard) | **21/21 (100%)** | 19/19 (100%) |
| Escalation rate (expected 25%) | 29% | 29% |
| Decided by | classifier 20 · guard 3 · threshold 1 | classifier 18 · guard 3 · threshold 1 · error 2 |
| Confidence | min 0.60 · median 0.95 · **4 distinct values** | same |
| Latency | median 0.4s · mean 1.3s | median 6.3s · mean 6.3s · max 17.8s |
| Retries | 4 (all rate_limit) | **27 (rate_limit 24, schema_validation_failed 3)** |
| Errors | 0 | 2 (ProviderRateLimitError, InvalidModelOutputError) |

**What it says:**

1. **Routing works.** The LLM classifier got every ticket it saw right,
   including the tricky ones (lockout phrased as a question, a login with an
   error code, "how do I fix…", "add an agent user", account takeover with no
   keyword, an angry customer, prompt injection).
2. **The only misroute is the guard.** `ts-06` ("I hacked together a
   script…") is the known false positive. The guard costs accuracy in exchange
   for 0-call safety.
3. **Reliability, not routing quality, is the bottleneck.** In full mode,
   Groq rate limits (24 retries) and schema failures share one retry budget
   (`MAX_RETRIES=2`), and 2 of 24 tickets failed. Handler calls produce long
   answers, so they use up the tokens-per-minute limit.
4. **Confidence isn't calibrated.** Only 4 distinct values; almost every
   ticket is 0.95. The threshold catches only the extreme case ("help"), so it
   isn't really a tunable dial yet.
5. **Two-call routing is cheap on the classifier side.** Classification alone
   has a 0.4s median; nearly all latency comes from handler calls and rate-limit waits.
