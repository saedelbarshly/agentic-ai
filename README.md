# AI CV & Job Analyzer

A FastAPI MVP that extracts CV evidence, compares it with a job description, and
uses two deterministic gates to route application-ready and gap-review reports.
The workflow is implemented in plain Python; candidate content is not persisted
or included in application logs.

## Requirements

- Python 3.12 (the `agentic-ai` Conda environment is used for this workspace)
- A Groq API key for live analyses

Activate the Conda environment and install dependencies from the existing
requirements file:

```powershell
conda activate agentic-ai
python --version
python -m pip install -r requirements.txt
```

Create a root `.env` from the example and add your Groq API key:

```powershell
Copy-Item .env.example .env
```

The model defaults to `openai/gpt-oss-20b`. `GROQ_MODEL`, request timeout,
retry settings, and input-size limits can be configured through environment
variables. `.env` is ignored by Git.

## Run

```powershell
uvicorn app.main:app --reload
```

- `GET /health` checks that the API is running.
- `POST /api/v1/analyze` accepts JSON with `cv_text` and `job_description`.

Example:

```json
{
  "cv_text": "Python developer. Developed Python APIs.",
  "job_description": "We need a Python developer with FastAPI."
}
```

The response includes the validated candidate profile, job analysis, both gate
results, the selected route, and either an application-ready result or a gap
review. A failed CV gate returns `needs_input` and does not make later model
calls. Required requirements that are not fully matched route to gap review;
preferred requirements alone do not trigger that route.

Groq calls use strict JSON Schema structured outputs where supported by the
selected model. Transient failures are retried with exponential backoff and
`Retry-After` support. Provider failures return safe HTTP errors with a request
ID; provider details and candidate content are not returned to clients.

## Tests

Unit and API integration tests use a mocked provider and do not require a Groq
API key:

```powershell
python -m pytest
```
