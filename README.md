# llama-jev-server

This is kind of a hack to see if local models can also give decision probabilities from their logits.

## Example request

Configure the upstream llama server using `.env` (see `.env.example`), then start
this API:

```sh
uv run uvicorn llama_jev_server:app --host 127.0.0.1 --port 8000
```

Send a request with all three question types:

```sh
# Set this to the client-facing key configured on the API, not LLAMA_API_KEY.
export SYSTEMONE_API_KEY='your-api-key'

curl --fail-with-body http://localhost:8000/v1/systemone \
  -H "Authorization: Bearer $SYSTEMONE_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "jev-latest",
  "state": {"observation": "A red apple is on the table."},
  "questions": {
    "apple_present": {
      "type": "noul",
      "instructions": "Is there an apple?",
      "criteria": {
        "true": "An apple is present",
        "false": "No apple is present"
      }
    },
    "apple_color": {
      "type": "choice",
      "instructions": "What color is the apple?",
      "criteria": {"red": "Red", "blue": "Blue", "green": "Green"}
    },
    "description_match": {
      "type": "score",
      "instructions": "How well does 'a red apple' match the observation?",
      "criteria": ["Poor", "Good"]
    }
  }
}
JSON
```

Omit the authorization header if API authentication is disabled. `jev-latest`
(or omitting `model`) selects the model configured by `JEV_MODEL`.

The response contains `answers` under the same question keys and token `usage`:

- `noul`: probability of “Yes”.
- `choice`: selected criterion key, probabilities by key, and confidence.
- `score`: expected zero-based criterion index, a legend, probabilities, and confidence.

## Tests

Run deterministic tests (no llama server required):

```sh
uv run python -m unittest discover -s tests -v
```

Include live integration tests:

```sh
RUN_LLAMA_INTEGRATION=1 uv run python -m unittest discover -s tests -v
```

Live tests use the existing environment / `.env` configuration (`LLAMA_BASE_URL`,
`LLAMA_API_KEY`, `JEV_MODEL`, and `SYSTEMONE_API_KEY` or `TYPESAFE_API_KEY`). They
exercise the FastAPI app in-process and make real HTTP requests to llama.cpp;
there is no need to start uvicorn. They perform inference and fail if the upstream
is unavailable, with a 180-second timeout per API request.

Coverage includes authentication, request validation, model aliases, upstream
errors, client requests and counters, and scoring/probability invariants. Live
checks cover all three question types and structured inputs without asserting
model-specific answers or exact token counts.
