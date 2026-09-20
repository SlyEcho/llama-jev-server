# llama-jev-server

This is kind of a hack to see if local models can also give decision probabilities from their logits.

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
