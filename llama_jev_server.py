"""FastAPI server implementing the TypeSafe System One API (POST /v1/systemone)
on top of a local llama.cpp server.

Spec: https://docs.typesafe.ai/concepts/system-one.md and https://docs.typesafe.ai/api

Run with:
    uv run uvicorn llama_jev_server:app --host 0.0.0.0 --port 8000

Environment variables:
    LLAMA_BASE_URL     (default http://localhost:8080)
    LLAMA_API_KEY      (optional, passed through to the llama server)
    JEV_MODEL          (default model when the request omits `model` or uses "jev-latest")
    SYSTEMONE_API_KEY  (enables Bearer-token auth on POST /v1/systemone; "TYPESAFE_API_KEY" also works)

Auth: when SYSTEMONE_API_KEY is set, requests to /v1/systemone must carry
    Authorization: Bearer <SYSTEMONE_API_KEY>
and get a 401 Unauthorized otherwise. / and /health stay open for probing.
"""

from __future__ import annotations

import asyncio
import datetime
import hmac
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

# Load .env from the CWD (if present) so plain `uvicorn llama_jev_server:app` works.
# Existing env vars always win (override=False).
load_dotenv()

logger = logging.getLogger("llama_jev_server")

# Client-facing API key (independent of the upstream key, LLAMA_API_KEY).
API_KEY = os.getenv("SYSTEMONE_API_KEY") or os.getenv("TYPESAFE_API_KEY")
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Llama server client
# ---------------------------------------------------------------------------


class LlamaClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self._counters_lock = asyncio.Lock()
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0

        if base_url is None:
            base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:8080")
        self.base_url = base_url

        if api_key is None:
            api_key = os.getenv("LLAMA_API_KEY")

        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(timeout=3600, base_url=base_url, headers=headers)
        self.model = model

    async def __aenter__(self) -> "LlamaClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def update_counters(
        self, input_tokens: int = 0, cached_tokens: int = 0, output_tokens: int = 0
    ):
        async with self._counters_lock:
            self.cached_tokens = max(self.cached_tokens, cached_tokens)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    async def apply_template(self, system: str = "", user: str = ""):
        if self.model.startswith("gpt-oss-"):
            return (
                f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                f"Knowledge cutoff: 2024-06\n"
                f"Current date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"Reasoning: medium\n\n"
                f"# Valid channels: analysis, commentary, final. Channel must be included for every message."
                f"<|end|><|start|>developer<|message|>{system}<|end|>"
                f"<|start|>user<|message|>{user}<|end|>"
                f"<|start|>assistant<|channel|>final<|message|>"
            )

        resp = await self.client.post(
            "/apply-template",
            json={
                "model": self.model,
                "reasoning_effort": "off",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["prompt"]

    async def completion(self, prompt: str, **kwargs):
        resp = await self.client.post(
            "/completion",
            json={
                "model": self.model,
                "prompt": prompt,
                **kwargs,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        await self.update_counters(
            input_tokens=result["timings"]["prompt_n"],
            cached_tokens=result["timings"]["cache_n"],
            output_tokens=result["timings"]["predicted_n"],
        )
        return result


# ---------------------------------------------------------------------------
# JEV scoring logic (System One semantics)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert assistant that can give a quick answer to a complex "
    "question. Answer with only one of the possible choices."
)


def softmax(logs: list[float]) -> list[float]:
    if not logs or any(math.isnan(x) or x == math.inf for x in logs):
        raise ValueError("Expected nonempty log probabilities without NaN or +inf")
    maximum = max(logs)
    if maximum == -math.inf:
        raise ValueError("At least one log probability must be finite")
    exps = [math.exp(x - maximum) for x in logs]
    exp_sum = math.fsum(exps)
    return [x / exp_sum for x in exps]


def argmax(a: list[float]) -> int:
    idx = 0
    for i in range(1, len(a)):
        if a[i] > a[idx]:
            idx = i
    return idx


def render_value(value: str | dict | list) -> str:
    """Render a (possibly structured) state / instructions / criterion as text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def grammar_for(text: str) -> str:
    """GGRAU/BNF-style llama.cpp grammar that forces exactly one string."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'root ::= "{escaped}"'


async def score_criteria(
    client: LlamaClient,
    state: str,
    instructions: str,
    labels: list[str],
    extra: str = "",
) -> list[float]:
    """Score each label by forcing the model to emit it and reading its logprob."""
    n = len(labels)
    logprobs = [0.0] * n
    resps = [None] * n

    user_prompt = (
        f"## Context:\n\n{state}\n\n### Question:\n\n{instructions}\n"
    )
    if extra:
        user_prompt += f"\n{extra}\n"
    user_prompt += (
        f"\n### Possible answers (1 of {n}):\n\n" + "\n".join(labels) + "\n"
    )

    prompt = await client.apply_template(SYSTEM_PROMPT, user_prompt)

    for i in range(n):
        resps[i] = await client.completion(
            prompt,
            n_probs=1,
            n_predict=1,
            grammar=grammar_for(labels[i]),
            min_keep=4,
            min_p=0,
            top_p=0,
            repeat_penalty=1,
            repeat_last_n=0,
            presence_penalty=0,
            frequency_penalty=0,
            samplers=["min_p", "top_p", "top_k"],
        )

    for i in range(n):
        logprobs[i] = resps[i]["completion_probabilities"][0]["logprob"]

    return logprobs


async def answer_question(client: LlamaClient, state: str, q: dict) -> dict:
    qtype = q["type"]
    instructions = render_value(q["instructions"])
    criteria = q.get("criteria")

    extra = ""
    if qtype == "noul":
        labels = ["Yes", "No"]
        choice_keys = None
        if isinstance(criteria, dict):
            defs = []
            for key, label in (("true", "Yes means"), ("false", "No means")):
                if criteria.get(key) is not None:
                    defs.append(f"{label}: {render_value(criteria[key])}")
            if defs:
                extra = "\n".join(defs)
    elif qtype == "choice":
        choice_keys = list(criteria)
        labels = []
        for key in choice_keys:
            value = criteria[key]
            if value is None:
                labels.append(key)
            elif isinstance(value, str):
                labels.append(value)
            else:
                labels.append(json.dumps(value, ensure_ascii=False))
    else:  # score
        choice_keys = None
        labels = [render_value(item) for item in criteria]

    n = len(labels)
    logprobs = await score_criteria(client, state, instructions, labels, extra)

    s = softmax(logprobs)
    confidence = math.exp(max(logprobs))

    if qtype == "noul":
        return {"type": "noul", "noul": s[0]}
    if qtype == "choice":
        return {
            "type": "choice",
            "choice": choice_keys[argmax(s)],
            "probabilities": {choice_keys[i]: s[i] for i in range(n)},
            "confidence": confidence,
        }
    return {
        "type": "score",
        "score": math.fsum([i * s[i] for i in range(n)]),
        "legend": {str(i): labels[i] for i in range(n)},
        "probabilities": {str(i): s[i] for i in range(n)},
        "confidence": confidence,
    }


async def llama_jev(req: dict) -> dict:
    model = req["model"]
    state = render_value(req["state"])
    questions = req["questions"]

    async with LlamaClient(model) as client:
        answers = {key: await answer_question(client, state, q) for key, q in questions.items()}

    return {
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": client.cached_tokens,
            "output_tokens": client.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# API schema (Typesafe System One)
# ---------------------------------------------------------------------------

# A "text-ish" value: per the spec, state / instructions / criteria entries
# are string | object | array (criteria values may also be null for choice).
Textish = str | dict[str, Any] | list[Any]


class NoulCriteria(BaseModel):
    true: Textish | None = None
    false: Textish | None = None


class BaseQuestion(BaseModel):
    type: str
    instructions: Textish


class NoulQuestion(BaseQuestion):
    type: Literal["noul"]
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseQuestion):
    type: Literal["choice"]
    criteria: dict[str, Textish | None] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseQuestion):
    type: Literal["score"]
    criteria: list[Textish] = Field(min_length=2, max_length=10)


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class NoulAnswer(BaseModel):
    type: str = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: str = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: str = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


# The model used when a request omits `model` (or uses the "jev-latest" alias).
DEFAULT_MODEL = os.getenv("JEV_MODEL", "gpt-oss-20b")


def resolve_model(model: str | None) -> str:
    if model is None or model == "jev-latest":
        return DEFAULT_MODEL
    return model


class SystemOneRequest(BaseModel):
    state: Textish = Field(description="String, JSON object, or array of text values")
    model: str | None = Field(
        default=None,
        description=f'Model to use (optional; defaults to {DEFAULT_MODEL!r}, "jev-latest" alias)',
    )
    questions: dict[str, Question] = Field(min_length=1)


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    usage: Usage


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def _upstream_errors(e: httpx.HTTPStatusError):
    status = e.response.status_code
    if status in (503, 529, 509):
        raise HTTPException(status_code=529, detail="Upstream model server is overloaded") from e
    if status == 429:
        raise HTTPException(status_code=429, detail="Upstream model server rate limit exceeded") from e
    if 400 <= status < 500:
        raise HTTPException(status_code=422, detail=f"Upstream rejected the request: {e.response.text[:500]}") from e
    raise HTTPException(status_code=502, detail=f"Upstream model server error: HTTP {status}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    if API_KEY is None:
        logger.warning(
            "SYSTEMONE_API_KEY is not set - /v1/systemone is UNAUTHENTICATED. "
            "Set SYSTEMONE_API_KEY before exposing this server."
        )
    yield


app = FastAPI(
    title="TypeSafe System One API (local JEV)", version="1.0.0", lifespan=lifespan
)


async def require_api_key(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    if API_KEY is None:
        return
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing API key (Authorization: Bearer <key>)")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/")
async def root():
    return {
        "name": "systemone-local",
        "spec": "https://docs.typesafe.ai/api",
        "default_model": DEFAULT_MODEL,
        "auth_enabled": API_KEY is not None,
        "llama_base_url": os.getenv("LLAMA_BASE_URL", "http://localhost:8080"),
        "endpoints": ["POST /v1/systemone", "GET /health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/systemone", response_model=SystemOneResponse, response_model_by_alias=True)
async def systemone(
    req: SystemOneRequest,
    _auth: Annotated[None, Depends(require_api_key)],
):
    payload = {
        "model": resolve_model(req.model),
        "state": req.state,
        "questions": {k: v.model_dump() for k, v in req.questions.items()},
    }
    try:
        return await llama_jev(payload)
    except httpx.HTTPStatusError as e:
        _upstream_errors(e)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach llama server: {e}") from e
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected llama server response: {e}") from e
