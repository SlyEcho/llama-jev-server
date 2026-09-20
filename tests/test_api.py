import unittest
from unittest.mock import AsyncMock, patch

import httpx

import llama_jev_server as server


PAYLOAD = {
    "state": "The sky is blue.",
    "questions": {"q": {"type": "noul", "instructions": "Is the sky blue?"}},
}
RESULT = {
    "model": "test-model",
    "answers": {"q": {"type": "noul", "noul": 0.8}},
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.auth = patch.object(server, "API_KEY", "test-secret")
        self.auth.start()
        self.addCleanup(self.auth.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
            headers={"Authorization": "Bearer test-secret"},
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_public_probes(self):
        for path in ("/", "/health"):
            response = await self.client.get(path, headers={"Authorization": "invalid"})
            self.assertEqual(response.status_code, 200)

    async def test_auth_rejections_do_not_call_upstream(self):
        with patch.object(server, "llama_jev", new_callable=AsyncMock) as upstream:
            for header in (None, "Basic test-secret", "Bearer wrong", "Bearer"):
                headers = {} if header is None else {"Authorization": header}
                request = self.client.build_request("POST", "/v1/systemone", json=PAYLOAD)
                request.headers.pop("Authorization", None)
                request.headers.update(headers)
                response = await self.client.send(request)
                self.assertEqual(response.status_code, 401)
            upstream.assert_not_awaited()

    async def test_auth_disabled(self):
        with patch.object(server, "API_KEY", None), patch.object(
            server, "llama_jev", new=AsyncMock(return_value=RESULT)
        ):
            response = await self.client.post("/v1/systemone", json=PAYLOAD)
        self.assertEqual(response.status_code, 200)

    async def test_model_resolution_and_response(self):
        for model in (None, "jev-latest", "custom-model"):
            with self.subTest(model=model), patch.object(
                server, "llama_jev", new=AsyncMock(return_value=RESULT)
            ) as upstream:
                payload = dict(PAYLOAD)
                if model is not None:
                    payload["model"] = model
                response = await self.client.post("/v1/systemone", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), RESULT)
                expected = "custom-model" if model == "custom-model" else server.DEFAULT_MODEL
                self.assertEqual(upstream.await_args.args[0]["model"], expected)

    async def test_invalid_questions(self):
        questions = [
            {},
            {"q": {"type": "unknown", "instructions": "x"}},
            {"q": {"type": "noul"}},
            {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
            {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(256)}}},
            {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}},
            {"q": {"type": "score", "instructions": "x", "criteria": ["x"] * 11}},
        ]
        with patch.object(server, "llama_jev", new_callable=AsyncMock) as upstream:
            for value in questions:
                with self.subTest(questions=value):
                    response = await self.client.post(
                        "/v1/systemone", json={"state": {}, "questions": value}
                    )
                    self.assertEqual(response.status_code, 422)
            upstream.assert_not_awaited()

    async def test_upstream_status_mapping(self):
        for status, expected in ((400, 422), (401, 422), (429, 429), (500, 502), (503, 529), (509, 529), (529, 529)):
            request = httpx.Request("POST", "http://upstream/completion")
            error = httpx.HTTPStatusError(
                "failure", request=request,
                response=httpx.Response(status, request=request, text="upstream failure"),
            )
            with self.subTest(status=status), patch.object(
                server, "llama_jev", new=AsyncMock(side_effect=error)
            ):
                response = await self.client.post("/v1/systemone", json=PAYLOAD)
                self.assertEqual(response.status_code, expected)

    async def test_upstream_transport_and_data_errors(self):
        for error in (httpx.ConnectError("offline"), httpx.ReadTimeout("timeout"), KeyError("timings"), ValueError("invalid probabilities"), TypeError("bad data")):
            with self.subTest(error=error), patch.object(
                server, "llama_jev", new=AsyncMock(side_effect=error)
            ):
                response = await self.client.post("/v1/systemone", json=PAYLOAD)
                self.assertEqual(response.status_code, 502)
