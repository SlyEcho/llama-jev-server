import unittest

import httpx

from llama_jev_server import LlamaClient


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, handler, model="test-model"):
        client = LlamaClient(model, base_url="http://upstream", api_key="upstream-secret")
        await client.client.aclose()
        client.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client.base_url,
            headers={"Authorization": "Bearer upstream-secret"},
        )
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_template_and_completion_requests(self):
        import json

        requests = []

        def handler(request):
            requests.append((request.url.path, json.loads(request.content)))
            self.assertEqual(request.headers["Authorization"], "Bearer upstream-secret")
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": "formatted prompt"})
            return httpx.Response(200, json={
                "timings": {"prompt_n": 12, "cache_n": 8, "predicted_n": 1},
                "completion_probabilities": [{"logprob": -0.5}],
            })

        client = await self.make_client(handler)
        prompt = await client.apply_template("system", "user")
        self.assertEqual(prompt, "formatted prompt")
        result = await client.completion(prompt, n_predict=1, grammar='root ::= "Yes"')
        self.assertEqual(result["completion_probabilities"][0]["logprob"], -0.5)
        self.assertEqual(requests[0], ("/apply-template", {
            "model": "test-model", "reasoning_effort": "off",
            "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        }))
        self.assertEqual(requests[1], ("/completion", {
            "model": "test-model", "prompt": prompt, "n_predict": 1,
            "grammar": 'root ::= "Yes"',
        }))
        self.assertEqual((client.input_tokens, client.cached_tokens, client.output_tokens), (12, 8, 1))

    async def test_gpt_oss_template_is_local(self):
        def handler(request):
            self.fail("GPT-OSS template should not make an HTTP request")

        client = await self.make_client(handler, model="gpt-oss-20b")
        prompt = await client.apply_template("system instructions", "user context")
        self.assertIn("<|start|>developer<|message|>system instructions", prompt)
        self.assertIn("<|start|>user<|message|>user context", prompt)
        self.assertTrue(prompt.endswith("<|start|>assistant<|channel|>final<|message|>"))

    async def test_upstream_http_failure(self):
        client = await self.make_client(lambda request: httpx.Response(503))
        with self.assertRaises(httpx.HTTPStatusError):
            await client.completion("prompt")
        with self.assertRaises(httpx.HTTPStatusError):
            await client.apply_template("system", "user")

    async def test_counters_preserve_cache_maximum(self):
        client = await self.make_client(lambda request: httpx.Response(200))
        await client.update_counters(10, 20, 1)
        await client.update_counters(5, 15, 2)
        self.assertEqual((client.input_tokens, client.cached_tokens, client.output_tokens), (15, 20, 3))
