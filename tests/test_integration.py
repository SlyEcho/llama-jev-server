"""Opt-in tests: real llama.cpp HTTP calls through the in-process FastAPI app."""

import asyncio
import math
import os
import unittest

import httpx

import llama_jev_server as server


@unittest.skipUnless(os.getenv("RUN_LLAMA_INTEGRATION") == "1", "Set RUN_LLAMA_INTEGRATION=1 to use the configured llama server")
class LiveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        headers = {"Authorization": f"Bearer {server.API_KEY}"} if server.API_KEY else {}
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test", headers=headers,
        )
        self.addAsyncCleanup(self.client.aclose)

    async def ask(self, questions, **extra):
        # ASGITransport does not enforce HTTPX's network timeout; bound the whole call.
        async with asyncio.timeout(180):
            response = await self.client.post("/v1/systemone", json={
                "state": {"observation": "A red apple is on the table."},
                "questions": questions, **extra,
            })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["model"], server.DEFAULT_MODEL)
        self.assertEqual(set(result["answers"]), set(questions))
        for count in result["usage"].values():
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
        self.assertGreater(result["usage"]["output_tokens"], 0)
        return result["answers"]

    def assert_distribution(self, answer, keys):
        probabilities = answer["probabilities"]
        self.assertEqual(set(probabilities), set(keys))
        for probability in probabilities.values():
            self.assertTrue(math.isfinite(probability))
            self.assertGreaterEqual(probability, 0)
            self.assertLessEqual(probability, 1)
        self.assertAlmostEqual(math.fsum(probabilities.values()), 1)
        self.assertGreaterEqual(answer["confidence"], 0)
        self.assertLessEqual(answer["confidence"], 1)

    async def test_all_question_types(self):
        answers = await self.ask({
            "exists": {"type": "noul", "instructions": "Is there an apple?", "criteria": {
                "true": "An apple is present", "false": "No apple is present",
            }},
            "color": {"type": "choice", "instructions": {"task": "What color is the apple?"}, "criteria": {
                "red": "Red", "blue": "Blue", "green": "Green",
            }},
            "match": {"type": "score", "instructions": "How well does 'a red apple' match the observation?", "criteria": ["Poor", "Good"]},
        })
        self.assertEqual(answers["exists"]["type"], "noul")
        self.assertGreaterEqual(answers["exists"]["noul"], 0)
        self.assertLessEqual(answers["exists"]["noul"], 1)
        choice = answers["color"]
        self.assertEqual(choice["type"], "choice")
        self.assert_distribution(choice, ["red", "blue", "green"])
        self.assertEqual(choice["probabilities"][choice["choice"]], max(choice["probabilities"].values()))
        score = answers["match"]
        self.assertEqual(score["type"], "score")
        self.assert_distribution(score, ["0", "1"])
        self.assertEqual(score["legend"], {"0": "Poor", "1": "Good"})
        self.assertAlmostEqual(score["score"], score["probabilities"]["1"])

    async def test_alias_and_single_choice(self):
        answers = await self.ask({
            "only": {"type": "choice", "instructions": ["Select the available answer"], "criteria": {"Yes": None}},
        }, model="jev-latest")
        self.assertEqual(answers["only"]["choice"], "Yes")
        self.assertEqual(answers["only"]["probabilities"], {"Yes": 1.0})
