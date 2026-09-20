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

    async def test_shared_prefixes_and_duplicate_labels(self):
        answers = await self.ask({
            "shared": {
                "type": "choice", "instructions": "Which description matches?",
                "criteria": {"red": "The red apple", "blue": "The blue apple"},
            },
            "prefix": {
                "type": "choice", "instructions": "Choose the most complete description.",
                "criteria": {"short": "Red", "long": "Red apple", "duplicate": "Red apple"},
            },
        })
        self.assert_distribution(answers["shared"], ["red", "blue"])
        self.assert_distribution(answers["prefix"], ["short", "long", "duplicate"])
        self.assertEqual(
            answers["prefix"]["probabilities"]["long"],
            answers["prefix"]["probabilities"]["duplicate"],
        )

    async def test_numeric_grammar_for_newline_terminated_tokens(self):
        async with asyncio.timeout(180), server.LlamaClient(server.DEFAULT_MODEL) as client:
            tokens = await client.tokenize("Red\n")
            prompt = await client.apply_template(server.SYSTEM_PROMPT, "Answer Red.")
            result = await client.completion(
                prompt, grammar=server.grammar_for(tokens), n_predict=len(tokens),
                n_probs=1, post_sampling_probs=False, return_tokens=True,
            )
        self.assertEqual(result["tokens"], tokens)
        self.assertEqual(result["content"], "Red\n")
        probabilities = result["completion_probabilities"]
        self.assertEqual([entry["id"] for entry in probabilities], tokens)
        for entry in probabilities:
            self.assertTrue(math.isfinite(entry["logprob"]))
            self.assertLessEqual(entry["logprob"], 0)

    async def test_alias_and_single_choice(self):
        answers = await self.ask({
            "only": {"type": "choice", "instructions": ["Select the available answer"], "criteria": {"Yes": None}},
        }, model="jev-latest")
        self.assertEqual(answers["only"]["choice"], "Yes")
        self.assertEqual(answers["only"]["probabilities"], {"Yes": 1.0})
