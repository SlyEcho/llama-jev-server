import math
import unittest
from unittest.mock import AsyncMock, patch

from llama_jev_server import answer_question, softmax


class SoftmaxTests(unittest.TestCase):
    def test_low_log_probabilities(self):
        probabilities = softmax([-1000, -1001])
        self.assertAlmostEqual(probabilities[0], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(math.fsum(probabilities), 1)

    def test_many_equal_choices(self):
        probabilities = softmax([-10] * 150)
        for probability in probabilities:
            self.assertAlmostEqual(probability, 1 / 150)
        self.assertAlmostEqual(math.fsum(probabilities), 1)

    def test_impossible_choice(self):
        self.assertEqual(softmax([-math.inf, -1000]), [0, 1])

    def test_invalid_inputs(self):
        for values in ([], [-math.inf], [math.nan, 0], [math.inf, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                softmax(values)


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_noul_criteria_are_included_in_prompt(self):
        with patch(
            "llama_jev_server.score_criteria",
            new=AsyncMock(return_value=[math.log(0.75), math.log(0.25)]),
        ) as scorer:
            answer = await answer_question(None, "context", {
                "type": "noul", "instructions": "Check",
                "criteria": {"true": "present", "false": "absent"},
            })
        self.assertEqual(answer["type"], "noul")
        self.assertAlmostEqual(answer["noul"], 0.75)
        self.assertEqual(scorer.await_args.args[3], ["Yes", "No"])
        self.assertEqual(scorer.await_args.args[4], "Yes means: present\nNo means: absent")

    async def test_score_is_expected_value(self):
        with patch(
            "llama_jev_server.score_criteria",
            new=AsyncMock(return_value=[math.log(p) for p in (0.2, 0.3, 0.5)]),
        ):
            answer = await answer_question(None, "context", {
                "type": "score", "instructions": "Rate",
                "criteria": ["Low", "Medium", "High"],
            })
        self.assertAlmostEqual(answer["score"], 1.3)
        self.assertEqual(answer["legend"], {"0": "Low", "1": "Medium", "2": "High"})
        self.assertAlmostEqual(answer["confidence"], 0.5)

    async def test_choice_retains_precision(self):
        probabilities = [0.334, 0.333, 0.333]
        with patch(
            "llama_jev_server.score_criteria",
            new=AsyncMock(return_value=[math.log(p) for p in probabilities]),
        ):
            answer = await answer_question(
                None,
                "context",
                {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": None, "b": None, "c": None},
                },
            )
        self.assertEqual(answer["choice"], "a")
        for actual, expected in zip(answer["probabilities"].values(), probabilities):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(answer["confidence"], 0.334)


if __name__ == "__main__":
    unittest.main()
