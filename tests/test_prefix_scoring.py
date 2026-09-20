import math
import unittest
from unittest.mock import AsyncMock

from llama_jev_server import distinguishing_prefixes, grammar_for, score_criteria


class PrefixTests(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(
            distinguishing_prefixes([[1, 2, 3, 10], [1, 2, 4, 10], [5, 6, 10]]),
            [[1, 2, 3], [1, 2, 4], [5]],
        )

    def test_newline_distinguishes_shorter_label(self):
        self.assertEqual(
            distinguishing_prefixes([[1, 10], [1, 2, 10]]),
            [[1, 10], [1, 2]],
        )

    def test_duplicates_share_prefix(self):
        self.assertEqual(
            distinguishing_prefixes([[1, 2, 10], [1, 2, 10], [1, 3, 10]]),
            [[1, 2], [1, 2], [1, 3]],
        )

    def test_single_label_still_scores_one_token(self):
        self.assertEqual(distinguishing_prefixes([[1, 2, 10]]), [[1]])

    def test_embedded_newline_prefix_falls_back_to_complete_sequence(self):
        self.assertEqual(
            distinguishing_prefixes([[1, 10], [1, 10, 2, 10]]),
            [[1, 10], [1, 10, 2]],
        )

    def test_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            distinguishing_prefixes([[]])

    def test_numeric_grammar(self):
        self.assertEqual(grammar_for([123, 456]), "root ::= <[123]> <[456]>")


class PrefixScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_selected_prefixes_and_reuses_duplicates(self):
        client = AsyncMock()
        client.apply_template.return_value = "prompt"
        client.tokenize.side_effect = [[1, 2, 10], [1, 3, 10]]
        client.completion.side_effect = [
            {"tokens": [1, 2], "completion_probabilities": [
                {"id": 1, "logprob": -0.5}, {"id": 2, "logprob": -0.25},
            ]},
            {"tokens": [1, 3], "completion_probabilities": [
                {"id": 1, "logprob": -0.5}, {"id": 3, "logprob": -2.0},
            ]},
        ]
        result = await score_criteria(client, "context", "question", ["Red", "Red apple", "Red"])
        self.assertEqual(result, [-0.75, -2.5, -0.75])
        self.assertEqual([call.args for call in client.tokenize.await_args_list], [("Red\n",), ("Red apple\n",)])
        self.assertEqual(client.completion.await_count, 2)
        for call, tokens in zip(client.completion.await_args_list, [[1, 2], [1, 3]]):
            self.assertEqual(call.args, ("prompt",))
            self.assertEqual(call.kwargs["n_predict"], len(tokens))
            self.assertEqual(call.kwargs["grammar"], grammar_for(tokens))
            self.assertIs(call.kwargs["post_sampling_probs"], False)
            self.assertIs(call.kwargs["return_tokens"], True)
        self.assertIn("followed by a newline", client.apply_template.await_args.args[0])

    async def test_incomplete_or_invalid_generation_is_rejected(self):
        results = [
            {"tokens": [], "completion_probabilities": []},
            {"tokens": [2], "completion_probabilities": [{"id": 2, "logprob": -1}]},
            {"tokens": [1], "completion_probabilities": []},
            {"tokens": [1], "completion_probabilities": [{"id": 2, "logprob": -1}]},
            {"tokens": [1], "completion_probabilities": [{"id": 1, "logprob": math.nan}]},
            {"tokens": [1], "completion_probabilities": [{"id": 1, "logprob": 0.1}]},
        ]
        for result in results:
            with self.subTest(result=result):
                client = AsyncMock()
                client.tokenize.return_value = [1, 10]
                client.completion.return_value = result
                with self.assertRaises(ValueError):
                    await score_criteria(client, "context", "question", ["Red"])
