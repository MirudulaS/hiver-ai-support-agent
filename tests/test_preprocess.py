import unittest

from src.preprocess import reconstruct_components, split_ids


def row(tweet_id, parent="", response=""):
    return {
        "tweet_id": str(tweet_id),
        "in_response_to_tweet_id": parent,
        "response_tweet_id": response,
    }


class ConversationReconstructionTests(unittest.TestCase):
    def test_simple_linear_conversation(self):
        components = reconstruct_components([row(1, response="2"), row(2, "1", "3"), row(3, "2")])
        self.assertEqual(components, {1: {1, 2, 3}})

    def test_branching_conversation(self):
        components = reconstruct_components([row(1, response="2,3"), row(2, "1"), row(3, "1")])
        self.assertEqual(components, {1: {1, 2, 3}})

    def test_missing_parent_remains_observed_component(self):
        components = reconstruct_components([row(2, "999")])
        self.assertEqual(components, {2: {2}})

    def test_multiple_response_ids_are_preserved_for_audit(self):
        self.assertEqual(split_ids("2,3,4"), [2, 3, 4])
        components = reconstruct_components([row(1, response="2,3"), row(2, "1"), row(3, "1")])
        self.assertEqual(components[1], {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
