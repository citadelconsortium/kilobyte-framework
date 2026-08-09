import unittest

from kilobyte.prompt import SYSTEM_PROMPT


class PromptContractTests(unittest.TestCase):
    def test_operator_steering_and_tone_are_explicit(self):
        prompt = SYSTEM_PROMPT.lower()
        self.assertIn("do not moralise", prompt)
        self.assertIn("do not interrogate, debate", prompt)
        self.assertIn("accept his corrections and steering immediately", prompt)

    def test_inference_route_stays_operator_selected(self):
        prompt = SYSTEM_PROMPT.lower()
        self.assertIn("inference route sir selected", prompt)
        self.assertIn("without his direction", prompt)


if __name__ == "__main__":
    unittest.main()
