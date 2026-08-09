import unittest

from kilobyte.profiles import CONVERSATION, ORCHESTRATOR, PROFILES, SECURITY, select


class ProfileSelectionTests(unittest.TestCase):
    def test_explicit_profile_wins(self):
        self.assertEqual(select("anything at all", explicit="security").name, "security")

    def test_keyword_routing(self):
        self.assertEqual(select("research the latest kernel CVEs").name, "research")
        self.assertEqual(select("there's a bug in my build, tests fail").name, "coding")
        self.assertEqual(select("run an nmap recon on the host").name, "security")
        self.assertEqual(select("why did the systemd service fail").name, "systems")

    def test_unclear_request_falls_back_to_orchestrator(self):
        # An unrouted request gets the orchestrator, which reads the goal, applies the right
        # discipline, and drives it to a finished result.
        self.assertEqual(select("tell me a joke").name, "orchestrator")

    def test_every_profile_has_grounding_language(self):
        # Each specialist must push toward evidence, not memory.
        for name, p in PROFILES.items():
            if name == "general":
                continue
            text = p.instructions.lower()
            self.assertTrue(
                any(w in text for w in ("evidence", "verify", "confirm", "tool", "run", "fetch", "source")),
                f"{name} profile lacks grounding language",
            )

    def test_orchestrator_is_the_default(self):
        self.assertIs(select(""), ORCHESTRATOR)

    def test_conversation_agent_teaches_follow_through(self):
        text = CONVERSATION.instructions.lower()
        self.assertIn("finished result", text)
        self.assertIn("never announce an action", text)

    def test_security_agent_is_custom_scoped_and_can_learn(self):
        text = " ".join(SECURITY.instructions.lower().split())
        self.assertIn("exact targets and scope sir explicitly approves", text)
        self.assertIn("no canned hacking playbook", text)
        self.assertIn("save_skill", SECURITY.tools)
        self.assertIn("recall", SECURITY.tools)
        self.assertNotIn("offensive playbook", text)


if __name__ == "__main__":
    unittest.main()
