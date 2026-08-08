import tempfile
import unittest
from pathlib import Path

from kilobyte.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db", message_limit=3, fact_limit=2)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_separate_sessions_and_bounded_messages(self):
        first = self.store.new_session("terminal")
        second = self.store.new_session("terminal")
        for index in range(4):
            self.store.add_message(first, "user", f"message-{index}")
        self.store.add_message(second, "user", "separate")
        self.assertNotIn("message-0", str(self.store.history(first)))
        self.assertEqual(self.store.history(second)[0]["content"], "separate")
        self.assertLessEqual(self.store.stats()["messages"], 3)

    def test_recall_and_fact_bound(self):
        self.store.remember("prefers concise terminal output", importance=0.9)
        self.store.remember("uses Arch Linux", importance=0.8)
        self.store.remember("temporary low value", importance=0.1)
        self.assertIn("Arch Linux", " ".join(self.store.recall("Which Linux does the user use?")))
        self.assertLessEqual(self.store.stats()["facts"], 2)


if __name__ == "__main__":
    unittest.main()



class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(Path(self.tmp.name) / "m.db")

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_saving_the_same_name_refines_rather_than_duplicates(self):
        self.memory.save_skill("build project", "when asked to build", "run make")
        self.memory.save_skill("build project", "when asked to build or compile", "run make -j2")
        skills = self.memory.list_skills()
        self.assertEqual(len(skills), 1)
        self.assertIn("compile", skills[0]["when_to_use"])

    def test_recall_matches_on_name_or_trigger(self):
        self.memory.save_skill("repair build", "when the build fails", "inspect, patch, rerun")
        self.assertTrue(self.memory.recall_skills("please repair the failing build"))
        self.assertEqual(self.memory.recall_skills("what is the weather"), [])

    def test_reliable_skills_are_preferred(self):
        self.memory.save_skill("flaky deploy", "when deploying", "step")
        self.memory.save_skill("solid deploy", "when deploying", "step")
        for _ in range(5):
            self.memory.record_skill_outcome("flaky deploy", False)
            self.memory.record_skill_outcome("solid deploy", True)
        self.assertEqual(self.memory.recall_skills("deploying now")[0]["name"], "solid deploy")

    def test_growth_is_bounded(self):
        memory = MemoryStore(Path(self.tmp.name) / "bounded.db", skill_limit=5)
        for index in range(20):
            memory.save_skill(f"skill {index}", "when testing", "step")
        self.assertLessEqual(len(memory.list_skills()), 5)
        memory.close()


class SessionBrowsingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(Path(self.tmp.name) / "m.db")

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_list_sessions_returns_nonempty_sessions_newest_first(self):
        a = self.memory.new_session("terminal", "first chat")
        self.memory.add_message(a, "user", "hello")
        b = self.memory.new_session("terminal", "second chat")
        self.memory.add_message(b, "user", "later")
        self.memory.new_session("terminal", "empty")  # no messages -> excluded
        sessions = self.memory.list_sessions()
        self.assertEqual([s["id"] for s in sessions], [b, a])
        self.assertTrue(all(s["messages"] > 0 for s in sessions))

    def test_search_messages_finds_across_sessions(self):
        a = self.memory.new_session("terminal", "a")
        self.memory.add_message(a, "user", "the firewall rule is wrong")
        b = self.memory.new_session("terminal", "b")
        self.memory.add_message(b, "assistant", "I fixed the firewall")
        hits = self.memory.search_messages("firewall")
        self.assertEqual(len(hits), 2)
        self.assertEqual(self.memory.search_messages("nonexistentword"), [])
