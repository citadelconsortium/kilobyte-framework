import tempfile
import unittest
from pathlib import Path

from kilobyte.brains import GGUF_MAGIC, BrainError, BrainManager

TINY = 64  # bytes; real minimum is 100 MiB, but tests must not write that


def _fake_gguf(path: Path, filler: bytes = b"\0", size: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(GGUF_MAGIC)
        handle.write(filler * (size - len(GGUF_MAGIC)))
    return path


class BrainValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = BrainManager(Path(self.tmp.name) / "models", min_bytes=TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def test_too_small_is_rejected(self):
        small = Path(self.tmp.name) / "small.gguf"
        small.write_bytes(GGUF_MAGIC + b"\0" * 10)  # below TINY
        with self.assertRaises(BrainError):
            self.mgr.validate(small)

    def test_wrong_magic_is_rejected(self):
        bad = Path(self.tmp.name) / "bad.gguf"
        bad.write_bytes(b"NOPE" + b"\0" * 200)
        with self.assertRaises(BrainError):
            self.mgr.validate(bad)

    def test_checksum_mismatch_is_rejected(self):
        good = _fake_gguf(Path(self.tmp.name) / "good.gguf")
        with self.assertRaises(BrainError):
            self.mgr.validate(good, expected_sha256="0" * 64)


class BrainLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = BrainManager(Path(self.tmp.name) / "models", min_bytes=TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def test_promotion_preserves_the_outgoing_brain_for_rollback(self):
        # A brain is already current.
        _fake_gguf(self.mgr.slot_path("current"), filler=b"A")
        current_hash = self.mgr.info("current", with_hash=True).sha256

        # A new candidate is staged and promoted.
        source = _fake_gguf(Path(self.tmp.name) / "new.gguf", filler=b"B")
        self.mgr.stage_candidate(source)
        promoted = self.mgr.promote()

        # Current is now the candidate; the old current is preserved as previous.
        self.assertTrue(promoted.exists)
        self.assertEqual(self.mgr.info("previous", with_hash=True).sha256, current_hash)
        # The candidate slot is cleared once promoted.
        self.assertFalse(self.mgr.info("candidate").exists)

    def test_rollback_restores_the_previous_brain(self):
        _fake_gguf(self.mgr.slot_path("current"), filler=b"A")
        good_hash = self.mgr.info("current", with_hash=True).sha256

        source = _fake_gguf(Path(self.tmp.name) / "new.gguf", filler=b"B")
        self.mgr.stage_candidate(source)
        self.mgr.promote()

        restored = self.mgr.rollback()
        self.assertEqual(restored.sha256, good_hash)

    def test_rollback_without_a_previous_is_refused(self):
        with self.assertRaises(BrainError):
            self.mgr.rollback()

    def test_staging_never_touches_current(self):
        _fake_gguf(self.mgr.slot_path("current"), filler=b"A")
        before = self.mgr.info("current", with_hash=True).sha256
        source = _fake_gguf(Path(self.tmp.name) / "cand.gguf", filler=b"B")
        self.mgr.stage_candidate(source)
        self.assertEqual(self.mgr.info("current", with_hash=True).sha256, before)


if __name__ == "__main__":
    unittest.main()


class BrainVersioningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = BrainManager(Path(self.tmp.name) / "models", min_bytes=TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def test_promotion_records_a_version_entry(self):
        source = _fake_gguf(Path(self.tmp.name) / "b.gguf", filler=b"B")
        self.mgr.stage_candidate(source)
        self.mgr.promote(brain_version="1.0", framework_version="0.1.0")
        history = self.mgr.versions()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["event"], "promote")
        self.assertEqual(history[0]["brain_version"], "1.0")
        self.assertEqual(self.mgr.current_version(), "1.0")

    def test_history_is_append_only_across_promotions(self):
        for version, filler in (("1.0", b"A"), ("1.1", b"B"), ("1.2", b"C")):
            src = _fake_gguf(Path(self.tmp.name) / f"{version}.gguf", filler=filler)
            self.mgr.stage_candidate(src)
            self.mgr.promote(brain_version=version)
        events = [e["brain_version"] for e in self.mgr.versions()]
        self.assertEqual(events, ["1.0", "1.1", "1.2"])
        self.assertEqual(self.mgr.current_version(), "1.2")
