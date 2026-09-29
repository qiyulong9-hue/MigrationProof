import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.run_store import create_run


class RunStoreTest(unittest.TestCase):
    def test_existing_run_is_preserved_and_new_suffix_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 9, 14, 15, 0, 0, tzinfo=timezone.utc)
            first = create_run(root, "task", "probe", now)
            marker = first / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            second = create_run(root, "task", "probe", now)
            self.assertNotEqual(first, second)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
