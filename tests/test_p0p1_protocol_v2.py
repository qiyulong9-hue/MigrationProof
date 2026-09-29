import unittest

from migrationproof.p0p1_protocol_v2 import build_sft_records


class MindSpeedDatasetCompatibilityTest(unittest.TestCase):
    def test_text_only_records_have_explicit_empty_images(self):
        records = build_sft_records("train", count=8, seed=42)
        self.assertTrue(all(record["images"] == [] for record in records))


if __name__ == "__main__":
    unittest.main()
