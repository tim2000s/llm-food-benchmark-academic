"""Unit tests for batch_common — the shared batch helpers."""
import json
import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch_common import (
    make_id_mapping,
    decode_custom_id,
    serialise_query_result,
    PROMPT_SHA256,
)
from food_nutrition_benchmark import QueryResult, FoodItemResult


class TestIdMapping(unittest.TestCase):
    def test_round_trip_simple(self):
        pairs = [(1, "apple.jpg"), (2, "banana.png"), (1, "fish.jpeg")]
        records, id_map = make_id_mapping(pairs)
        self.assertEqual(len(records), 3)
        self.assertEqual(len(id_map), 3)
        # Each record has a unique custom_id and the right metadata
        for rec in records:
            cid = rec["custom_id"]
            self.assertIn(cid, id_map)
            it_back, img_back = decode_custom_id(cid, id_map)
            self.assertEqual(it_back, rec["iteration"])
            self.assertEqual(img_back, rec["image_file"])

    def test_handles_filenames_with_special_characters(self):
        # The previous broken Anthropic encoder lost data on these
        problematic = [
            (1, "IMG-20260410-WA0016.jpg"),
            (2, "IMG_20260410_WA0017.jpeg"),
            (3, "MVIMG-test.with.dots.jpg"),
            (4, "file with spaces.png"),
            (5, "CAFE-Ñoño.jpg"),
        ]
        records, id_map = make_id_mapping(problematic)
        for (it, img), rec in zip(problematic, records):
            it_back, img_back = decode_custom_id(rec["custom_id"], id_map)
            self.assertEqual(it_back, it)
            self.assertEqual(img_back, img)  # exact round-trip

    def test_custom_id_format_satisfies_anthropic(self):
        # Anthropic requires ^[a-zA-Z0-9_-]{1,64}$
        import re
        pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        pairs = [(i, f"image_{i}.jpg") for i in range(1, 1000)]
        records, _ = make_id_mapping(pairs)
        for rec in records:
            self.assertRegex(rec["custom_id"], pattern)

    def test_decode_unknown_id_returns_none(self):
        records, id_map = make_id_mapping([(1, "a.jpg")])
        it, img = decode_custom_id("nonexistent_id", id_map)
        self.assertIsNone(it)
        self.assertIsNone(img)

    def test_id_map_serialises_to_json(self):
        # The id_map needs to be JSON-serialisable for state file storage
        pairs = [(i, f"img{i}.jpg") for i in range(10)]
        _, id_map = make_id_mapping(pairs)
        s = json.dumps(id_map)
        loaded = json.loads(s)
        self.assertEqual(len(loaded), 10)
        for cid, entry in loaded.items():
            self.assertIn("iteration", entry)
            self.assertIn("image_file", entry)


class TestSerialiseQueryResult(unittest.TestCase):
    def test_includes_is_batch_field(self):
        qr = QueryResult(
            model="test", provider="test", image_file="x.jpg",
            iteration=1, timestamp="2026-04-11T00:00:00",
            latency_s=1.0, success=True,
        )
        d = serialise_query_result(qr, is_batch=True)
        self.assertTrue(d["is_batch"])
        self.assertEqual(d["model"], "test")

    def test_includes_token_usage(self):
        qr = QueryResult(
            model="test", provider="test", image_file="x.jpg",
            iteration=1, timestamp="2026-04-11T00:00:00",
            latency_s=1.0, success=True,
            input_tokens=1234, output_tokens=567,
        )
        d = serialise_query_result(qr)
        self.assertEqual(d["input_tokens"], 1234)
        self.assertEqual(d["output_tokens"], 567)

    def test_includes_error_class(self):
        qr = QueryResult(
            model="test", provider="test", image_file="x.jpg",
            iteration=1, timestamp="2026-04-11T00:00:00",
            latency_s=0.0, success=False,
            error="HTTP 429", error_class="rate_limit",
        )
        d = serialise_query_result(qr)
        self.assertEqual(d["error_class"], "rate_limit")
        self.assertFalse(d["success"])

    def test_food_items_serialised(self):
        qr = QueryResult(
            model="test", provider="test", image_file="x.jpg",
            iteration=1, timestamp="2026-04-11T00:00:00",
            latency_s=1.0, success=True,
            food_items=[FoodItemResult(name="apple", carbs_per_100=14.0)],
        )
        d = serialise_query_result(qr)
        self.assertEqual(len(d["food_items"]), 1)
        self.assertEqual(d["food_items"][0]["name"], "apple")
        self.assertEqual(d["food_items"][0]["carbs_per_100"], 14.0)


class TestLegacyOpenAIDecoder(unittest.TestCase):
    """The OpenAI runner has a backwards-compatible decoder for state files
    written before the batch_common refactor. The legacy custom_id format is
    `<image>|iter<N>` and the decoder reconstructs an id_map on the fly."""

    def test_round_trip_basic(self):
        from openai_batch_runner import _legacy_id_map_from_custom_ids
        cids = ["IMG_001.jpg|iter1", "MVIMG-test.with.dots.jpg|iter42"]
        id_map = _legacy_id_map_from_custom_ids(cids)
        self.assertEqual(id_map["IMG_001.jpg|iter1"],
                         {"iteration": 1, "image_file": "IMG_001.jpg"})
        self.assertEqual(id_map["MVIMG-test.with.dots.jpg|iter42"],
                         {"iteration": 42, "image_file": "MVIMG-test.with.dots.jpg"})

    def test_handles_image_with_pipe_or_iter_in_name(self):
        # The regex anchors on the FINAL `|iter<N>$`, so an image with `iter`
        # earlier in the name should still be parsed correctly.
        from openai_batch_runner import _legacy_id_map_from_custom_ids
        id_map = _legacy_id_map_from_custom_ids(["my_iter_test.jpg|iter5"])
        self.assertEqual(id_map["my_iter_test.jpg|iter5"]["image_file"], "my_iter_test.jpg")
        self.assertEqual(id_map["my_iter_test.jpg|iter5"]["iteration"], 5)

    def test_skips_non_legacy_ids(self):
        from openai_batch_runner import _legacy_id_map_from_custom_ids
        # The new format `idx<N>` should NOT be matched here — the new
        # download path uses the state-file id_map for those.
        id_map = _legacy_id_map_from_custom_ids(["idx0", "idx42", "totally_unrelated"])
        self.assertEqual(id_map, {})


class TestLegacyAnthropicDecoder(unittest.TestCase):
    """Anthropic state files written before the batch_common refactor have
    no id_map. The legacy custom_id format was deterministic (image . and -
    replaced by _, then `_iter<N>`, truncated at 64 chars), and the original
    submission order is reproducible from the state metadata."""

    def test_encode_round_trip(self):
        from anthropic_batch_runner import _encode_legacy_anthropic_cid
        self.assertEqual(_encode_legacy_anthropic_cid("IMG-20260410-WA0016.jpg", 7),
                         "IMG_20260410_WA0016_jpg_iter7")
        self.assertEqual(_encode_legacy_anthropic_cid("MVIMG_20260308_142023.jpg", 1),
                         "MVIMG_20260308_142023_jpg_iter1")

    def test_reconstruct_single_subbatch(self):
        from anthropic_batch_runner import _legacy_id_map_for_subbatch
        state = {
            "iterations": 3,
            "n_requests": 6,
            "sub_index": 1,
            "submitted_at": "20260411_xxx",
            "images": ["a.jpg", "b.jpg"],
        }
        id_map = _legacy_id_map_for_subbatch(state, [state])
        self.assertEqual(len(id_map), 6)
        # Verify pair coverage
        pairs = {(e["image_file"], e["iteration"]) for e in id_map.values()}
        expected = {(img, it) for it in range(1, 4) for img in ["a.jpg", "b.jpg"]}
        self.assertEqual(pairs, expected)

    def test_reconstruct_multipart_no_collisions(self):
        from anthropic_batch_runner import _legacy_id_map_for_subbatch
        # 4 iters, 3 images = 12 pairs, split into 3 parts of 5/5/2
        images = ["a.jpg", "b.jpg", "c.jpg"]
        siblings = [
            {"iterations": 4, "n_requests": 5, "sub_index": 1,
             "submitted_at": "ts", "images": images},
            {"iterations": 4, "n_requests": 5, "sub_index": 2,
             "submitted_at": "ts", "images": images},
            {"iterations": 4, "n_requests": 2, "sub_index": 3,
             "submitted_at": "ts", "images": images},
        ]
        all_pairs = set()
        all_cids = set()
        for m in siblings:
            sub = _legacy_id_map_for_subbatch(m, siblings)
            for cid, e in sub.items():
                self.assertNotIn(cid, all_cids, f"cid {cid} duplicated across parts")
                all_cids.add(cid)
                all_pairs.add((e["image_file"], e["iteration"]))
        # Full coverage of 4 * 3 = 12 pairs
        self.assertEqual(len(all_pairs), 12)

    def test_returns_empty_when_state_lacks_iterations(self):
        from anthropic_batch_runner import _legacy_id_map_for_subbatch
        state = {"iterations": 0, "n_requests": 0, "sub_index": 1,
                 "submitted_at": "x", "images": ["a.jpg"]}
        self.assertEqual(_legacy_id_map_for_subbatch(state, [state]), {})


class TestPromptHash(unittest.TestCase):
    def test_prompt_hash_is_set(self):
        self.assertIsInstance(PROMPT_SHA256, str)
        self.assertEqual(len(PROMPT_SHA256), 64)  # SHA256 hex = 64 chars

    def test_prompt_hash_is_deterministic(self):
        # Re-import to verify the hash is computed deterministically
        import importlib
        import batch_common
        importlib.reload(batch_common)
        self.assertEqual(batch_common.PROMPT_SHA256, PROMPT_SHA256)


if __name__ == "__main__":
    unittest.main()
