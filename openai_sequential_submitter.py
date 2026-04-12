#!/usr/bin/env python3
"""
Sequential OpenAI batch submitter that respects the enqueued-token limit.

OpenAI enforces an "enqueued tokens" limit per organization per model
(900,000 tokens for gpt-5.4 as observed). This means only a few batches
can be in flight at once. This script:

1. Reads `resubmit_pairs.json` (a list of [iteration, image_file] tuples)
2. Builds sub-batches small enough to stay under the token limit
3. Submits a sub-batch, waits for it to complete, then submits the next
4. Auto-resumes from existing state files (skips completed sub-batches)
5. Retries submission errors with exponential backoff

Usage:
    python3 openai_sequential_submitter.py --model gpt-5.4
    python3 openai_sequential_submitter.py --model gpt-5.4 --max-per-subbatch 75
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from food_nutrition_benchmark import IMAGES_DIR, RESULTS_DIR, MAX_IMAGE_DIM, DEFAULT_JPEG_QUALITY
from batch_common import (
    log,
    PROMPT_SHA256,
    resize_to_jpeg_bytes,
    discover_images,
    make_id_mapping,
    base_state_metadata,
    write_state_file,
    load_state_file,
)
from openai_batch_runner import build_request, upload_via_curl, BATCH_STATE_DIR

# Conservative — well under the 900K enqueued-token cap.
# Each food request uses ~3K input tokens and up to 6K output tokens.
# 100 requests × ~9K = 900K. Use 80 to leave some headroom.
DEFAULT_MAX_PER_SUBBATCH = 80
POLL_INTERVAL_SEC = 60
MAX_WAIT_PER_BATCH_SEC = 24 * 60 * 60
MAX_SUBMIT_RETRIES = 5


def already_submitted_pairs(timestamp_prefix: str) -> set[tuple[int, str]]:
    """Scan existing state files (matching the timestamp prefix) and return
    the set of (iteration, image_file) pairs already submitted."""
    submitted = set()
    for sf in BATCH_STATE_DIR.glob(f"batch_seq_{timestamp_prefix}*.json"):
        try:
            meta = load_state_file(sf)
        except Exception:
            continue
        for entry in meta.get("id_map", {}).values():
            submitted.add((entry["iteration"], entry["image_file"]))
    return submitted


def submit_one_subbatch(client, api_key: str, model: str, image_data: dict[str, str],
                          pairs: list[tuple[int, str]], sub_idx: int, timestamp: str,
                          jpeg_quality: int):
    """Submit one sub-batch and return its batch_id and state file path."""
    records, id_map = make_id_mapping(pairs)

    # Write the JSONL
    jsonl_path = BATCH_STATE_DIR / f"input_seq_{model}_{timestamp}_seq{sub_idx:03d}.jsonl"
    BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for rec in records:
            req = build_request(model, rec["custom_id"], image_data[rec["image_file"]])
            f.write(json.dumps(req) + "\n")
    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    log.info(f"  Sub-batch {sub_idx}: {len(pairs)} requests, {size_mb:.1f}MB")

    # Upload
    file_id = upload_via_curl(jsonl_path, api_key)
    log.info(f"  Uploaded: {file_id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"display_name": f"food-nutrition-{model}-seq{sub_idx:03d}"},
    )
    log.info(f"  Batch: {batch.id} (status: {batch.status})")

    # State file — use the shared format and store the id_map
    images = discover_images()  # full list (state file metadata expects this)
    meta = base_state_metadata(model, "openai", images, pairs, id_map,
                                sub_idx, n_subbatches=0,  # unknown total in resume mode
                                jpeg_quality=jpeg_quality)
    meta["batch_id"] = batch.id
    meta["input_file_id"] = file_id
    meta["timestamp"] = timestamp
    state_name = f"batch_seq_{timestamp}_seq{sub_idx:03d}.json"
    state_file = write_state_file(BATCH_STATE_DIR, state_name, meta)

    # Delete local JSONL after upload
    try:
        jsonl_path.unlink()
    except Exception:
        pass

    return batch.id, state_file


def wait_for_completion(client, batch_id: str, max_wait_sec: int) -> str | None:
    """Poll a batch until it completes or fails."""
    start = time.monotonic()
    last_status = None
    while True:
        elapsed = time.monotonic() - start
        if elapsed > max_wait_sec:
            log.warning(f"  TIMEOUT after {elapsed:.0f}s")
            return None
        try:
            b = client.batches.retrieve(batch_id)
        except Exception as e:
            log.warning(f"  retrieve failed (will retry): {e}")
            time.sleep(POLL_INTERVAL_SEC)
            continue
        if b.status != last_status:
            log.info(f"  [{elapsed:.0f}s] {b.status}  completed={b.request_counts.completed}/{b.request_counts.total}")
            last_status = b.status
        if b.status in ("completed", "failed", "expired", "cancelled"):
            return b.status
        time.sleep(POLL_INTERVAL_SEC)


def submit_with_retries(client, api_key: str, model: str, image_data: dict[str, str],
                          chunk: list[tuple[int, str]], sub_idx: int, timestamp: str,
                          jpeg_quality: int):
    """Try to submit a sub-batch up to MAX_SUBMIT_RETRIES times with exponential backoff."""
    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        try:
            return submit_one_subbatch(client, api_key, model, image_data, chunk,
                                         sub_idx, timestamp, jpeg_quality)
        except Exception as e:
            wait = min(300, 30 * (2 ** (attempt - 1)))
            log.error(f"  submission attempt {attempt}/{MAX_SUBMIT_RETRIES} failed: {e}")
            if attempt == MAX_SUBMIT_RETRIES:
                log.error(f"  giving up on sub-batch {sub_idx}")
                return None, None
            log.info(f"  retrying in {wait}s")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Sequential OpenAI batch submitter")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-per-subbatch", type=int, default=DEFAULT_MAX_PER_SUBBATCH)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--pairs-file", type=Path,
                        default=BATCH_STATE_DIR / "resubmit_pairs.json",
                        help="Path to resubmit_pairs.json")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Use a specific run timestamp (auto-resume from existing state). "
                             "Default: a new timestamp for a fresh run.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    if not args.pairs_file.exists():
        log.error(f"{args.pairs_file} not found")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    with open(args.pairs_file) as f:
        all_pairs = [tuple(p) for p in json.load(f)]
    log.info(f"Loaded {len(all_pairs)} pairs from {args.pairs_file}")

    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info(f"Run timestamp: {timestamp}")

    # Auto-resume: skip pairs already submitted under this timestamp
    already = already_submitted_pairs(timestamp)
    if already:
        log.info(f"Auto-resume: {len(already)} pairs already submitted under this timestamp")
        all_pairs = [p for p in all_pairs if tuple(p) not in already]
        log.info(f"Remaining pairs: {len(all_pairs)}")

    if not all_pairs:
        log.info("Nothing to do.")
        return

    # Pre-resize images
    images_in_pairs = sorted(set(img for _, img in all_pairs))
    image_data = {}
    for img_name in images_in_pairs:
        path = IMAGES_DIR / img_name
        if not path.exists():
            log.error(f"Image file missing: {img_name}")
            sys.exit(1)
        b = resize_to_jpeg_bytes(path, MAX_IMAGE_DIM["openai"], args.jpeg_quality)
        image_data[img_name] = base64.b64encode(b).decode("ascii")
    log.info(f"Resized {len(image_data)} images")

    # Chunk
    chunks = [all_pairs[i:i + args.max_per_subbatch]
              for i in range(0, len(all_pairs), args.max_per_subbatch)]
    n_chunks = len(chunks)
    log.info(f"Will submit {n_chunks} sub-batches of up to {args.max_per_subbatch} requests each")
    log.info(f"Estimated runtime: {n_chunks * 5}-{n_chunks * 15} minutes")

    # Find the next sub_idx to use, accounting for any existing state files
    existing_indices = set()
    for sf in BATCH_STATE_DIR.glob(f"batch_seq_{timestamp}*.json"):
        try:
            meta = load_state_file(sf)
            existing_indices.add(meta.get("sub_index", 0))
        except Exception:
            continue
    next_sub_idx = max(existing_indices, default=0) + 1

    for i, chunk in enumerate(chunks):
        sub_idx = next_sub_idx + i
        log.info(f"\n=== Sub-batch {i + 1}/{n_chunks} (sub_idx={sub_idx}) ===")

        batch_id, state_file = submit_with_retries(
            client, api_key, args.model, image_data, chunk,
            sub_idx, timestamp, args.jpeg_quality,
        )
        if batch_id is None:
            log.error(f"  could not submit sub-batch {sub_idx}, moving on")
            continue

        status = wait_for_completion(client, batch_id, MAX_WAIT_PER_BATCH_SEC)
        if status == "failed":
            try:
                b = client.batches.retrieve(batch_id)
                for e in (b.errors.data if b.errors else []):
                    log.error(f"    {e.code}: {e.message}")
            except Exception:
                pass
        elif status == "completed":
            log.info(f"  Batch COMPLETED")
        else:
            log.warning(f"  Batch ended with status: {status}")

    log.info("\nAll sub-batches processed.")


if __name__ == "__main__":
    main()
