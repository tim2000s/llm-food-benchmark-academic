#!/usr/bin/env python3
"""
Anthropic Message Batches API Runner
=====================================
Submits batch jobs via the Anthropic Message Batches API at 50% cost discount,
typically <1h turnaround.

Usage:
    python3 anthropic_batch_runner.py submit   --model claude-sonnet-4-6 --iterations 500
    python3 anthropic_batch_runner.py status
    python3 anthropic_batch_runner.py download

Each sub-batch is saved to its own state file (with a unique sub_index)
and produces a unique results file at download time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from food_nutrition_benchmark import RESULTS_DIR, DEFAULT_JPEG_QUALITY
from batch_common import (
    log,
    PROMPT_SHA256,
    SYSTEM_PROMPT,
    discover_images,
    preload_image_data,
    make_id_mapping,
    base_state_metadata,
    write_state_file,
    load_state_file,
    build_result_dicts,
    write_results_file,
)

BATCH_STATE_DIR = RESULTS_DIR / "batch_state_anthropic"
PROVIDER = "claude"

# Anthropic batch limit is 256 MB. Build aggressively to 150 MB to leave headroom
# for HTTP/JSON overhead and the per-request envelope.
SAFE_BATCH_MB = 150


# Legacy custom_id encoding used by older runs (pre-batch_common refactor):
# `<image with . and - replaced by _>_iter<N>`, truncated to 64 chars.
# This was lossy in general (dashes and dots collide on _) but unambiguous
# for the 13-image dataset used in this project. We rebuild the id_map for
# legacy state files by replaying the deterministic submission order:
#     pairs = [(it, img) for it in range(1, iterations+1) for img in images]
# split into chunks of `requests_per_subbatch`.
def _encode_legacy_anthropic_cid(image_file: str, iteration: int) -> str:
    return f"{image_file.replace('.', '_').replace('-', '_')}_iter{iteration}"[:64]


def _legacy_id_map_for_subbatch(meta: dict, sibling_metas: list[dict]) -> dict:
    """Reconstruct an id_map for a legacy Anthropic state file.

    Uses the original (it, img) submission order and the per-sub-batch chunk
    size derived from sibling state files for the same submission.
    """
    images = meta.get("images") or []
    iterations = int(meta.get("iterations") or 0)
    sub_idx = int(meta.get("sub_index") or 1)
    if not images or iterations <= 0:
        return {}

    submitted_at = meta.get("submitted_at")
    sibling_sizes = [int(m.get("n_requests") or 0)
                     for m in sibling_metas
                     if m.get("submitted_at") == submitted_at]
    if not sibling_sizes:
        return {}
    # The original code used a uniform `requests_per_subbatch` chunk; the last
    # part is whatever remained. So the per-chunk size equals the maximum size
    # observed across siblings.
    rps = max(sibling_sizes)

    full_pairs = [(it, img) for it in range(1, iterations + 1) for img in images]
    start = (sub_idx - 1) * rps
    sub_pairs = full_pairs[start:start + rps]

    id_map = {}
    for it, img in sub_pairs:
        cid = _encode_legacy_anthropic_cid(img, it)
        id_map[cid] = {"iteration": it, "image_file": img}
    return id_map


def build_request(model: str, custom_id: str, image_b64: str) -> dict:
    """Build a single batch request for the Anthropic Messages API.

    custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — we use idx<N> form.
    """
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 8000,
            "temperature": 0.01,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    }},
                ],
            }],
        },
    }


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_batch(model: str, iterations: int, api_key: str,
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> list[str]:
    client = anthropic.Anthropic(api_key=api_key)
    images = discover_images()
    if not images:
        log.error("No images found")
        sys.exit(1)

    n_total = iterations * len(images)
    log.info(f"Building batch: {iterations} iterations x {len(images)} images = {n_total} requests")
    log.info(f"Model: {model}")

    image_data = preload_image_data(images, PROVIDER, jpeg_quality)

    # Build all (iteration, image) pairs and the records/id_map
    pairs = [(it, img.name) for it in range(1, iterations + 1) for img in images]
    records, full_id_map = make_id_mapping(pairs)

    # Estimate per-request size (sample one record per image and take the max)
    max_request_size = 0
    for img_name, img_b64 in image_data.items():
        sample = build_request(model, "idx0", img_b64)
        sz = len(json.dumps(sample))
        max_request_size = max(max_request_size, sz)
    log.info(f"Worst-case per-request size: {max_request_size // 1024}KB")

    estimated_total_mb = max_request_size * len(records) / 1024 / 1024
    log.info(f"Estimated payload: ~{estimated_total_mb:.0f}MB (Anthropic limit: 256MB)")

    requests_per_subbatch = max(1, int((SAFE_BATCH_MB * 1024 * 1024) / max_request_size))
    requests_per_subbatch = min(len(records), requests_per_subbatch)
    n_subbatches = (len(records) + requests_per_subbatch - 1) // requests_per_subbatch
    log.info(f"Splitting into {n_subbatches} sub-batch(es) of up to {requests_per_subbatch} requests each")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_ids = []

    for sub_idx in range(1, n_subbatches + 1):
        start = (sub_idx - 1) * requests_per_subbatch
        sub_records = records[start: start + requests_per_subbatch]
        if not sub_records:
            continue
        log.info(f"Sub-batch {sub_idx}/{n_subbatches}: {len(sub_records)} requests")

        # Build the actual request payloads
        sub_requests = [build_request(model, r["custom_id"], image_data[r["image_file"]])
                        for r in sub_records]

        try:
            batch = client.messages.batches.create(requests=sub_requests)
        except Exception as e:
            log.error(f"  Batch creation failed: {e}")
            continue
        log.info(f"  Batch created: {batch.id} (status: {batch.processing_status})")

        # Build a partial id_map containing only this sub-batch's mappings
        sub_id_map = {r["custom_id"]: {"iteration": r["iteration"], "image_file": r["image_file"]}
                      for r in sub_records}
        sub_pairs = [(r["iteration"], r["image_file"]) for r in sub_records]

        meta = base_state_metadata(model, PROVIDER, images, sub_pairs, sub_id_map,
                                    sub_idx, n_subbatches, jpeg_quality)
        meta["batch_id"] = batch.id
        meta["timestamp"] = timestamp

        state_name = f"batch_{timestamp}_part{sub_idx:03d}.json"
        write_state_file(BATCH_STATE_DIR, state_name, meta)
        log.info(f"  State: {state_name}")
        batch_ids.append(batch.id)

    return batch_ids


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_all(api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    if not state_files:
        log.info("No Anthropic batches found.")
        return
    log.info(f"Found {len(state_files)} batch(es):")
    for sf in state_files:
        meta = load_state_file(sf)
        try:
            batch = client.messages.batches.retrieve(meta["batch_id"])
            rc = batch.request_counts
            log.info(f"  {sf.name}  model={meta['model']}  sub={meta.get('sub_index')}/{meta.get('n_subbatches')}  "
                     f"status={batch.processing_status}  succeeded={rc.succeeded}/{rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired}")
        except Exception as e:
            log.error(f"  {sf.name}: {e}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _expected_results_filename(meta: dict) -> str:
    ts = meta.get("timestamp") or meta.get("submitted_at") or "unknown"
    sub = meta.get("sub_index", 0)
    n_sub = meta.get("n_subbatches", 1)
    return f"results_anthropic_batch_{ts}_part{sub:03d}_of_{n_sub:03d}.json"


def download_all(api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    # Pre-load all state metas so the legacy decoder can derive the per-chunk
    # size from siblings of the same submission.
    sibling_metas = [load_state_file(sf) for sf in state_files]
    for sf in state_files:
        meta = load_state_file(sf)
        results_file = RESULTS_DIR / _expected_results_filename(meta)
        if results_file.exists():
            log.info(f"  {sf.name}: already downloaded -> {results_file.name}")
            continue

        try:
            batch = client.messages.batches.retrieve(meta["batch_id"])
        except Exception as e:
            log.error(f"  {sf.name}: retrieve failed: {e}")
            continue

        log.info(f"  {sf.name}: status = {batch.processing_status}")
        if batch.processing_status != "ended":
            log.info(f"    skipping (not yet ended)")
            continue

        try:
            results_iter = client.messages.batches.results(meta["batch_id"])
        except Exception as e:
            log.error(f"    results stream failed: {e}")
            continue

        items = []
        for result in results_iter:
            cid = result.custom_id
            api_error = None
            api_error_class = None
            raw_text = None
            usage = None

            if result.result.type == "succeeded":
                msg = result.result.message
                # Capture token usage
                if hasattr(msg, "usage") and msg.usage is not None:
                    usage = {
                        "input_tokens": getattr(msg.usage, "input_tokens", None),
                        "output_tokens": getattr(msg.usage, "output_tokens", None),
                    }
                # Verify model
                if hasattr(msg, "model") and msg.model and msg.model != meta["model"]:
                    log.warning(f"    model mismatch on {cid}: requested={meta['model']}, returned={msg.model}")
                if msg.content:
                    raw_text = msg.content[0].text
                else:
                    raw_text = ""
            else:
                # Anthropic batch result types: errored, canceled, expired
                err_obj = getattr(result.result, "error", None)
                api_error = f"{result.result.type}: {err_obj if err_obj else ''}"[:500]
                api_error_class = "api"

            items.append({
                "custom_id": cid,
                "raw_text": raw_text,
                "usage": usage,
                "api_error": api_error,
                "api_error_class": api_error_class,
            })

        # Backwards compatibility: legacy state files lack id_map. Reconstruct
        # one from the deterministic legacy encoding (lossless for our 13-image
        # set, where no two filenames collide after `.`/`-` → `_` substitution).
        id_map = meta.get("id_map")
        if not id_map:
            id_map = _legacy_id_map_for_subbatch(meta, sibling_metas)
            if not id_map:
                log.error(f"    state file has no id_map and could not "
                          f"reconstruct one — skipping")
                continue
            log.info(f"    legacy state file: rebuilt id_map for {len(id_map)} ids")

        results = build_result_dicts(items, meta["model"], PROVIDER, id_map)

        ts = meta.get("timestamp") or meta.get("submitted_at") or "unknown"
        sub = meta.get("sub_index") or 0
        meta_extra = {
            "run_id": f"anthropic_batch_{ts}_part{sub:03d}",
            "iterations": "batch",
            "display": meta["model"],
            "images": meta["images"],
            "image_max_dim": meta.get("max_image_dim"),
            "jpeg_quality": meta.get("jpeg_quality"),
            "prompt_sha256": meta.get("prompt_sha256"),
            "batch_provider_id": meta.get("batch_id"),
        }
        write_results_file(results_file, meta["model"], PROVIDER, meta_extra, results)
        n_succ = sum(1 for r in results if r["success"])
        log.info(f"    saved: {results_file.name}  ({n_succ}/{len(results)} succeeded)")


def main():
    parser = argparse.ArgumentParser(description="Anthropic Batch API Runner")
    parser.add_argument("action", choices=["submit", "status", "download"])
    parser.add_argument("--model", required=True,
                        help="Model ID, e.g. claude-sonnet-4-6 (no default — must be set explicitly)")
    parser.add_argument("--iterations", "-n", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    if args.action == "submit":
        submit_batch(args.model, args.iterations, api_key, args.jpeg_quality)
    elif args.action == "status":
        status_all(api_key)
    elif args.action == "download":
        download_all(api_key)


if __name__ == "__main__":
    main()
