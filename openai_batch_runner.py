#!/usr/bin/env python3
"""
OpenAI Batch API Runner
========================
Submits batch jobs via the OpenAI Batch API at 50% cost discount.

Usage:
    python3 openai_batch_runner.py submit   --model gpt-5.4 --iterations 500
    python3 openai_batch_runner.py status
    python3 openai_batch_runner.py download

Each sub-batch is saved to its own state file (with a unique sub_index)
and produces a unique results file at download time. The script never
overwrites previously-downloaded data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from food_nutrition_benchmark import RESULTS_DIR, MAX_IMAGE_DIM, DEFAULT_JPEG_QUALITY
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

BATCH_STATE_DIR = RESULTS_DIR / "batch_state_openai"
PROVIDER = "openai"
ENDPOINT = "/v1/responses"

# OpenAI hard limit is 200 MB per upload. Build to 140 MB worst case.
MAX_BATCH_MB = 140


def build_request(model: str, custom_id: str, image_b64: str) -> dict:
    """Build a single batch line for the OpenAI Responses API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }],
            "max_output_tokens": 6000,
            "temperature": 0.01,
            "store": False,
        },
    }


def upload_via_curl(jsonl_path: Path, api_key: str) -> str:
    """Upload a JSONL file via curl. The OpenAI SDK has historical broken-pipe
    issues with large multipart uploads, so we shell out for reliability."""
    result = subprocess.run(
        [
            "curl", "-s", "-w", "\nHTTP %{http_code}",
            "https://api.openai.com/v1/files",
            "-H", f"Authorization: Bearer {api_key}",
            "-F", "purpose=batch",
            "-F", f"file=@{jsonl_path}",
        ],
        capture_output=True, text=True, timeout=900,
    )
    out = result.stdout
    if "HTTP 200" not in out:
        raise RuntimeError(f"Upload failed: {out}")
    return json.loads(out.split("\nHTTP")[0])["id"]


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_batch(model: str, iterations: int, api_key: str,
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> list[str]:
    client = openai.OpenAI(api_key=api_key)
    images = discover_images()
    if not images:
        log.error(f"No images found")
        sys.exit(1)

    n_total = iterations * len(images)
    log.info(f"Building batch: {iterations} iterations x {len(images)} images = {n_total} requests")
    log.info(f"Model: {model}")

    image_data = preload_image_data(images, PROVIDER, jpeg_quality)

    # Determine sub-batch size from worst-case (largest image) request size
    max_request_size = 0
    for img_name, img_b64 in image_data.items():
        sample_req = build_request(model, "idx0", img_b64)
        sz = len(json.dumps(sample_req)) + 1
        max_request_size = max(max_request_size, sz)
    log.info(f"Worst-case per-request size: {max_request_size // 1024}KB")

    requests_per_subbatch = max(1, int((MAX_BATCH_MB * 1024 * 1024) / max_request_size))
    requests_per_subbatch = min(n_total, requests_per_subbatch)
    n_subbatches = (n_total + requests_per_subbatch - 1) // requests_per_subbatch
    log.info(f"Splitting into {n_subbatches} sub-batch(es) of up to {requests_per_subbatch} requests each")

    # Build the (iteration, image) pairs once and chunk them
    pairs = [(it, img.name) for it in range(1, iterations + 1) for img in images]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_ids = []

    for sub_idx in range(1, n_subbatches + 1):
        start = (sub_idx - 1) * requests_per_subbatch
        chunk = pairs[start: start + requests_per_subbatch]
        log.info(f"Sub-batch {sub_idx}/{n_subbatches}: {len(chunk)} requests")

        # Build records and id_map for this sub-batch
        records, id_map = make_id_mapping(chunk)

        # Write JSONL
        jsonl_path = BATCH_STATE_DIR / f"input_{model}_{timestamp}_part{sub_idx:03d}.jsonl"
        BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w") as f:
            for rec in records:
                req = build_request(model, rec["custom_id"], image_data[rec["image_file"]])
                f.write(json.dumps(req) + "\n")
        size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        log.info(f"  JSONL: {jsonl_path.name} ({size_mb:.1f} MB)")
        if size_mb > 199:
            log.error(f"  File exceeds 200 MB limit, skipping")
            jsonl_path.unlink()
            continue

        # Upload
        log.info(f"  Uploading via curl...")
        try:
            file_id = upload_via_curl(jsonl_path, api_key)
        except Exception as e:
            log.error(f"  Upload failed: {e}")
            continue
        log.info(f"  Uploaded: {file_id}")

        # Create batch
        try:
            batch = client.batches.create(
                input_file_id=file_id,
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata={"display_name": f"food-nutrition-{model}-{timestamp}-p{sub_idx:03d}"},
            )
        except Exception as e:
            log.error(f"  Batch creation failed: {e}")
            continue
        log.info(f"  Batch created: {batch.id} (status: {batch.status})")

        # State file
        meta = base_state_metadata(model, PROVIDER, images, chunk, id_map,
                                    sub_idx, n_subbatches, jpeg_quality)
        meta["batch_id"] = batch.id
        meta["input_file_id"] = file_id
        meta["timestamp"] = timestamp
        state_name = f"batch_{timestamp}_part{sub_idx:03d}.json"
        write_state_file(BATCH_STATE_DIR, state_name, meta)
        log.info(f"  State: {state_name}")

        # Delete the local JSONL after successful upload to free disk
        try:
            jsonl_path.unlink()
        except Exception:
            pass

        batch_ids.append(batch.id)

    return batch_ids


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_all(api_key: str):
    client = openai.OpenAI(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    if not state_files:
        log.info("No OpenAI batches found.")
        return
    log.info(f"Found {len(state_files)} batch(es):")
    for sf in state_files:
        meta = load_state_file(sf)
        try:
            batch = client.batches.retrieve(meta["batch_id"])
            log.info(f"  {sf.name}  model={meta['model']}  sub={meta.get('sub_index')}/{meta.get('n_subbatches')}  "
                     f"status={batch.status}  counts={batch.request_counts}")
        except Exception as e:
            log.error(f"  {sf.name}: {e}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _expected_results_filename(meta: dict) -> str:
    """Build the unique results filename for a sub-batch.

    Uses BOTH submitted timestamp AND sub_index so multi-part batches don't
    collide on filename (this was bug B3 in the review)."""
    ts = meta.get("timestamp") or meta.get("submitted_at") or "unknown"
    sub = meta.get("sub_index", 0)
    n_sub = meta.get("n_subbatches", 1)
    return f"results_openai_batch_{ts}_part{sub:03d}_of_{n_sub:03d}.json"


# Legacy custom_id format used by older runs (pre-batch_common refactor):
# `<image_filename>|iter<N>`. The format is self-describing, so we can
# reconstruct an id_map for any state file that lacks one.
_LEGACY_CID_RE = re.compile(r"^(.+)\|iter(\d+)$")


def _legacy_id_map_from_custom_ids(custom_ids: list[str]) -> dict:
    """Build an id_map from legacy custom_ids of the form `<image>|iter<N>`.

    Skips any custom_id that doesn't match the legacy pattern (caller should
    log if the result is empty).
    """
    out = {}
    for cid in custom_ids:
        m = _LEGACY_CID_RE.match(cid)
        if m:
            out[cid] = {"iteration": int(m.group(2)), "image_file": m.group(1)}
    return out


def download_all(api_key: str):
    client = openai.OpenAI(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    for sf in state_files:
        meta = load_state_file(sf)
        results_file = RESULTS_DIR / _expected_results_filename(meta)
        if results_file.exists():
            log.info(f"  {sf.name}: already downloaded -> {results_file.name}")
            continue

        try:
            batch = client.batches.retrieve(meta["batch_id"])
        except Exception as e:
            log.error(f"  {sf.name}: retrieve failed: {e}")
            continue

        log.info(f"  {sf.name}: status = {batch.status}")
        if batch.status != "completed":
            log.info(f"    skipping (not yet completed)")
            continue
        if not batch.output_file_id:
            log.warning(f"    no output_file_id")
            continue

        try:
            content = client.files.content(batch.output_file_id).text
        except Exception as e:
            log.error(f"    download failed: {e}")
            continue

        lines = [l for l in content.splitlines() if l.strip()]
        log.info(f"    got {len(lines)} responses")

        # Backwards compatibility: state files written before the
        # batch_common refactor lack `id_map`. The legacy custom_id format
        # `<image>|iter<N>` is self-describing, so we synthesise an id_map
        # from the response payload.
        id_map = meta.get("id_map")
        if not id_map:
            cids = [json.loads(l).get("custom_id", "") for l in lines]
            id_map = _legacy_id_map_from_custom_ids(cids)
            if not id_map:
                log.error(f"    state file has no id_map and custom_ids "
                          f"are not in the legacy format — cannot decode")
                continue
            log.info(f"    legacy state file: built id_map for {len(id_map)} ids")

        # Convert each line to a normalised item dict for build_result_dicts
        items = []
        for line in lines:
            item = json.loads(line)
            cid = item.get("custom_id", "")
            response = item.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}

            # Capture token usage if present
            usage = None
            if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                usage = {
                    "input_tokens": body["usage"].get("input_tokens"),
                    "output_tokens": body["usage"].get("output_tokens"),
                }

            api_error = None
            api_error_class = None
            raw_text = None

            if status_code != 200:
                api_error = f"HTTP {status_code}: {json.dumps(body)[:300]}"
                api_error_class = "api"
            else:
                # Extract text from output[].content[].text
                output = body.get("output", [])
                pieces = []
                for out in output:
                    for c in out.get("content", []):
                        if c.get("type") == "output_text":
                            pieces.append(c.get("text", ""))
                raw_text = "".join(pieces)

            # Verify the model returned matches what we asked for
            actual_model = body.get("model")
            if actual_model and actual_model != meta["model"]:
                log.warning(f"    model mismatch on {cid}: requested={meta['model']}, returned={actual_model}")

            items.append({
                "custom_id": cid,
                "raw_text": raw_text,
                "usage": usage,
                "api_error": api_error,
                "api_error_class": api_error_class,
            })

        results = build_result_dicts(items, meta["model"], PROVIDER, id_map)

        ts = meta.get("timestamp") or meta.get("submitted_at") or "unknown"
        sub = meta.get("sub_index") or 0
        meta_extra = {
            "run_id": f"openai_batch_{ts}_part{sub:03d}",
            "iterations": "batch",
            "display": meta["model"],
            "images": meta.get("images", []),
            "image_max_dim": meta.get("max_image_dim"),
            "jpeg_quality": meta.get("jpeg_quality"),
            "prompt_sha256": meta.get("prompt_sha256"),
            "batch_provider_id": meta.get("batch_id"),
        }
        write_results_file(results_file, meta["model"], PROVIDER, meta_extra, results)
        n_succ = sum(1 for r in results if r["success"])
        log.info(f"    saved: {results_file.name}  ({n_succ}/{len(results)} succeeded)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API Runner")
    parser.add_argument("action", choices=["submit", "status", "download"])
    parser.add_argument("--model", required=True,
                        help="Model ID, e.g. gpt-5.4 (no default — must be set explicitly)")
    parser.add_argument("--iterations", "-n", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    if args.action == "submit":
        submit_batch(args.model, args.iterations, api_key, args.jpeg_quality)
    elif args.action == "status":
        status_all(api_key)
    elif args.action == "download":
        download_all(api_key)


if __name__ == "__main__":
    main()
