#!/usr/bin/env python3
"""
Gemini Batch API Runner
========================
Submits batch jobs via the Gemini Batch API at 50% cost discount.

Usage:
    python3 gemini_batch_runner.py submit   --model gemini-2.5-pro --iterations 500
    python3 gemini_batch_runner.py status
    python3 gemini_batch_runner.py download
    python3 gemini_batch_runner.py run      --model gemini-2.5-pro --iterations 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types as genai_types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from food_nutrition_benchmark import RESULTS_DIR, MAX_IMAGE_DIM, DEFAULT_JPEG_QUALITY
from batch_common import (
    log,
    PROMPT_SHA256,
    SYSTEM_PROMPT,
    discover_images,
    resize_to_jpeg_bytes,
    make_id_mapping,
    base_state_metadata,
    write_state_file,
    load_state_file,
    build_result_dicts,
    write_results_file,
)

# Note the explicit "_gemini" suffix — earlier versions used a generic
# "batch_state/" directory which collided with assumptions in other tools.
BATCH_STATE_DIR = RESULTS_DIR / "batch_state_gemini"
PROVIDER = "gemini"


def build_request(custom_id: str, file_uri: str, mime_type: str = "image/jpeg") -> dict:
    return {
        "key": custom_id,
        "request": {
            "contents": [{
                "parts": [
                    {"text": SYSTEM_PROMPT},
                    {"file_data": {
                        "mime_type": mime_type,
                        "file_uri": file_uri,
                    }},
                ],
            }],
            "generation_config": {
                "temperature": 0.01,
                "top_p": 0.95,
                "top_k": 8,
                "max_output_tokens": 8000,
            },
        },
    }


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_batch(model: str, iterations: int, api_key: str,
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    client = genai.Client(api_key=api_key)
    images = discover_images()
    if not images:
        log.error("No images found")
        sys.exit(1)

    n_total = iterations * len(images)
    log.info(f"Building batch: {iterations} iterations x {len(images)} images = {n_total} requests")
    log.info(f"Model: {model}")

    BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Pre-resize each image once and upload via Files API. This is the key
    # difference from OpenAI/Anthropic — we don't inline the base64 in the
    # batch JSONL, we reference uploaded files. This dramatically reduces
    # JSONL size for large iteration counts.
    log.info("Uploading images via Files API...")
    image_uris = {}
    for img_path in images:
        resized_bytes = resize_to_jpeg_bytes(img_path, MAX_IMAGE_DIM[PROVIDER], jpeg_quality)
        # Files API requires a file path, so write to a tmp file in the state dir
        tmp_path = BATCH_STATE_DIR / f"_tmp_{timestamp}_{img_path.stem}.jpg"
        tmp_path.write_bytes(resized_bytes)
        try:
            uploaded = client.files.upload(
                file=str(tmp_path),
                config=genai_types.UploadFileConfig(
                    mime_type="image/jpeg",
                    display_name=f"food-{img_path.stem}-{timestamp}",
                ),
            )
        finally:
            tmp_path.unlink(missing_ok=True)  # always delete the tmp file
        image_uris[img_path.name] = uploaded.uri
        log.info(f"  {img_path.name}: {len(resized_bytes) // 1024}KB → {uploaded.uri}")

    # Build records and id_map for the full batch
    pairs = [(it, img.name) for it in range(1, iterations + 1) for img in images]
    records, id_map = make_id_mapping(pairs)

    # Build JSONL referencing the uploaded URIs
    jsonl_path = BATCH_STATE_DIR / f"_input_{model}_{timestamp}.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            req = build_request(rec["custom_id"], image_uris[rec["image_file"]])
            f.write(json.dumps(req) + "\n")
    file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    log.info(f"Built JSONL: {file_size_mb:.1f}MB, {len(records)} requests")

    # Upload the batch JSONL
    log.info("Uploading batch JSONL...")
    try:
        uploaded = client.files.upload(
            file=str(jsonl_path),
            config=genai_types.UploadFileConfig(
                mime_type="application/jsonl",
                display_name=f"food-batch-{model}-{timestamp}",
            ),
        )
    finally:
        # Always delete the local JSONL after upload — no point keeping it
        jsonl_path.unlink(missing_ok=True)
    log.info(f"Uploaded as: {uploaded.name}")

    # Submit the batch
    batch_job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": f"food-nutrition-{model}-{timestamp}"},
    )
    log.info(f"Batch submitted: {batch_job.name} (state: {batch_job.state.name})")

    # State file
    meta = base_state_metadata(model, PROVIDER, images, pairs, id_map,
                                sub_index=1, n_subbatches=1, jpeg_quality=jpeg_quality)
    meta["batch_name"] = batch_job.name
    meta["uploaded_jsonl"] = uploaded.name
    meta["timestamp"] = timestamp
    state_name = f"batch_{timestamp}.json"
    write_state_file(BATCH_STATE_DIR, state_name, meta)
    log.info(f"State: {state_name}")

    return batch_job.name


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status_all(api_key: str):
    client = genai.Client(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    if not state_files:
        log.info("No Gemini batches found.")
        return
    log.info(f"Found {len(state_files)} batch(es):")
    for sf in state_files:
        meta = load_state_file(sf)
        try:
            batch = client.batches.get(name=meta["batch_name"])
            log.info(f"  {sf.name}  model={meta['model']}  state={batch.state.name}")
        except Exception as e:
            log.error(f"  {sf.name}: {e}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _expected_results_filename(meta: dict) -> str:
    ts = meta.get("timestamp") or "unknown"
    sub = meta.get("sub_index", 0)
    n_sub = meta.get("n_subbatches", 1)
    return f"results_gemini_batch_{ts}_part{sub:03d}_of_{n_sub:03d}.json"


def download_all(api_key: str):
    client = genai.Client(api_key=api_key)
    state_files = sorted(BATCH_STATE_DIR.glob("batch_*.json"))
    for sf in state_files:
        meta = load_state_file(sf)
        results_file = RESULTS_DIR / _expected_results_filename(meta)
        if results_file.exists():
            log.info(f"  {sf.name}: already downloaded -> {results_file.name}")
            continue

        try:
            batch = client.batches.get(name=meta["batch_name"])
        except Exception as e:
            log.error(f"  {sf.name}: retrieve failed: {e}")
            continue

        log.info(f"  {sf.name}: state = {batch.state.name}")
        if batch.state.name != "JOB_STATE_SUCCEEDED":
            log.info(f"    skipping (not yet succeeded)")
            continue

        # Get the raw responses (inline or file-based)
        results_data = []
        if hasattr(batch.dest, "inlined_responses") and batch.dest.inlined_responses:
            for r in batch.dest.inlined_responses:
                results_data.append(r)
        elif hasattr(batch.dest, "file_name") and batch.dest.file_name:
            try:
                content = client.files.download(file=batch.dest.file_name)
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                for line in content.splitlines():
                    if line.strip():
                        results_data.append(json.loads(line))
            except Exception as e:
                log.error(f"    file download failed: {e}")
                continue

        log.info(f"    got {len(results_data)} responses")

        items = []
        for item in results_data:
            cid = item.get("key", "")
            response = item.get("response") or {}
            api_error = None
            api_error_class = None
            raw_text = None
            usage = None

            # Detect API-level failure (no 'candidates' in the response)
            if isinstance(response, str):
                # Some response formats wrap text directly
                raw_text = response
            elif isinstance(response, dict):
                # Check for usage metadata
                um = response.get("usageMetadata") or response.get("usage_metadata")
                if isinstance(um, dict):
                    usage = {
                        "input_tokens": um.get("promptTokenCount") or um.get("prompt_token_count"),
                        "output_tokens": um.get("candidatesTokenCount") or um.get("candidates_token_count"),
                    }

                # Check for finishReason indicating safety filter etc.
                candidates = response.get("candidates")
                if not candidates:
                    api_error = f"no candidates in response: {json.dumps(response)[:300]}"
                    api_error_class = "api"
                else:
                    try:
                        first = candidates[0]
                        finish_reason = first.get("finishReason") or first.get("finish_reason")
                        parts = first.get("content", {}).get("parts") or []
                        text_pieces = [p.get("text", "") for p in parts if "text" in p]
                        raw_text = "".join(text_pieces)
                        if not raw_text and finish_reason and finish_reason != "STOP":
                            api_error = f"finishReason={finish_reason}, no text"
                            api_error_class = "api"
                    except (KeyError, IndexError, TypeError) as e:
                        api_error = f"unexpected response shape: {e}"
                        api_error_class = "api"
            else:
                api_error = f"unexpected response type: {type(response).__name__}"
                api_error_class = "api"

            items.append({
                "custom_id": cid,
                "raw_text": raw_text,
                "usage": usage,
                "api_error": api_error,
                "api_error_class": api_error_class,
            })

        results = build_result_dicts(items, meta["model"], PROVIDER, meta["id_map"])

        meta_extra = {
            "run_id": f"gemini_batch_{meta.get('timestamp')}",
            "iterations": "batch",
            "display": meta["model"],
            "images": meta["images"],
            "image_max_dim": meta.get("max_image_dim"),
            "jpeg_quality": meta.get("jpeg_quality"),
            "prompt_sha256": meta.get("prompt_sha256"),
            "batch_provider_id": meta.get("batch_name"),
        }
        write_results_file(results_file, meta["model"], PROVIDER, meta_extra, results)
        n_succ = sum(1 for r in results if r["success"])
        log.info(f"    saved: {results_file.name}  ({n_succ}/{len(results)} succeeded)")


def main():
    parser = argparse.ArgumentParser(description="Gemini Batch API Runner")
    parser.add_argument("action", choices=["submit", "status", "download", "run"])
    parser.add_argument("--model", required=True,
                        help="Gemini model ID (no default — must be set explicitly)")
    parser.add_argument("--iterations", "-n", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY not set")
        sys.exit(1)

    if args.action == "submit":
        submit_batch(args.model, args.iterations, api_key, args.jpeg_quality)
    elif args.action == "status":
        status_all(api_key)
    elif args.action == "download":
        download_all(api_key)
    elif args.action == "run":
        batch_name = submit_batch(args.model, args.iterations, api_key, args.jpeg_quality)
        log.info(f"Polling for completion (exponential backoff: 30s -> 60s -> 120s -> 300s -> 600s)")
        client = genai.Client(api_key=api_key)
        delay = 30
        while True:
            time.sleep(delay)
            batch = client.batches.get(name=batch_name)
            log.info(f"  state: {batch.state.name}")
            if batch.state.name in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                break
            delay = min(600, int(delay * 2))
        log.info(f"Batch finished: {batch.state.name}")
        if batch.state.name == "JOB_STATE_SUCCEEDED":
            download_all(api_key)


if __name__ == "__main__":
    main()
