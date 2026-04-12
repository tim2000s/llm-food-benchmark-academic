#!/usr/bin/env python3
"""
Shared utilities for the OpenAI / Anthropic / Gemini batch runners.

This module exists to eliminate the cross-runner code duplication of:
- Image preprocessing (resize, encode, JPEG quality)
- Image discovery
- custom_id encoding / decoding (with reversible mapping in state files)
- State file format
- Result conversion to the common QueryResult schema
- Logging
"""
import base64
import hashlib
import io
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from food_nutrition_benchmark import (
    SYSTEM_PROMPT,
    IMAGES_DIR,
    RESULTS_DIR,
    MAX_IMAGE_DIM,
    DEFAULT_JPEG_QUALITY,
    parse_response,
    QueryResult,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")


# ---------------------------------------------------------------------------
# Constants for state file metadata (so a results file is reproducible)
# ---------------------------------------------------------------------------

PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_to_jpeg_bytes(path: Path, max_dim: int, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Resize an image so longest side <= max_dim, return JPEG bytes.

    JPEG quality is parameterised; default 85 matches the real-time runner.
    """
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()


def discover_images() -> list[Path]:
    """Find all image files in IMAGES_DIR, sorted by filename for determinism."""
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    return sorted(
        (p for p in IMAGES_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name,
    )


def preload_image_data(images: list[Path], provider: str,
                        jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> dict[str, str]:
    """Resize and base64-encode each image once. Returns {filename: base64_string}."""
    max_dim = MAX_IMAGE_DIM[provider]
    out = {}
    for img_path in images:
        b = resize_to_jpeg_bytes(img_path, max_dim, jpeg_quality)
        out[img_path.name] = base64.b64encode(b).decode("ascii")
        log.info(f"  Resized {img_path.name}: {len(b) // 1024}KB")
    return out


# ---------------------------------------------------------------------------
# Reversible custom_id encoding
# ---------------------------------------------------------------------------
# Anthropic requires custom_id to match ^[a-zA-Z0-9_-]{1,64}$.
# OpenAI is more permissive but it's still safer to use a restricted alphabet.
# We use a lossless approach: store an integer index in the state file mapping
# back to the original (image_file, iteration). The custom_id itself is just
# `idx<N>` so we never need filename decoding.


def make_id_mapping(pairs: list[tuple[int, str]]) -> tuple[list[dict], dict]:
    """Build a list of (custom_id, iteration, image_file) records and a mapping
    dict suitable for saving in the state file.

    pairs: list of (iteration, image_file) tuples
    Returns:
        records: list of dicts ready to be used to build requests
                  each has 'custom_id', 'iteration', 'image_file'
        id_map: dict {custom_id: {"iteration": int, "image_file": str}}
                  suitable for saving in the state file
    """
    records = []
    id_map = {}
    for idx, (it, img) in enumerate(pairs):
        cid = f"idx{idx}"
        records.append({
            "custom_id": cid,
            "iteration": it,
            "image_file": img,
        })
        id_map[cid] = {"iteration": it, "image_file": img}
    return records, id_map


def decode_custom_id(custom_id: str, id_map: dict) -> tuple[Optional[int], Optional[str]]:
    """Look up a custom_id in the state-file id_map. Returns (iteration, image_file)
    or (None, None) if not found."""
    entry = id_map.get(custom_id)
    if entry is None:
        return None, None
    return entry["iteration"], entry["image_file"]


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------


def base_state_metadata(model: str, provider: str, images: list[Path],
                         pairs: list[tuple[int, str]],
                         id_map: dict, sub_index: int, n_subbatches: int,
                         jpeg_quality: int) -> dict:
    """Build the standard state-file metadata block. Always include enough
    information to reconstruct the request set without needing the original
    image files."""
    return {
        "provider": provider,
        "model": model,
        "n_images": len(images),
        "images": [p.name for p in images],
        "n_requests": len(pairs),
        "sub_index": sub_index,
        "n_subbatches": n_subbatches,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": PROMPT_SHA256,
        "max_image_dim": MAX_IMAGE_DIM[provider],
        "jpeg_quality": jpeg_quality,
        "id_map": id_map,
    }


def write_state_file(state_dir: Path, fname: str, data: dict) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / fname
    with open(state_file, "w") as f:
        json.dump(data, f, indent=2)
    return state_file


def load_state_file(state_file: Path) -> dict:
    with open(state_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result conversion (shared across all three providers)
# ---------------------------------------------------------------------------


def serialise_query_result(qr: QueryResult, is_batch: bool = True) -> dict:
    """Convert a QueryResult to a JSON-serialisable dict.

    is_batch=True is added so downstream code can distinguish batch vs real-time
    queries on a per-row basis.
    """
    return {
        "model": qr.model,
        "provider": qr.provider,
        "image_file": qr.image_file,
        "iteration": qr.iteration,
        "timestamp": qr.timestamp,
        "latency_s": qr.latency_s,  # may be None for batch
        "success": qr.success,
        "error": qr.error,
        "error_class": qr.error_class,
        "image_type": qr.image_type,
        "brief_description": qr.brief_description,
        "food_items": [asdict(fi) for fi in qr.food_items],
        "input_tokens": qr.input_tokens,
        "output_tokens": qr.output_tokens,
        "coercion_failures": qr.coercion_failures,
        "total_time_with_retries_s": qr.total_time_with_retries_s,
        "n_retries": qr.n_retries,
        "is_batch": is_batch,
    }


def write_results_file(out_path: Path, model: str, provider: str,
                        meta_extra: dict, results: list[dict]) -> None:
    """Write a benchmark results file in the standard schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "run_id": meta_extra.get("run_id"),
            "iterations": meta_extra.get("iterations"),
            "concurrency": "batch",
            "models": {model: {"provider": provider, "display": meta_extra.get("display", model)}},
            "images": meta_extra.get("images", []),
            "image_max_dim": meta_extra.get("image_max_dim"),
            "jpeg_quality": meta_extra.get("jpeg_quality"),
            "prompt_sha256": meta_extra.get("prompt_sha256"),
            "total_queries": len(results),
            "total_success": sum(1 for r in results if r["success"]),
            "elapsed_s": meta_extra.get("elapsed_s", 0),
            "batch_mode": True,
            "batch_provider_id": meta_extra.get("batch_provider_id"),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience: convert a list of (cid, raw_text, error_class, error, usage,
# image_type, ...) into a list of QueryResult dicts
# ---------------------------------------------------------------------------


def build_result_dicts(items: list[dict], model: str, provider: str,
                        id_map: dict) -> list[dict]:
    """Given a list of raw response items, build the standard result-dicts list.

    Each item dict should have:
        custom_id (str)
        raw_text (str | None)        # response text, None on hard failure
        usage (dict | None)          # {"input_tokens": int, "output_tokens": int}
        api_error (str | None)       # API-level error string (e.g. "400 Bad Request")
        api_error_class (str | None) # 'rate_limit', 'transient', 'transport', 'api', or None

    Returns a list of result-dict objects ready to write to the results file.
    """
    out = []
    for item in items:
        cid = item.get("custom_id")
        iteration, image_file = decode_custom_id(cid, id_map)
        if image_file is None:
            log.warning(f"Unknown custom_id: {cid} — skipping")
            continue

        api_error = item.get("api_error")
        if api_error:
            # API-level failure: skip parse, return a synthetic failure record
            qr = QueryResult(
                model=model, provider=provider, image_file=image_file,
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_s=0.0, success=False,
                error=api_error[:500],
                error_class=item.get("api_error_class") or "api",
                input_tokens=(item.get("usage") or {}).get("input_tokens"),
                output_tokens=(item.get("usage") or {}).get("output_tokens"),
            )
        else:
            qr = parse_response(
                raw=item.get("raw_text") or "",
                model=model,
                provider=provider,
                image_file=image_file,
                iteration=iteration,
                latency=0.0,
                usage=item.get("usage"),
            )
            # Latency is genuinely unknown for batch — clear the spurious 0.0
            qr.latency_s = None
        out.append(serialise_query_result(qr, is_batch=True))
    return out
