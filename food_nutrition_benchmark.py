#!/usr/bin/env python3
"""
Food Nutrition AI Benchmark
============================
Tests reproducibility and accuracy of macronutrient estimation across
vision-capable LLM APIs (OpenAI, Anthropic, Google) using food photographs.

The benchmark issues each query as an independent stateless API call:
fresh HTTP client per request, no conversation history, no caching, no
cross-call state. This ensures the model has no memory of previous queries.

The default prompt is adapted from the iAPS open-source automated insulin
delivery system as a representative production prompt for LLM food analysis.

Note on determinism: even at temperature 0.01, none of the providers
guarantee bit-for-bit reproducibility. All three have backend
nondeterminism from sources including mixture-of-experts routing,
batched inference, and floating-point ordering. This benchmark measures
the residual variation observed in practice.

Usage:
    python food_nutrition_benchmark.py [--iterations N] [--parallel P] [--models MODEL,...]

Environment variables:
    OPENAI_API_KEY            OpenAI API key
    ANTHROPIC_API_KEY         Anthropic API key
    GOOGLE_API_KEY            Google AI Studio API key
    BENCHMARK_IMAGES_DIR      Override image directory (default: ./Test-Images)
    BENCHMARK_RESULTS_DIR     Override results directory (default: ./results)
    BENCHMARK_JPEG_QUALITY    JPEG quality for image preprocessing (default: 85)
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Standard library imports above; third-party imports done lazily inside the
# call_* functions to keep the cli help responsive when SDKs are missing.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = Path(os.environ.get("BENCHMARK_IMAGES_DIR", str(BASE_DIR / "Test-Images")))
RESULTS_DIR = Path(os.environ.get("BENCHMARK_RESULTS_DIR", str(BASE_DIR / "results")))
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_ITERATIONS = 100
MAX_ITERATIONS = 10000
DEFAULT_PARALLEL = 5          # concurrent requests per model
DEFAULT_JPEG_QUALITY = int(os.environ.get("BENCHMARK_JPEG_QUALITY", "85"))

# Models to test by default. Override in config.json. The model IDs here
# match the names accepted by each provider's API as of April 2026.
DEFAULT_MODELS = {
    "gpt-5.4":              {"provider": "openai",  "display": "GPT-5.4"},
    "claude-sonnet-4-6":    {"provider": "claude",  "display": "Claude Sonnet 4.6"},
    "gemini-2.5-pro":       {"provider": "gemini",  "display": "Gemini 2.5 Pro"},
}

# Rate-limit defaults (requests-per-minute). Override in config.json.
RATE_LIMITS = {
    "openai":  500,
    "claude":  200,
    "gemini":  300,
}

# Per-provider maximum image dimensions (longest side, pixels). These match
# each provider's documented vision API limits as of April 2026.
#
# IMPORTANT METHODOLOGICAL NOTE: Different providers accept different
# maximum image sizes. Resizing each image to its provider-specific limit
# means the comparison is not strictly apples-to-apples — Claude is given
# a smaller version of the same image than GPT or Gemini. This is a real
# limitation of inter-provider comparisons. To force all providers to use
# the same dimension, set BENCHMARK_COMMON_DIM=1568 (or any other value)
# in the environment.
MAX_IMAGE_DIM = {
    "openai": 2048,
    "claude": 1568,   # Anthropic's documented max for vision
    "gemini": 2048,
}
_common_dim = os.environ.get("BENCHMARK_COMMON_DIM")
if _common_dim:
    _v = int(_common_dim)
    MAX_IMAGE_DIM = {k: _v for k in MAX_IMAGE_DIM}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# Suppress verbose HTTP and gRPC logging
for noisy in ["httpx", "httpcore", "openai", "anthropic", "google", "grpc",
              "google.api_core", "google.generativeai", "absl"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Increase file descriptor limit on POSIX to handle gRPC connection churn
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 10240), hard))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Prompt construction (adapted from iAPS open-source AID app)
# ---------------------------------------------------------------------------

PROMPT_HEADER = """\
You are my diabetes nutrition specialist.
Your primary goal is to estimate the nutrition content of the meal."""

PROMPT_PREFERENCES = """\
USER PREFERENCES:

• NUTRITION_AUTHORITY is United States Department of Agriculture (USDA), FDA Nutrition Facts
• USER_REGION is United States (US)

• DO NOT change standards based on USER_REGION."""

PROMPT_STANDARDS = """\
Use NUTRITION_AUTHORITY for:
  • standard serving sizes
  • nutrition values per 100 grams/milliliters

Standard serving size guidelines:
  • a portion = what is actually present in the image or described by the user.
  • a standard serving = a canonical reference amount defined by NUTRITION_AUTHORITY (slice, piece, cup, 100g, 100ml — NOT derived from the photo).
  • standard serving sizes are reference definitions and MUST NOT be inferred from the photographed or described quantity, count, or total portion size.
  • portions and standard servings must never be conflated.
  • if a reliable standard serving cannot be identified from NUTRITION_AUTHORITY/common standards, you MUST set standard_serving and standard_serving_size to null.
  • always include grams/milliliters per standard serving IF known; otherwise null.

Carbohydrate & fiber convention (ALWAYS EU-STYLE)
\t•\tcarbsPer100 MUST contain only available / digestible carbohydrates that provide about 4 kcal/g.
\t•\tcarbsPer100 MUST NOT include any dietary fiber, even if the country or nutrition authority is the United States.
\t•\tfiberPer100 is always the total dietary fiber in grams.
\t•\tcarbsPer100 and fiberPer100 are two separate values. Fiber is never counted inside carbsPer100.

This convention is global and fixed and does not change with NUTRITION_AUTHORITY or USER_REGION.
Even in US mode, keep fiber separate and excluded from carbsPer100.

Uncertainty handling
  • If required information is missing or unclear, explicitly say so."""

PROMPT_PHOTO_INSTRUCTIONS = """\
DO NOT invent visual details.

After inspecting the image, you MUST first classify the image into exactly one of the following types:
  • food_photo – a photo of prepared food intended for consumption (e.g., plated meal, bowl, snack, packaged food with visible contents).
  • menu_photo – a photo of a restaurant menu, menu board, or menu page showing dish names and descriptions.
  • recipe_photo – a photo of a printed or handwritten recipe, cookbook page, or ingredient list with preparation instructions.

FOR FOOD PHOTOS (food_photo):
  • identify each distinct food component.
  • if the image is ambiguous, choose the most conservative applicable type and explicitly state the uncertainty.
  • for each food, estimate nutrition values per 100 grams/milliliters according to NUTRITION_AUTHORITY.
  • identify the standard serving size (both descriptive and in grams or milliliters), if it is a known NUTRITION_AUTHORITY/common reference; otherwise explicitly return null.
  • estimate portion size on the photo based on visual evidence.
  • do NOT replace the standard serving size with the identified portion size: you must NEVER conflate serving size and portion.
  • do NOT overestimate portions: if you are uncertain about portion sizes, take a more conservative estimate.

FOR MENU PHOTOS (menu_photo):
  • treat descriptions as text only.
  • identify each distinct food component in the menu.
  • detect and identify the language.
  • detect the serving size for each food component according to the menu, if available.
  • if the serving size is not specified, estimate the standard serving size for this food.
  • for each food, estimate nutrition values per 100 grams/milliliters.
  • assume portion size is equal to the serving size specified in the menu or the standard serving size.

FOR RECIPE PHOTOS (recipe_photo):
  • treat descriptions as text only.
  • interpret the listed ingredients and quantities.
  • if quantities are missing or unclear, state the limitation.
  • estimate typical serving size for the recipe.
  • assume portion size is equal to the standard serving size.
  • for each food, estimate nutrition values per 100 grams/milliliters.
  • do not claim to see any visual clues

CRITICAL VALIDATION RULE (must be applied before output):
• if portion_estimate_size == standard_serving_size AND the portion represents
  more or less than one unit (e.g. multiple slices, pieces, servings), this is INVALID.
• in such cases, you MUST correct the standard serving to a reference amount
  (e.g. single slice, 100 g) OR set standard_serving_size to null.
• do NOT reuse or scale down the portion to fabricate a standard serving."""

PROMPT_PRE_RESPONSE = """\
FINAL SELF-CHECK BEFORE RESPONDING:
• Verify that standard_serving_size was NOT derived from portion_estimate_size.
• If both values are equal, explain why this is justified by a reference standard;
  otherwise set standard_serving and standard_serving_size to null.
• It is better to return null than to conflate portion and standard serving."""

RESPONSE_SCHEMA = """\
RESPOND IN JSON FORMAT:
{
  "image_type": "string enum: food_photo or menu_photo or recipe_photo",
  "food_items": [
    {
      "name": "string, required; specific food name; in English",
      "standard_name": "string; concise image-search query for this product. Branded/menu item: include the brand + product name. Generic food: use only the common product name. Use only nouns, plus an optional color. Do not use any other adjectives. Never include rawness, doneness, peel/skin state, serving style, cut form, or texture.",
      "confidence": "decimal 0 to 1; required; confidence for this item",
      "units": "string enum; one of: 'grams' or 'milliliters'; as appropriate for this meal; do NOT translate;",
      "carbs_per_100": "decimal, grams of available / digestible carbohydrates per 100 grams or milliliters",
      "fat_per_100": "decimal, grams of fat per 100 grams or milliliters",
      "fiber_per_100": "decimal, grams of fiber per 100 grams or milliliters",
      "protein_per_100": "decimal, grams of protein per 100 grams or milliliters",
      "sugars_per_100": "decimal, grams of sugars per 100 grams or milliliters",
      "calories_per_100": "decimal, calories per 100 grams or milliliters",
      "portion_estimate_size": "decimal, exact size of the identified portion; in grams or milliliters; do not include unit",
      "standard_serving_size": "decimal, the identified standard serving size in grams or milliliters, if available; do not include unit",
      "standard_serving": "description of the identified standard serving, if available; if natural description is available - do NOT add size in grams/milliliters; in English",
      "glycemic_index": "decimal, glycemic index if available",
      "visual_cues": "visual elements analyzed; in English",
      "preparation_method": "cooking details observed; in English",
      "assessment_notes": "explain how you calculated this specific portion size, what visual references you used for measurement; in English"
    }
  ],
  "overall_description": "describe what you see on the photo; in English",
  "brief_description": "generate a SHORT UI TITLE describing the analyzed food; in English",
  "diabetes_considerations": "carb sources, GI impact (low/medium/high), timing considerations; in English"
}"""


def build_prompt() -> str:
    """Assemble the full system prompt (matches iAPS prompt assembly)."""
    return "\n\n".join([
        PROMPT_HEADER,
        PROMPT_PREFERENCES,
        PROMPT_STANDARDS,
        PROMPT_PHOTO_INSTRUCTIONS,
        PROMPT_PRE_RESPONSE,
        RESPONSE_SCHEMA,
    ])


SYSTEM_PROMPT = build_prompt()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FoodItemResult:
    name: str
    confidence: Optional[float] = None
    portion_estimate_size: Optional[float] = None
    standard_serving_size: Optional[float] = None
    calories_per_100: Optional[float] = None
    carbs_per_100: Optional[float] = None
    fat_per_100: Optional[float] = None
    fiber_per_100: Optional[float] = None
    protein_per_100: Optional[float] = None
    sugars_per_100: Optional[float] = None
    glycemic_index: Optional[float] = None


@dataclass
class QueryResult:
    model: str
    provider: str
    image_file: str
    iteration: int
    timestamp: str
    latency_s: float
    success: bool
    error: Optional[str] = None
    error_class: Optional[str] = None  # 'api', 'parse', 'transport', None
    image_type: Optional[str] = None
    brief_description: Optional[str] = None
    food_items: list = field(default_factory=list)
    raw_response: Optional[str] = None
    # Token usage from provider response (None if unavailable)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Number of fields silently coerced from non-numeric to None during parsing.
    # Useful for tracking JSON adherence quality.
    coercion_failures: int = 0
    # Total time including retries (latency_s is final-attempt only)
    total_time_with_retries_s: Optional[float] = None
    # Number of retries before this final result
    n_retries: int = 0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_image_bytes(path: Path, max_dim: int, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Resize image so longest side <= max_dim, return JPEG bytes.

    The resize and re-encode step introduces a small lossy transformation
    on top of whatever compression the source image already had. JPEG
    quality is parameterised; the default of 85 is a reasonable balance
    between file size and detail preservation.
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


def load_image_base64(path: Path, provider: str = "openai") -> str:
    """Read an image, resize for the target provider, and return base64."""
    max_dim = MAX_IMAGE_DIM.get(provider, 2048)
    img_bytes = resize_image_bytes(path, max_dim)
    return base64.standard_b64encode(img_bytes).decode("ascii")


def image_media_type(path: Path) -> str:
    """All images become JPEG after resize."""
    return "image/jpeg"


# ---------------------------------------------------------------------------
# Provider API calls (uses official SDKs)
# ---------------------------------------------------------------------------


async def call_openai(model: str, image_b64: str, media_type: str, api_key: str) -> tuple[str, dict]:
    """Call OpenAI vision API and return (raw_text, usage_dict).

    Each call creates a fresh client with no shared session/history,
    ensuring every request is fully independent and stateless.
    """
    import openai
    import httpx
    async with openai.AsyncOpenAI(
        api_key=api_key,
        http_client=httpx.AsyncClient(),
    ) as client:
        data_uri = f"data:{media_type};base64,{image_b64}"
        resp = await client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }],
            max_output_tokens=6000,
            temperature=0.01,
            store=False,  # do not store for training or retrieval
        )
        usage = {}
        if hasattr(resp, "usage") and resp.usage is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return resp.output_text, usage


async def call_claude(model: str, image_b64: str, media_type: str, api_key: str) -> tuple[str, dict]:
    """Call Anthropic Claude vision API and return (raw_text, usage_dict)."""
    import anthropic
    import httpx
    async with anthropic.AsyncAnthropic(
        api_key=api_key,
        http_client=httpx.AsyncClient(),
    ) as client:
        resp = await client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.01,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    }},
                ],
            }],
        )
        usage = {}
        if hasattr(resp, "usage") and resp.usage is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", None),
                "output_tokens": getattr(resp.usage, "output_tokens", None),
            }
        return resp.content[0].text, usage


async def call_gemini(model: str, image_b64: str, media_type: str, api_key: str) -> tuple[str, dict]:
    """Call Google Gemini vision API and return (raw_text, usage_dict).

    Uses REST transport to avoid gRPC channel/file-descriptor leaks under
    high query volumes.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")
    gmodel = genai.GenerativeModel(model)
    image_bytes = base64.standard_b64decode(image_b64)
    resp = await asyncio.to_thread(
        gmodel.generate_content,
        [
            SYSTEM_PROMPT,
            {"mime_type": media_type, "data": image_bytes},
        ],
        generation_config=genai.types.GenerationConfig(
            temperature=0.01,
            top_p=0.95,
            top_k=8,
            max_output_tokens=8000,
        ),
    )
    usage = {}
    if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
        usage = {
            "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", None),
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", None),
        }
    return resp.text, usage


PROVIDER_FUNCS = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini,
}

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json(raw: str) -> dict:
    """Extract a JSON object from a (possibly markdown-fenced) LLM response.

    Strategy (tried in order):
    1. Strip markdown fences and try parsing the entire string as JSON
    2. Find a balanced { ... } block by counting braces (skipping strings
       and escapes)
    3. Fall back to first '{' to last '}' substring (legacy heuristic)

    Raises ValueError if no parseable JSON object is found.
    """
    cleaned = raw.replace("```json", "").replace("```", "").replace("`", "").strip()

    # 1. Try parsing the whole cleaned string
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Find a balanced JSON object by counting braces
    def find_balanced_object(text: str) -> Optional[str]:
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_string = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i:j + 1]
            i += 1
        return None

    candidate = find_balanced_object(cleaned)
    if candidate:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Legacy fallback: first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        # Try to fix common issues: trailing commas before }
        candidate = cleaned[start:end + 1]
        candidate = re.sub(r",(\s*[\]}])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError("No JSON object found in response")


def safe_float(val) -> tuple[Optional[float], bool]:
    """Convert a value to float; return (value, was_coerced).

    was_coerced is True if the input was non-None but could not be converted
    (i.e. data was silently dropped). This lets callers track JSON adherence
    quality at the field level.
    """
    if val is None:
        return None, False
    try:
        return float(val), False
    except (ValueError, TypeError):
        return None, True


def parse_response(raw: str, model: str, provider: str, image_file: str,
                   iteration: int, latency: float,
                   usage: Optional[dict] = None,
                   total_time_with_retries: Optional[float] = None,
                   n_retries: int = 0) -> QueryResult:
    """Parse a raw LLM response into a QueryResult."""
    ts = datetime.now(timezone.utc).isoformat()
    usage = usage or {}
    try:
        data = extract_json(raw)
        items = []
        coercion_failures = 0
        for fi in data.get("food_items", []):
            field_values = {}
            for fname in ["confidence", "portion_estimate_size", "standard_serving_size",
                          "calories_per_100", "carbs_per_100", "fat_per_100",
                          "fiber_per_100", "protein_per_100", "sugars_per_100",
                          "glycemic_index"]:
                v, coerced = safe_float(fi.get(fname))
                field_values[fname] = v
                if coerced:
                    coercion_failures += 1
            items.append(FoodItemResult(
                name=fi.get("name", "unknown"),
                **field_values,
            ))
        return QueryResult(
            model=model, provider=provider, image_file=image_file,
            iteration=iteration, timestamp=ts, latency_s=round(latency, 3),
            success=True, image_type=data.get("image_type"),
            brief_description=data.get("brief_description"),
            food_items=items, raw_response=raw,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            coercion_failures=coercion_failures,
            total_time_with_retries_s=round(total_time_with_retries, 3) if total_time_with_retries else None,
            n_retries=n_retries,
        )
    except Exception as e:
        return QueryResult(
            model=model, provider=provider, image_file=image_file,
            iteration=iteration, timestamp=ts, latency_s=round(latency, 3),
            success=False, error=str(e),
            error_class="parse",
            raw_response=raw,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_time_with_retries_s=round(total_time_with_retries, 3) if total_time_with_retries else None,
            n_retries=n_retries,
        )


# ---------------------------------------------------------------------------
# Rate-limited concurrent executor
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token-bucket rate limiter (requests per minute).

    NOTE: This serialises requests through the `acquire()` lock and updates
    the timestamp at the end of acquire. The achieved rate is therefore
    `1 / (interval + acquire_time)` rather than exactly `1 / interval`.
    For benchmarking purposes this is conservative — the actual rate is
    slightly under the configured RPM.
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


def classify_error(err_str: str) -> str:
    """Classify an error string into 'rate_limit', 'transient', 'api', or 'transport'."""
    s = err_str.lower()
    if "429" in err_str or "rate_limit" in s or "resource_exhausted" in s or "overloaded" in s:
        return "rate_limit"
    if any(code in err_str for code in ["500", "502", "503", "529"]) or "internal" in s:
        return "transient"
    if "400" in err_str or "401" in err_str or "403" in err_str or "404" in err_str:
        return "api"
    if "connection" in s or "timeout" in s or "broken pipe" in s:
        return "transport"
    return "api"


async def run_single_query(
    sem: asyncio.Semaphore,
    limiter: RateLimiter,
    model_id: str,
    provider: str,
    image_b64: str,
    media_type: str,
    api_key: str,
    image_file: str,
    iteration: int,
    max_retries: int = 5,
) -> QueryResult:
    """Execute one API call with concurrency and rate limiting.

    Retries on rate-limit and transient errors with exponential backoff.

    NOTE: When a query succeeds after one or more retries, latency_s reflects
    only the latency of the final (successful) attempt. The total time
    including all retries is recorded separately as total_time_with_retries_s.
    """
    func = PROVIDER_FUNCS[provider]
    async with sem:
        await limiter.acquire()
        overall_start = time.monotonic()
        n_retries = 0
        last_err_class = None
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                result = await func(model_id, image_b64, media_type, api_key)
                # call_* now returns (raw_text, usage_dict)
                if isinstance(result, tuple) and len(result) == 2:
                    raw, usage = result
                else:
                    raw, usage = result, {}
                latency = time.monotonic() - t0
                total_time = time.monotonic() - overall_start
                return parse_response(raw, model_id, provider, image_file, iteration,
                                       latency, usage=usage,
                                       total_time_with_retries=total_time,
                                       n_retries=n_retries)
            except Exception as e:
                latency = time.monotonic() - t0
                err_str = str(e)
                err_class = classify_error(err_str)
                last_err_class = err_class
                if attempt < max_retries and err_class in ("rate_limit", "transient", "transport"):
                    n_retries += 1
                    wait = (2 ** attempt) * 2 + (attempt * 3 if provider == "gemini" else 0)
                    log.warning(
                        f"  {model_id} img={image_file} iter={iteration} "
                        f"retry {attempt + 1} after {wait}s [{err_class}]: {err_str[:120]}"
                    )
                    await asyncio.sleep(wait)
                    continue
                total_time = time.monotonic() - overall_start
                return QueryResult(
                    model=model_id, provider=provider, image_file=image_file,
                    iteration=iteration,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_s=round(latency, 3),
                    success=False, error=err_str[:500],
                    error_class=err_class,
                    total_time_with_retries_s=round(total_time, 3),
                    n_retries=n_retries,
                )


async def run_model_image_batch(
    model_id: str,
    provider: str,
    display_name: str,
    api_key: str,
    image_path: Path,
    iterations: int,
    concurrency: int,
    rpm: int,
) -> list[QueryResult]:
    """Run all iterations for one model + one image."""
    image_b64 = load_image_base64(image_path, provider=provider)
    media_type = image_media_type(image_path)
    sem = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(rpm)
    image_file = image_path.name

    log.info(f"  [{display_name}] Starting {iterations} queries for {image_file}")

    tasks = [
        run_single_query(sem, limiter, model_id, provider, image_b64, media_type,
                         api_key, image_file, i)
        for i in range(1, iterations + 1)
    ]
    results = await asyncio.gather(*tasks)

    successes = sum(1 for r in results if r.success)
    log.info(f"  [{display_name}] {image_file}: {successes}/{iterations} succeeded")
    return list(results)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load config.json if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def get_api_key(provider: str, config: dict) -> Optional[str]:
    """Resolve API key from config or environment."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    key = config.get("api_keys", {}).get(provider)
    if key:
        return key
    return os.environ.get(env_map.get(provider, ""))


def discover_images() -> list[Path]:
    """Find all image files in IMAGES_DIR, sorted by filename for determinism."""
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    images = sorted(
        (p for p in IMAGES_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name,
    )
    return images


def serialise_results(results: list[QueryResult]) -> list[dict]:
    """Convert results to JSON-serialisable dicts."""
    out = []
    for r in results:
        d = {
            "model": r.model,
            "provider": r.provider,
            "image_file": r.image_file,
            "iteration": r.iteration,
            "timestamp": r.timestamp,
            "latency_s": r.latency_s,
            "success": r.success,
            "error": r.error,
            "error_class": r.error_class,
            "image_type": r.image_type,
            "brief_description": r.brief_description,
            "food_items": [asdict(fi) for fi in r.food_items],
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "coercion_failures": r.coercion_failures,
            "total_time_with_retries_s": r.total_time_with_retries_s,
            "n_retries": r.n_retries,
        }
        out.append(d)
    return out


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file's contents (for reproducibility audit)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def run_benchmark(
    iterations: int,
    concurrency: int,
    model_filter: Optional[list[str]] = None,
):
    config = load_config()
    models = config.get("models", DEFAULT_MODELS)

    # Filter models if specified
    if model_filter:
        models = {k: v for k, v in models.items() if k in model_filter or v.get("display", "") in model_filter}

    if not models:
        log.error("No models configured. Check config.json or --models flag.")
        sys.exit(1)

    images = discover_images()
    if not images:
        log.error(f"No images found in {IMAGES_DIR}. Add food images (.jpg/.png) and re-run.")
        sys.exit(1)

    log.info(f"Benchmark: {iterations} iterations x {len(images)} images x {len(models)} models")
    log.info(f"Images: {[p.name for p in images]}")
    log.info(f"Models: {list(models.keys())}")

    # Validate API keys
    providers_needed = set(v["provider"] for v in models.values())
    api_keys = {}
    for prov in providers_needed:
        key = get_api_key(prov, config)
        if not key:
            log.error(f"No API key for {prov}. Set env var or add to config.json.")
            sys.exit(1)
        api_keys[prov] = key

    # Determine rate limits and per-provider concurrency
    rate_limits = config.get("rate_limits", RATE_LIMITS)
    concurrency_limits = config.get("concurrency", {})

    all_results: list[QueryResult] = []
    run_start = time.monotonic()

    # Launch all model+image combos as concurrent tasks
    batch_tasks = []
    for model_id, model_info in models.items():
        provider = model_info["provider"]
        display = model_info.get("display", model_id)
        rpm = rate_limits.get(provider, 60)
        provider_concurrency = concurrency_limits.get(provider, concurrency)
        api_key = api_keys[provider]
        for img_path in images:
            batch_tasks.append(
                run_model_image_batch(
                    model_id, provider, display, api_key, img_path,
                    iterations, provider_concurrency, rpm,
                )
            )

    batch_results = await asyncio.gather(*batch_tasks)
    for batch in batch_results:
        all_results.extend(batch)

    elapsed = time.monotonic() - run_start
    total_success = sum(1 for r in all_results if r.success)
    log.info(f"Benchmark complete: {total_success}/{len(all_results)} succeeded in {elapsed:.0f}s")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"results_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump({
            "meta": {
                "run_id": run_id,
                "iterations": iterations,
                "concurrency": concurrency,
                "models": models,
                "images": [p.name for p in images],
                "image_max_dim": MAX_IMAGE_DIM,
                "jpeg_quality": DEFAULT_JPEG_QUALITY,
                "total_queries": len(all_results),
                "total_success": total_success,
                "elapsed_s": round(elapsed, 1),
            },
            "results": serialise_results(all_results),
        }, f, indent=2)
    log.info(f"Results saved to {results_file}")
    log.info(f"Results SHA256: {sha256_file(results_file)}")
    log.info(f"To analyse: python3 deep_dive_realtime.py --data {results_file}")

    return results_file


def main():
    parser = argparse.ArgumentParser(description="Food Nutrition AI Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Iterations per image per model (default {DEFAULT_ITERATIONS}, max {MAX_ITERATIONS})")
    parser.add_argument("--parallel", "-p", type=int, default=DEFAULT_PARALLEL,
                        help=f"Concurrent requests per model (default {DEFAULT_PARALLEL})")
    parser.add_argument("--models", "-m", type=str, default=None,
                        help="Comma-separated model IDs to test (default: all)")
    args = parser.parse_args()

    iterations = min(max(1, args.iterations), MAX_ITERATIONS)
    concurrency = max(1, args.parallel)
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    asyncio.run(run_benchmark(iterations, concurrency, model_filter))


if __name__ == "__main__":
    main()
