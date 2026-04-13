# LLM Food Benchmark — Academic Submission

Reproducibility and accuracy benchmark for LLM vision API carbohydrate estimation from food photographs. Companion repository for the Diabetologia submission.

## Study

- **26,904 batch API queries** across 4 models (GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro, Gemini 3.1 Pro Preview)
- **13 food photographs**, each submitted 500–560 times per model
- **Primary outcome:** within-image reproducibility (CV, range, distributional normality)
- **Secondary outcome:** accuracy vs reference values stratified by quality tier

## Repository structure

### Batch submission scripts
| Script | Description |
|---|---|
| `batch_common.py` | Shared utilities: image preprocessing, ID mapping, state files, result conversion |
| `openai_batch_runner.py` | OpenAI Batch API runner (submit / status / download) |
| `anthropic_batch_runner.py` | Anthropic Message Batches API runner |
| `gemini_batch_runner.py` | Google Gemini Batch API runner |
| `openai_sequential_submitter.py` | Sequential OpenAI submitter respecting enqueued-token limits |

### Analysis scripts
| Script | Description |
|---|---|
| `build_batch_dataset.py` | Consolidates batch results into a single analysis-ready dataset |
| `deep_dive_batch.py` | Full statistical analysis (CV, MAE, Shapiro-Wilk, Welch's t, Cohen's d, clinical risk translation) |

### Core infrastructure
| File | Description |
|---|---|
| `food_nutrition_benchmark.py` | Prompt, JSON parser, QueryResult schema, real-time runner |
| `usda_reference.json` | Reference carbohydrate values with quality tiers (1=packet label, 2=weighed, 3=portioned, 4=visual estimate) |
| `config.json` | Model and provider configuration |

### Tests
- `tests/test_batch_common.py` — 18 unit tests covering ID mapping, legacy decoders, serialisation, prompt hash

### Test images
- `Test-Images/` — 13 food photographs used in the study

## Known food identification issues

Analysis of food item names across all 26,904 batch queries revealed identification errors in **8 of 13 images**. See Supplementary Material S4 in the Diabetologia submission for the full per-image breakdown. Key findings:

- **Bakewell tart**: Claude 100% "Linzer torte"; only Gemini 3.1 Pro correctly identifies it (99%)
- **Crema catalana**: All 4 models identify as "crème brûlée" (84-100%)
- **Soup ingredients**: No model identifies chorizo or butter beans; all say generic "tomato soup"
- **Stuffed pork loin**: Claude correctly identifies stuffing + pork (MAE 4.8g); GPT-5.4 says "chicken + stuffing" (wrong meat)
- **Pizza**: Gemini models add burrata salad from adjacent plate in 30-74% of queries
- **Churros**: Identification varies; Claude sometimes says "mille crepe cake" in realtime data

Food misidentification — not just portion estimation error — is a significant contributor to carbohydrate estimation failures and is discussed in the Diabetologia manuscript.

## Reproducibility

All queries used an identical prompt (SHA-256: see `batch_common.PROMPT_SHA256`), temperature 0.01, and independent stateless API calls. The complete dataset (26,904 results) is available as a supplementary data file.

## Requirements

```
pip install -r requirements.txt
```

Requires API keys set as environment variables:
- `OPENAI_API_KEY` for GPT-5.4
- `ANTHROPIC_API_KEY` for Claude Sonnet 4.6
- `GOOGLE_API_KEY` for Gemini models

## License

MIT — see LICENSE.

## Citation

Street T (2026). Reproducibility and accuracy of large language model vision APIs for carbohydrate estimation from food photographs: a four-model batch comparison with implications for automated insulin dosing. [Submitted to Diabetologia]
