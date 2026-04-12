# Test images

Place your food photographs in this folder. Supported formats: `.jpg`, `.jpeg`, `.png`.

The benchmark will automatically:
- Detect all images in this folder
- Resize each to the provider-specific maximum dimension (1568 px for Claude, 2048 px for OpenAI/Gemini) before encoding
- Submit each image to each configured model the requested number of times

For meaningful analysis, you should add at least 5–10 photographs spanning a range of meal types (single-item, composite, low-carb, restaurant dishes, etc.). The original benchmark used 13 photographs.

If you want to compare model accuracy against ground truth, also populate `usda_reference.json` with reference values for each image filename.
