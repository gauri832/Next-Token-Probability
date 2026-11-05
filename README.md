---
title: Next-Token Probability Lab
emoji: 🎲
sdk: gradio
app_file: app.py
pinned: false
---

# Next-Token Probability Lab (P5)

Shows how GPT-2 picks each next word and how temperature, top-k and top-p change the output.
Individual CA project, Generative AI, LY M.Sc. AI & ML.

**Live app:** <PASTE URL> | **Demo video:** <PASTE LINK>

## Features (tabs)
1. Next-token probabilities with temperature / top-k / top-p sliders (raw vs adjusted bars)
2. Temperature 0.2 vs 1.2 side by side, with samples
3. Text generation + text-to-image, toxic-token filter, optional few-shot prefix
4. Self-attention heatmap
5. BERT masked-word vs GPT-2 next-word
6. Measured results (from `results.json`)

## Syllabus mapping
| Unit | Where |
|---|---|
| I Foundations | Probability chart, entropy, zero- vs few-shot comparison (`evaluate.py`), lifecycle diagram in report |
| II GANs & LLMs | GPT-2 Transformer, attention tab, prompt-based learning comparison |
| III GPT/BERT/multimodal | GPT-2 generation, BERT fill-mask, text-to-image |
| IV Tools | PyTorch hand-written sampler (`core.adjust`), char-LSTM, Gradio, sd-turbo |
| V Responsible AI | Factual-error vs temperature, toxic-token filter, licences, `MODEL_CARD.md` |

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Reproduce the measurements
```bash
pip install -r requirements-eval.txt
python evaluate.py          # writes results.json (use --quick for a smoke test)
python build_report.py --name "Your Name" --roll 30 --url <APP_URL> --repo <REPO_URL>
```

## Deploy on Hugging Face Spaces
1. huggingface.co > New Space > SDK **Gradio** > CPU basic (free).
2. Upload all files (or `git push` to the Space repo), including `results.json`.
3. Optional image model: Space > Settings > **Secrets** > add `HF_TOKEN` (never commit it). Without it the image step falls back to a text card.
4. Wait for the build, open the public URL, test every tab.


## Submission status (5 Nov 2026)

The bundled `results.json` is explicitly synthetic placeholder data and is not a real evaluation. Run `python evaluate.py` and replace it with genuine output before reporting results. The application has not been verified against downloaded models or deployed. Add your Hugging Face Space and GitHub URLs after deployment.
