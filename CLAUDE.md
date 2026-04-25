# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: predict which LLM response users prefer (A wins, B wins, or Tie) from Chatbot Arena human preference data. Metric: log loss.

## Data

- `data/train.csv`: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie
- `data/test.csv`: id, prompt, response_a, response_b
- Target: 3-class (A/B/Tie) — multi-label columns in train, single label in test

Key biases in data:
- **Position bias**: model_a wins more when listed first
- **Verbosity bias**: longer responses win more

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API)
source ~/.venv/kaggle/bin/activate
kaggle competitions download -c llm-classification-finetuning
unzip -d data/ llm-classification-finetuning.zip

# Run training / baselines
python train.py
```

## Code Architecture

Single file: `train.py` — three approach tiers:

| Function | Approach | Status |
|---|---|---|
| `tfidf_baseline` | TF-IDF + LogisticRegression | Working |
| `sbert_baseline` | Sentence-BERT (all-MiniLM-L6-v2) + LogReg | Working |
| `lora_finetune` | Qwen2-0.5B-Instruct + LoRA + PEFT | Not implemented (`raise NotImplementedError`) |

`analyze_biases()` — explores position/verbosity biases in training data.

SBERT approach encodes response_a and response_b separately, then builds features: `[emb_a, emb_b, emb_a - emb_b, cosine_sim]`.

LoRA config in code: r=16, alpha=32, targets q_proj/k_proj/v_proj/o_proj, dropout=0.1.

## Dependencies

PyTorch + transformers + peft + sentence-transformers + scikit-learn + pandas + tqdm + accelerate + bitsandbytes.

## Notes

- Device auto-selects cuda/cpu
- Random seed: 42
- TF-IDF max_features=50k, ngram (1,2)
- SBERT uses batch_size=64