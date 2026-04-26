# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: predict which LLM response users prefer (A wins, B wins, or Tie) from Chatbot Arena human preference data. Metric: log loss.

Target: log_loss < 0.85.

## Data

- `data/train.csv`: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie
- `data/test.csv`: id, prompt, response_a, response_b
- Target: 3-class (A/B/Tie) — multi-label columns in train, single label in test

Key biases:
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

# Run the full pipeline (train.py — local/Colab)
python train.py

# Kaggle notebook submission
# notebooks/kaggle_submission.ipynb — T4/P100 GPU, TPU fallback
```

## Code Architecture

Two pipelines exist:

### `train.py` — Local/Colab pipeline (larger model)
- TF-IDF + LogReg → SBERT + LogReg → DeBERTa-v3-base (LoRA not implemented) → Ensemble
- `deberta_finetune()`: MAX_LENGTH=512, BATCH_SIZE=8, GRAD_ACCUM=4, AMP, label smoothing
- Grid search ensemble weights per model count

### `notebooks/kaggle_submission.ipynb` — Kaggle submission pipeline (optimized for GPU)
- Phase 1: TF-IDF + LogReg (5-fold CV OOF)
- Phase 2: SBERT (all-MiniLM-L6-v2) + LogReg + optional TabPFN (5-fold CV OOF)
- Phase 3: DeBERTa-v3-small finetuning (position flip, BTD auxiliary loss, pure FP32)
- Phase 4: Weighted ensemble grid search

Pipeline flow: each phase produces OOF predictions + test predictions, then weighted average optimized on OOF log_loss.

### Key classes/functions

- `ComparisonDataset`: PyTorch Dataset with 50% position randomization during training
- `encode_pairs()`: builds SBERT features [emb_a, emb_b, emb_prompt, diff, cos_sim, handcrafted] (~1158 dims with prompt encoding)
- Ensemble grid search: dynamic for N models, step size 0.1, per-model row normalization before weighting

## Notebook Config (kaggle_submission.ipynb)

| Parameter | Value |
|-----------|-------|
| MODEL_NAME | microsoft/deberta-v3-small |
| MAX_LENGTH | 512 |
| BATCH_SIZE | 16 |
| GRAD_ACCUM | 2 |
| EPOCHS | 3 |
| DeBERTa LR | 1e-6 backbone, 3e-7 head |
| SBERT | all-MiniLM-L6-v2 |
| TF-IDF | max_features=50k, ngram (1,2) |
| TabPFN limit | 10000 rows, 2000 features |

## Dependencies

PyTorch + transformers + peft + sentence-transformers + scikit-learn + pandas + tqdm + accelerate + bitsandbytes + tabpfn.

## Notes

- Device auto-selects cuda/cpu, TPU detection via torch_xla
- Random seed: 42
- DeBERTa OOF is partial (last 10% validation split, not full CV)
- BTD auxiliary loss weight: 0.1 (in notebook cell-16)
- TabPFN skipped when features > 2000 (SBERT with prompt encoding produces ~1158)
