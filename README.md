# LLM Classification Finetuning

Kaggle competition: predict which LLM response users prefer (A wins, B wins, or Tie) from Chatbot Arena human preference data. Metric: log loss.

## Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Kaggle Competition Workflow

### 1. Authenticate with Kaggle

```bash
# Install kaggle API
uv pip install kaggle

# Create Kaggle API token (do once):
# 1. Go to https://www.kaggle.com/account → API → "Create New Token"
# 2. Download kaggle.json to ~/.config/kaggle/kaggle.json
#    (or ~/.kaggle/kaggle.json on older setups)

# Verify auth
kaggle competitions list
```

### 2. Download Competition Data

```bash
# Download data to data/ directory
kaggle competitions download -c llm-classification-finetuning -p data/
unzip -d data/ llm-classification-finetuning.zip
```

Files:
- `data/train.csv` — training data with labels
- `data/test.csv` — test data for submission

### 3. Train Model

```bash
# Activate environment
source .venv/bin/activate

# Run training (all phases: TF-IDF, SBERT, DeBERTa, ensemble)
python train.py
```

Output: `output/submission.csv`

### 4. Submit to Kaggle

```bash
# Check current leaderboard position
kaggle competitions leaderboard llm-classification-finetuning -p

# Submit predictions
kaggle competitions submit llm-classification-finetuning \
    -f output/submission.csv \
    -m "Phase 1: TF-IDF + SBERT + DeBERTa ensemble"

# Check submission status
kaggle competitions submissions llm-classification-finetuning
```

## Project Structure

```
llm-classification-ft/
├── CLAUDE.md          # Claude Code guidance
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── train.py           # Training script (all phases)
├── data/              # Competition data (download via Kaggle API)
└── output/            # Generated submissions
```

## Training Pipeline

| Phase | Approach | Target Log Loss |
|-------|----------|----------------|
| 1 | Enhanced TF-IDF + LogReg | < 1.0 |
| 2 | SBERT (all-MiniLM-L6-v2) + LogReg | < 0.95 |
| 3 | DeBERTa-v3-base finetuning | < 0.87 |
| 4 | Ensemble (weighted average) | < 0.84 |

Key techniques:
- Position bias mitigation (randomize response order during training)
- Label smoothing (0.05)
- Mixed precision (fp16)
- Cosine LR scheduler with warmup
- Temperature scaling calibration

## Competition

[Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning) | [ArXiv](https://arxiv.org/pdf/2603.04409)

## Key Biases

- **Position bias**: A wins more when listed first
- **Verbosity bias**: Longer responses win more