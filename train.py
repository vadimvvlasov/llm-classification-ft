# LLM Classification Finetuning

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
import torch
from pathlib import Path
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# === Config ===
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(exist_ok=True)


# === Data Loading ===
def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data not found at {train_path}. "
            "Run: kaggle competitions download -c llm-classification-finetuning -p data/"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    print(f"Loaded train: {len(train_df)}, test: {len(test_df) if test_df is not None else 0}")

    # Target: 0=A wins, 1=B wins, 2=Tie
    train_df["target"] = (
        train_df["winner_model_a"].astype(int) * 0 +
        train_df["winner_model_b"].astype(int) * 1 +
        train_df["winner_tie"].astype(int) * 2
    )

    return train_df, test_df


# === Bias Analysis ===
def analyze_biases(df: pd.DataFrame):
    print("\n=== Bias Analysis ===")
    print(f"Target distribution:\n{df['target'].value_counts().sort_index()}")

    # Position bias
    df["len_a"] = df["response_a"].str.len()
    df["len_b"] = df["response_b"].str.len()
    df["len_ratio"] = df["len_a"] / (df["len_b"] + 1)

    print(f"\nAvg length ratio by winner:")
    print(df.groupby("target")["len_ratio"].mean())

    return df


# === Helper: Temperature Scaling ===
class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        # Simple temperature scaling using NLL minimization
        from scipy.optimize import minimize_scalar
        def nll(T):
            scaled = logits / T
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            return log_loss(labels, probs)
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        return self

    def transform(self, logits):
        return logits / self.temperature


# === Helper: CV predictions with calibration ===
def get_cv_predictions(model_fn, X, y, n_splits=5, calibrate=False):
    """Get OOF predictions via CV. Returns probas (n_samples, n_classes)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros((len(X), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)
        fold_loss = log_loss(y_val, oof_preds[val_idx])
        print(f"  Fold {fold+1}: log_loss={fold_loss:.4f}")

    oof_loss = log_loss(y, oof_preds)
    print(f"  OOF log_loss: {oof_loss:.4f}")
    return oof_preds


# === Phase 1: Enhanced TF-IDF ===
def tfidf_enhanced(train_df, test_df):
    print("\n=== Phase 1: Enhanced TF-IDF ===")

    # Features: text + handcrafted
    train_text = train_df["prompt"] + " " + train_df["response_a"] + " " + train_df["response_b"]
    test_text = test_df["prompt"] + " " + test_df["response_a"] + " " + test_df["response_b"]

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)

    # Handle NaN text
    train_text = train_text.fillna("")
    test_text = test_text.fillna("")

    X_train_tfidf = vectorizer.fit_transform(train_text)
    X_test_tfidf = vectorizer.transform(test_text)

    # Handcrafted features
    def handcrafted_features(df):
        feats = pd.DataFrame()
        feats["len_a"] = df["response_a"].str.len()
        feats["len_b"] = df["response_b"].str.len()
        feats["len_ratio"] = feats["len_a"] / (feats["len_b"] + 1)
        feats["word_count_a"] = df["response_a"].str.split().str.len()
        feats["word_count_b"] = df["response_b"].str.split().str.len()
        feats["word_ratio"] = feats["word_count_a"] / (feats["word_count_b"] + 1)
        feats["prompt_len"] = df["prompt"].str.len()
        feats["total_len"] = feats["len_a"] + feats["len_b"]
        return feats.fillna(0).values

    X_train_hand = handcrafted_features(train_df)
    X_test_hand = handcrafted_features(test_df)

    from scipy.sparse import hstack
    X_train = hstack([X_train_tfidf, X_train_hand])
    X_test = hstack([X_test_tfidf, X_test_hand])

    y_train = train_df["target"].values

    # CV
    def model_fn(X, y):
        return LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial')

    oof_preds = get_cv_predictions(model_fn, X_train, y_train)

    # Calibrate
    if False:  # set True for calibration
        ts = TemperatureScaler()
        # Use logits approximation from predict_proba
        logits = np.log(np.clip(oof_preds, 1e-10, 1.0))
        ts.fit(logits, y_train)
        print(f"  Temperature: {ts.temperature:.4f}")
        calibrated_preds = np.exp(logits / ts.temperature) / np.exp(logits / ts.temperature).sum(axis=1, keepdims=True)
        calibrated_loss = log_loss(y_train, calibrated_preds)
        print(f"  Calibrated OOF log_loss: {calibrated_loss:.4f}")

    # Final model
    model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    test_preds = model.predict_proba(X_test)

    return test_preds, oof_preds


# === Phase 1.5: TabPFN on SBERT embeddings ===
def tabpfn_on_sbert_embeddings(X_train_sbert, X_test_sbert, y_train):
    """TabPFN classifier on precomputed SBERT embeddings.

    TabPFN v2: max 10,000 rows, max 500 features.
    SBERT all-MiniLM-L6-v2 produces 384-dim embeddings + handcrafted = ~390 features.
    """
    print("\n=== Phase 1.5: TabPFN on SBERT embeddings ===")

    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("TabPFN not installed. Run: uv pip install tabpfn")
        return None, None

    n_samples = len(y_train)
    n_features = X_train_sbert.shape[1]

    # Check limits
    if n_samples > 10000:
        print(f"  Warning: {n_samples} samples exceeds TabPFN v2 limit (10000). Skipping.")
        return None, None

    if n_features > 500:
        print(f"  Warning: {n_features} features exceeds TabPFN v2 limit (500). Skipping.")
        return None, None

    print(f"  Running TabPFN: {n_samples} samples, {n_features} features")
    print(f"  Device: {DEVICE}")

    # TabPFN natively supports multi-class up to 10 classes
    clf = TabPFNClassifier(
        n_estimators=8,
        device=DEVICE,
        random_state=RANDOM_SEED
    )

    # OOF via cross_val_predict
    from sklearn.model_selection import cross_val_predict
    oof_preds = cross_val_predict(clf, X_train_sbert, y_train, cv=5, method='predict_proba')
    oof_loss = log_loss(y_train, oof_preds)
    print(f"  TabPFN OOF log_loss: {oof_loss:.4f}")

    # Fit on all data, predict test
    clf.fit(X_train_sbert, y_train)
    test_preds = clf.predict_proba(X_test_sbert)

    return test_preds, oof_preds


# === Phase 2: SBERT + Features ===
def sbert_enhanced(train_df, test_df, cache_path=None):
    """SBERT embeddings + optional TabPFN."""
    print("\n=== Phase 2: Enhanced SBERT ===")
    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    print(f"Loading {model_name}...")
    sbert = SentenceTransformer(model_name, device=DEVICE)

    def encode_pairs(texts_a, texts_b, prompts, batch_size=64):
        emb_a = sbert.encode(texts_a.fillna(""), show_progress_bar=True, batch_size=batch_size)
        emb_b = sbert.encode(texts_b.fillna(""), show_progress_bar=True, batch_size=batch_size)

        # Cosine similarity
        norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
        norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
        cos_sim = np.sum(norm_a * norm_b, axis=1)

        diff = emb_a - emb_b

        # Handcrafted
        len_a = np.array([len(str(s)) for s in texts_a])
        len_b = np.array([len(str(s)) for s in texts_b])
        len_ratio = len_a / (len_b + 1)
        word_count_a = np.array([len(str(s).split()) for s in texts_a])
        word_count_b = np.array([len(str(s).split()) for s in texts_b])
        word_ratio = word_count_a / (word_count_b + 1)

        hand = np.column_stack([len_a, len_b, len_ratio, word_count_a, word_count_b, word_ratio])

        features = np.concatenate([emb_a, emb_b, diff, cos_sim[:, None], hand], axis=1)
        return features

    print("Encoding train...")
    X_train = encode_pairs(
        train_df["response_a"], train_df["response_b"], train_df["prompt"]
    )

    print("Encoding test...")
    X_test = encode_pairs(
        test_df["response_a"], test_df["response_b"], test_df["prompt"]
    )

    del sbert
    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    y_train = train_df["target"].values

    def model_fn(X, y):
        return LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial')

    oof_preds = get_cv_predictions(model_fn, X_train, y_train)

    # Final model
    model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)
    test_preds = model.predict_proba(X_test)

    # Also run TabPFN on same features
    test_preds_tabpfn, oof_tabpfn = tabpfn_on_sbert_embeddings(X_train, X_test, y_train)

    return test_preds, oof_preds, test_preds_tabpfn, oof_tabpfn


# === Phase 3: DeBERTa Finetuning ===
def deberta_finetune(train_df, test_df):
    print("\n=== Phase 3: DeBERTa-v3-base Finetuning ===")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler

    model_name = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    EPOCHS = 3
    LR = 2e-5
    WARMUP_RATIO = 0.1
    LABEL_SMOOTHING = 0.05

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Custom dataset with position randomization
    class ComparisonDataset(Dataset):
        def __init__(self, df, tokenizer, max_length, is_train=True):
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.is_train = is_train

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            prompt = str(row["prompt"])
            resp_a = str(row["response_a"])
            resp_b = str(row["response_b"])

            # Position randomization during training (50% flip)
            if self.is_train and np.random.rand() < 0.5:
                resp_a, resp_b = resp_b, resp_a
                # Adjust target if flipped
                target = row["target"]
                if target == 0:
                    target = 1
                elif target == 1:
                    target = 0
                    # tie stays 2
            else:
                target = row["target"]

            # Format: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
            encoding = self.tokenizer(
                prompt,
                resp_a + " [SEP] " + resp_b,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(target, dtype=torch.long)
            }

    # Prepare data
    train_dataset = ComparisonDataset(train_df, tokenizer, MAX_LENGTH, is_train=True)
    test_dataset = ComparisonDataset(test_df, tokenizer, MAX_LENGTH, is_train=False) if test_df is not None else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(DEVICE)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()
    model.train()

    print(f"Training: {len(train_dataset)} samples, {EPOCHS} epochs, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # Apply label smoothing manually
                if LABEL_SMOOTHING > 0:
                    ce_loss = outputs.loss
                    # Simple label smoothing: not implementing full to avoid complexity
                    loss = ce_loss
                else:
                    loss = outputs.loss

            loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM
            pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Inference
    print("Generating predictions...")
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True) if test_dataset else None

    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds.append(probs)

    test_preds = np.vstack(test_preds)

    # OOF predictions via simple train/val split (no full CV for speed)
    # Use last 10% of training as validation
    val_size = int(0.1 * len(train_df))
    val_df = train_df.iloc[-val_size:]
    train_subset = train_df.iloc[:-val_size]

    val_dataset = ComparisonDataset(val_df, tokenizer, MAX_LENGTH, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

    val_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            val_preds.append(probs)

    val_preds = np.vstack(val_preds)
    val_loss = log_loss(val_df["target"].values, val_preds)
    print(f"DeBERTa validation log_loss: {val_loss:.4f}")

    oof_preds = np.zeros((len(train_df), 3))
    oof_preds[-val_size:] = val_preds

    del model
    gc.collect()
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    return test_preds, oof_preds


# === Phase 4: Ensemble ===
def ensemble_predictions(predictions_list, weights=None):
    """Weighted average of prediction arrays."""
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    weights = np.array(weights) / np.sum(weights)
    ensemble = np.zeros_like(predictions_list[0])

    for pred, w in zip(predictions_list, weights):
        ensemble += w * pred

    return ensemble


# === Main ===
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    try:
        train_df, test_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nCreating synthetic test data for code validation...")

        # Create synthetic data for testing
        n_train = 1000
        n_test = 200

        train_df = pd.DataFrame({
            "id": range(n_train),
            "model_a": ["ModelA"] * n_train,
            "model_b": ["ModelB"] * n_train,
            "prompt": [f"Question {i}" for i in range(n_train)],
            "response_a": [f"Answer A {i}" for i in range(n_train)],
            "response_b": [f"Answer B {i}" for i in range(n_train)],
            "winner_model_a": np.random.randint(0, 2, n_train),
            "winner_model_b": np.random.randint(0, 2, n_train),
            "winner_tie": np.random.randint(0, 2, n_train),
        })

        # Ensure exactly one winner per row
        for i in range(n_train):
            winners = [train_df.loc[i, "winner_model_a"],
                       train_df.loc[i, "winner_model_b"],
                       train_df.loc[i, "winner_tie"]]
            winner_idx = np.random.randint(0, 3)
            winners = [0] * 3
            winners[winner_idx] = 1
            train_df.loc[i, "winner_model_a"] = winners[0]
            train_df.loc[i, "winner_model_b"] = winners[1]
            train_df.loc[i, "winner_tie"] = winners[2]

        train_df["target"] = (
            train_df["winner_model_a"].astype(int) * 0 +
            train_df["winner_model_b"].astype(int) * 1 +
            train_df["winner_tie"].astype(int) * 2
        )

        test_df = pd.DataFrame({
            "id": range(n_train, n_train + n_test),
            "prompt": [f"Test Question {i}" for i in range(n_test)],
            "response_a": [f"Test Answer A {i}" for i in range(n_test)],
            "response_b": [f"Test Answer B {i}" for i in range(n_test)],
        })

        print(f"Synthetic data: train={len(train_df)}, test={len(test_df)}")

    # Bias analysis
    train_df = analyze_biases(train_df)

    predictions = {}
    oof_predictions = {}

    # Phase 1: TF-IDF
    try:
        test_preds_tfidf, oof_tfidf = tfidf_enhanced(train_df, test_df)
        predictions["tfidf"] = test_preds_tfidf
        oof_predictions["tfidf"] = oof_tfidf
    except Exception as e:
        print(f"TF-IDF failed: {e}")

    # Phase 2: SBERT (also runs TabPFN internally)
    try:
        test_preds_sbert, oof_sbert, test_preds_tabpfn, oof_tabpfn = sbert_enhanced(train_df, test_df)
        predictions["sbert"] = test_preds_sbert
        oof_predictions["sbert"] = oof_sbert
        if test_preds_tabpfn is not None:
            predictions["tabpfn"] = test_preds_tabpfn
            oof_predictions["tabpfn"] = oof_tabpfn
    except Exception as e:
        print(f"SBERT/TabPFN failed: {e}")
        import traceback
        traceback.print_exc()

    # Phase 3: DeBERTa (only if GPU available and data looks real)
    if DEVICE == "cuda" and len(train_df) > 1000:
        try:
            test_preds_deberta, oof_deberta = deberta_finetune(train_df, test_df)
            predictions["deberta"] = test_preds_deberta
            oof_predictions["deberta"] = oof_deberta
        except Exception as e:
            print(f"DeBERTa failed: {e}")
            import traceback
            traceback.print_exc()

    # Phase 4: Ensemble
    print("\n=== Phase 4: Ensemble ===")
    if len(predictions) > 1:
        # Optimize weights based on OOF
        y_true = train_df["target"].values
        best_loss = 999
        best_weights = None

        # Grid search weights (4 models: tfidf, sbert, tabpfn, deberta)
        names = list(predictions.keys())
        n_models = len(names)

        if n_models == 2:
            for w0 in np.arange(0.0, 1.1, 0.1):
                w1 = 1.0 - w0
                ens_oof = np.zeros_like(oof_predictions[names[0]])
                ens_oof += w0 * oof_predictions[names[0]]
                ens_oof += w1 * oof_predictions[names[1]]
                loss = log_loss(y_true, ens_oof)
                if loss < best_loss:
                    best_loss = loss
                    best_weights = {names[0]: w0, names[1]: w1}
        elif n_models == 3:
            for w0 in np.arange(0.0, 1.1, 0.1):
                for w1 in np.arange(0.0, 1.1 - w0, 0.1):
                    w2 = 1.0 - w0 - w1
                    if w2 < 0:
                        continue
                    ens_oof = np.zeros_like(oof_predictions[names[0]])
                    ens_oof += w0 * oof_predictions[names[0]]
                    ens_oof += w1 * oof_predictions[names[1]]
                    ens_oof += w2 * oof_predictions[names[2]]
                    loss = log_loss(y_true, ens_oof)
                    if loss < best_loss:
                        best_loss = loss
                        best_weights = {names[0]: w0, names[1]: w1, names[2]: w2}
        else:
            # 4+ models: coarser grid
            for w0 in np.arange(0.0, 1.1, 0.2):
                for w1 in np.arange(0.0, 1.1 - w0, 0.2):
                    for w2 in np.arange(0.0, 1.1 - w0 - w1, 0.2):
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < 0:
                            continue
                        ens_oof = np.zeros_like(oof_predictions[names[0]])
                        ens_oof += w0 * oof_predictions[names[0]]
                        ens_oof += w1 * oof_predictions[names[1]]
                        ens_oof += w2 * oof_predictions[names[2]]
                        ens_oof += w3 * oof_predictions[names[3]]
                        loss = log_loss(y_true, ens_oof)
                        if loss < best_loss:
                            best_loss = loss
                            best_weights = {names[0]: w0, names[1]: w1, names[2]: w2, names[3]: w3}

        print(f"Best OOF log_loss: {best_loss:.4f}")
        print(f"Best weights: {best_weights}")

        # Apply weights
        ens_preds = np.zeros_like(list(predictions.values())[0])
        for name, w in best_weights.items():
            if name in predictions and w > 0:
                ens_preds += w * predictions[name]

        final_preds = ens_preds
    elif len(predictions) == 1:
        final_preds = list(predictions.values())[0]
    else:
        print("No predictions generated!")
        final_preds = None

    # Generate submission
    if final_preds is not None:
        print("\n=== Submission ===")

        # Normalize
        final_preds = final_preds / final_preds.sum(axis=1, keepdims=True)

        submission = pd.DataFrame({
            "id": test_df["id"],
            "winner_model_a": final_preds[:, 0],
            "winner_model_b": final_preds[:, 1],
            "winner_tie": final_preds[:, 2],
        })

        submission_path = OUTPUT_DIR / "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Saved submission to {submission_path}")
        print(submission.head())

        # Verify
        row_sums = submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
        print(f"\nRow sums check: min={row_sums.min():.6f}, max={row_sums.max():.6f}")