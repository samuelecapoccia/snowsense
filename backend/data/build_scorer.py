"""
Componente 3 — Build Script (eseguire una volta sola)
Genera gli artefatti necessari al HybridScorer:
  1. resort_embeddings.npy   — embedding testuale di ogni resort
  2. mlp_model.pkl           — MLPRegressor addestrato su dati sintetici
  3. feature_scaler.pkl      — StandardScaler per le feature MLP
  4. resort_index.json       — mapping posizione array → index dataframe

Esecuzione:
    python -m backend.data.build_scorer

Tempo stimato: 3-5 minuti (download modello embedding al primo avvio + training)
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH   = ROOT / "data" / "european_ski_resorts_geo.csv"
PROFILES_PATH  = ROOT / "data" / "synthetic_profiles.json"
ARTIFACTS_DIR  = ROOT / "data" / "scorer_artifacts"

from backend.core.scoring import (
    resort_to_features,
    resort_to_text,
    profile_to_features,
    rule_based_score,
    compute_dataset_stats,
)


# ---------------------------------------------------------------------------
# 1. Training data generation
# ---------------------------------------------------------------------------

def generate_training_data(
    df: pd.DataFrame,
    profiles: list,
    stats: dict,
) -> tuple:
    """
    Genera coppie (profile_features + resort_features, score) per tutti i
    profili × tutti i resort.
    Restituisce X (n_samples, 22), y (n_samples,).
    """
    print(f"  Generazione training data: {len(profiles)} profili × {len(df)} resort "
          f"= {len(profiles) * len(df):,} coppie...")

    X_rows = []
    y_rows = []

    for profile in profiles:
        intent = profile["ground_truth_intent"]
        # Aggiungi snowpark/night_ski dal profilo strutturato (più completo dell'intent)
        intent_full = dict(intent)
        intent_full.setdefault("snowpark",   profile["profile"].get("snowpark"))
        intent_full.setdefault("night_ski",  profile["profile"].get("night_ski"))
        intent_full.setdefault("level",      profile["profile"].get("level"))
        intent_full.setdefault("budget",     profile["profile"].get("budget"))
        intent_full.setdefault("experience_type", profile["profile"].get("experience_type"))

        p_feats = profile_to_features(intent_full)

        for orig_idx, row in df.iterrows():
            r_feats = resort_to_features(row, stats)
            score   = rule_based_score(intent_full, row)

            X_rows.append(np.concatenate([p_feats, r_feats]))
            y_rows.append(score)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    print(f"  Dataset: X={X.shape}, y={y.shape}  "
          f"(score medio={y.mean():.2f}, std={y.std():.2f})")
    return X, y


# ---------------------------------------------------------------------------
# 2. MLP Training
# ---------------------------------------------------------------------------

def train_mlp(X: np.ndarray, y: np.ndarray) -> tuple:
    """Addestra un MLPRegressor e restituisce (model, scaler, metrics)."""
    print("  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )

    print(f"  Train: {len(X_train):,} campioni  |  Test: {len(X_test):,} campioni")
    print("  Training MLP (2 hidden layers: 64, 32)...")

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(np.mean(np.abs(y_pred - y_test)))

    print(f"  Training completato in {mlp.n_iter_} iterazioni")
    print(f"  RMSE test: {rmse:.3f}  |  MAE test: {mae:.3f}  (scala 0-10)")

    return mlp, scaler, {"rmse": round(rmse, 4), "mae": round(mae, 4)}


# ---------------------------------------------------------------------------
# 3. Resort Embeddings
# ---------------------------------------------------------------------------

def compute_resort_embeddings(df: pd.DataFrame) -> tuple:
    """
    Calcola e normalizza gli embedding per tutti i resort.
    Restituisce (embeddings_matrix, resort_index_list).
    """
    print(f"  Caricamento modello sentence-transformers "
          f"(paraphrase-multilingual-MiniLM-L12-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    texts = []
    index = []
    for orig_idx, row in df.iterrows():
        texts.append(resort_to_text(row))
        index.append(int(orig_idx))

    print(f"  Encoding {len(texts)} resort...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,   # L2 norm → cosine = dot product
        show_progress_bar=True,
    )
    print(f"  Completato in {time.time()-t0:.1f}s  |  Shape: {embeddings.shape}")
    return embeddings, index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("SnowSense — Build Scorer Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carica dataset e profili
    print("\n[1/3] Caricamento dati...")
    df = pd.read_csv(DATASET_PATH, index_col=0)
    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)
    stats = compute_dataset_stats(df)
    print(f"  Dataset: {len(df)} resort  |  Profili: {len(profiles)}")

    # Genera training data e addestra MLP
    print("\n[2/3] Training MLP...")
    X, y = generate_training_data(df, profiles, stats)
    mlp, scaler, metrics = train_mlp(X, y)

    # Salva MLP e scaler
    with open(ARTIFACTS_DIR / "mlp_model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    with open(ARTIFACTS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Salvati: mlp_model.pkl, feature_scaler.pkl")

    # Calcola e salva embedding
    print("\n[3/3] Calcolo resort embeddings...")
    embeddings, resort_index = compute_resort_embeddings(df)
    np.save(ARTIFACTS_DIR / "resort_embeddings.npy", embeddings)
    with open(ARTIFACTS_DIR / "resort_index.json", "w") as f:
        json.dump(resort_index, f)
    print(f"  Salvati: resort_embeddings.npy ({embeddings.shape}), resort_index.json")

    # Salva metriche per il report
    summary = {
        "n_resorts": len(df),
        "n_profiles": len(profiles),
        "n_training_samples": int(len(X)),
        "mlp_hidden_layers": [64, 32],
        "mlp_metrics": metrics,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(embeddings.shape[1]),
        "alpha_embed": 0.30,
        "alpha_mlp": 0.50,
        "alpha_dist": 0.20,
    }
    with open(ARTIFACTS_DIR / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Build completato!")
    print(f"  MLP RMSE: {metrics['rmse']}  |  MAE: {metrics['mae']}")
    print(f"  Artefatti in: {ARTIFACTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
