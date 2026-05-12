"""
Componente 3 — Motore di Scoring Ibrido (Opzione D)

Combina due approcci:
  1. Embedding similarity  — cosine similarity tra testo della query e descrizione testuale resort
  2. MLP score             — rete neurale addestrata su coppie (profilo_utente, resort) → punteggio 0-10

Score finale:
    score = ALPHA_EMBED * embed_score + ALPHA_MLP * mlp_score + ALPHA_DIST * dist_score

Il motore è stateless a regime: carica artefatti pre-calcolati da data/scorer_artifacts/.
Per costruire gli artefatti la prima volta: python -m backend.data.build_scorer

Uso:
    from backend.core.scoring import HybridScorer
    scorer = HybridScorer()
    results = scorer.rank(intent, top_k=3)
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH      = ROOT / "data" / "european_ski_resorts_geo.csv"
ARTIFACTS_DIR     = ROOT / "data" / "scorer_artifacts"
EMBEDDINGS_PATH   = ARTIFACTS_DIR / "resort_embeddings.npy"
MLP_PATH          = ARTIFACTS_DIR / "mlp_model.pkl"
SCALER_PATH       = ARTIFACTS_DIR / "feature_scaler.pkl"
RESORT_INDEX_PATH = ARTIFACTS_DIR / "resort_index.json"

# Pesi ibridi senza city_reference (somma = 1.0)
ALPHA_EMBED = 0.375
ALPHA_MLP   = 0.625
ALPHA_DIST  = 0.0    # non usato senza city_reference

# Pesi con city_reference — distanza domina (somma = 1.0)
ALPHA_EMBED_GEO = 0.15
ALPHA_MLP_GEO   = 0.35
ALPHA_DIST_GEO  = 0.50

# Mapping country (NLU output → dataset)
COUNTRY_ALIASES: Dict[str, str] = {
    "italy":       "Italy",
    "france":      "France",
    "austria":     "Austria",
    "switzerland": "Switzerland",
    "spain":       "Spain",
    "germany":     "Germany",
    "andorra":     "Andorra",
    "norway":      "Norway",
    "sweden":      "Sweden",
    "finland":     "Finland",
    "bulgaria":    "Bulgaria",
    "slovenia":    "Slovenia",
    "slovakia":    "Slovakia",
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def profile_to_features(intent: Dict[str, Any]) -> np.ndarray:
    """
    Converte un intent NLU in un vettore di feature per il MLP.
    Shape: (12,)
    """
    # Level one-hot (3)
    level_map = {"beginner": [1, 0, 0], "intermediate": [0, 1, 0], "expert": [0, 0, 1]}
    level_vec = level_map.get(intent.get("level"), [0, 0, 0])

    # Budget one-hot (3)
    budget_map = {"low": [1, 0, 0], "medium": [0, 1, 0], "high": [0, 0, 1]}
    budget_vec = budget_map.get(intent.get("budget"), [0, 0, 0])

    # Experience one-hot (4)
    exp_map = {
        "family":    [1, 0, 0, 0],
        "apres_ski": [0, 1, 0, 0],
        "sport":     [0, 0, 1, 0],
        "panoramic": [0, 0, 0, 1],
    }
    exp_vec = exp_map.get(intent.get("experience_type"), [0, 0, 0, 0])

    # Snowpark preference: -1 no / 0 null / +1 yes  (1)
    sp = intent.get("snowpark")
    sp_val = 1.0 if sp is True else (-1.0 if sp is False else 0.0)

    # Night ski preference (1)
    ns = intent.get("night_ski")
    ns_val = 1.0 if ns is True else (-1.0 if ns is False else 0.0)

    return np.array(level_vec + budget_vec + exp_vec + [sp_val, ns_val], dtype=np.float32)


def resort_to_features(row: pd.Series, stats: Dict[str, float]) -> np.ndarray:
    """
    Converte una riga del dataset in un vettore di feature per il MLP.
    Shape: (10,)
    stats: dizionario con max di ogni colonna numerica per normalizzazione.
    """
    total = max(float(row["TotalSlope"]), 1.0)

    price_norm      = float(row["DayPassPriceAdult"]) / stats["max_price"]
    highest_norm    = float(row["HighestPoint"]) / stats["max_highest"]
    beginner_ratio  = float(row["BeginnerSlope"]) / total
    inter_ratio     = float(row["IntermediateSlope"]) / total
    diff_ratio      = float(row["DifficultSlope"]) / total
    slope_norm      = min(float(row["TotalSlope"]) / stats["max_slope"], 1.0)
    snowpark        = 1.0 if str(row["Snowparks"]).lower() == "yes" else 0.0
    nightski        = 1.0 if str(row["NightSki"]).lower() == "yes" else 0.0
    lifts_norm      = float(row["LiftCapacity"]) / stats["max_lift_cap"]
    cannons_norm    = float(row["SnowCannons"]) / stats["max_cannons"]

    return np.array(
        [price_norm, highest_norm, beginner_ratio, inter_ratio, diff_ratio,
         slope_norm, snowpark, nightski, lifts_norm, cannons_norm],
        dtype=np.float32,
    )


def compute_dataset_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "max_price":    float(df["DayPassPriceAdult"].max()) or 1.0,
        "max_highest":  float(df["HighestPoint"].max()) or 1.0,
        "max_slope":    float(df["TotalSlope"].max()) or 1.0,
        "max_lift_cap": float(df["LiftCapacity"].max()) or 1.0,
        "max_cannons":  float(df["SnowCannons"].max()) or 1.0,
    }


# ---------------------------------------------------------------------------
# Testo per embedding
# ---------------------------------------------------------------------------

LEVEL_IT   = {"beginner": "principiante", "intermediate": "intermedio", "expert": "esperto"}
BUDGET_IT  = {"low": "economico (meno di 35€)", "medium": "nella media (35-55€)", "high": "alto (più di 55€)"}
EXP_IT     = {
    "family":    "vacanza in famiglia con bambini, piste facili",
    "apres_ski": "après-ski e vita notturna, atmosfera festosa",
    "sport":     "sciatore sportivo, piste difficili e molti km",
    "panoramic": "paesaggi e panorami alpini mozzafiato",
}


def resort_to_text(row: pd.Series) -> str:
    """Genera una descrizione testuale in italiano del resort per l'embedding."""
    snowpark_txt = "Snowpark disponibile." if str(row["Snowparks"]).lower() == "yes" else ""
    nightski_txt = "Sci notturno disponibile." if str(row["NightSki"]).lower() == "yes" else ""
    return (
        f"Stazione sciistica in {row['Country']}. "
        f"Quota massima {row['HighestPoint']}m, quota minima {row['LowestPoint']}m. "
        f"Piste: {row['BeginnerSlope']}km facili, {row['IntermediateSlope']}km intermedie, "
        f"{row['DifficultSlope']}km difficili (totale {row['TotalSlope']}km). "
        f"Ski pass adulto {row['DayPassPriceAdult']}€. "
        f"{snowpark_txt} {nightski_txt}"
        f"Impianti: {row['TotalLifts']} con capacità {row['LiftCapacity']} persone/ora."
    )


def intent_to_query_text(intent: Dict[str, Any]) -> str:
    """Costruisce un testo di query dall'intent NLU per l'embedding."""
    parts = []
    if intent.get("level"):
        parts.append(f"Sciatore {LEVEL_IT.get(intent['level'], intent['level'])}.")
    if intent.get("experience_type"):
        parts.append(EXP_IT.get(intent["experience_type"], ""))
    if intent.get("budget"):
        parts.append(f"Budget {BUDGET_IT.get(intent['budget'], intent['budget'])}.")
    if intent.get("country"):
        parts.append(f"Preferenza paese: {intent['country']}.")
    if intent.get("snowpark"):
        parts.append("Snowpark richiesto.")
    if intent.get("night_ski"):
        parts.append("Sci notturno richiesto.")
    return " ".join(parts) if parts else "Stazione sciistica europea."


# ---------------------------------------------------------------------------
# Scoring rule-based (usato per generare training data)
# ---------------------------------------------------------------------------

def rule_based_score(intent: Dict[str, Any], row: pd.Series) -> float:
    """
    Punteggio deterministico 0-10 basato su regole.
    Usato come label di training per il MLP.
    """
    score = 5.0
    total = max(float(row["TotalSlope"]), 1.0)

    # --- Livello ---
    level = intent.get("level")
    if level == "beginner":
        score += 3.0 * (float(row["BeginnerSlope"]) / total)
        if float(row["DifficultSlope"]) / total > 0.4:
            score -= 1.5  # troppo difficile per principianti
    elif level == "expert":
        score += 3.0 * (float(row["DifficultSlope"]) / total)
        if float(row["TotalSlope"]) < 30:
            score -= 1.0  # troppo piccola per esperti
    else:  # intermediate
        score += 2.0 * (float(row["IntermediateSlope"]) / total)

    # --- Budget ---
    price = float(row["DayPassPriceAdult"])
    budget = intent.get("budget")
    if budget == "low":
        if price <= 35:
            score += 2.0
        elif price > 55:
            score -= 3.0
        elif price > 45:
            score -= 1.0
    elif budget == "medium":
        if 35 <= price <= 55:
            score += 2.0
        elif price > 65:
            score -= 1.5
        elif price < 25:
            score -= 0.5
    elif budget == "high":
        if price >= 50:
            score += 1.0
        elif price < 30:
            score -= 0.5

    # --- Snowpark ---
    has_snowpark = str(row["Snowparks"]).lower() == "yes"
    sp_pref = intent.get("snowpark")
    if sp_pref is True:
        score += 1.5 if has_snowpark else -2.0
    elif sp_pref is False and has_snowpark:
        score -= 0.5

    # --- Sci notturno ---
    has_night = str(row["NightSki"]).lower() == "yes"
    ns_pref = intent.get("night_ski")
    if ns_pref is True:
        score += 1.5 if has_night else -2.0

    # --- Tipo esperienza ---
    exp = intent.get("experience_type")
    if exp == "sport":
        # Premia km totali e piste difficili
        score += min(float(row["TotalSlope"]), 300) / 100  # max +3
        score += float(row["DifficultSlope"]) / total      # max +1
    elif exp == "family":
        score += 2.0 * (float(row["BeginnerSlope"]) / total)
        score += min(float(row["TotalLifts"]), 50) / 50    # più impianti = più servizi
    elif exp == "panoramic":
        score += min(float(row["HighestPoint"]), 3500) / 1750  # max +2
    elif exp == "apres_ski":
        # Resort grandi → più vita notturna
        score += min(float(row["LiftCapacity"]), 150000) / 75000  # max +2

    return float(np.clip(score, 0.0, 10.0))


# ---------------------------------------------------------------------------
# Distanza Haversine
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def geocode_city(city: str) -> Optional[tuple]:
    """Geocodifica una città con Nominatim. Restituisce (lat, lon) o None."""
    try:
        from geopy.geocoders import Nominatim
        import time
        gc = Nominatim(user_agent="snowsense_scorer")
        loc = gc.geocode(city, timeout=5)
        time.sleep(1.1)
        return (loc.latitude, loc.longitude) if loc else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HybridScorer
# ---------------------------------------------------------------------------

class HybridScorer:
    """
    Motore di scoring ibrido content-based.
    Carica artefatti pre-addestrati da data/scorer_artifacts/.
    """

    def __init__(self) -> None:
        self.df = pd.read_csv(DATASET_PATH, index_col=0)
        self.stats = compute_dataset_stats(self.df)

        if not EMBEDDINGS_PATH.exists() or not MLP_PATH.exists():
            raise FileNotFoundError(
                "Artefatti non trovati. Esegui prima: python -m backend.data.build_scorer"
            )

        self.embeddings: np.ndarray = np.load(EMBEDDINGS_PATH)
        with open(MLP_PATH, "rb") as f:
            self.mlp = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(RESORT_INDEX_PATH) as f:
            self.resort_index: List[int] = json.load(f)  # lista di index del dataframe

        # Lazy: sentence-transformers caricato solo al primo uso
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._embedder

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def rank(
        self,
        intent: Dict[str, Any],
        top_k: int = 3,
        country_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dato un intent NLU, restituisce i top_k resort ordinati per score ibrido.

        Ogni elemento della lista contiene:
            resort_id, Resort, Country, scores (embed, mlp, dist, final), dati resort
        """
        df = self.df.copy()
        idx_map = {orig_idx: pos for pos, orig_idx in enumerate(self.resort_index)}

        # Filtro paese (opzionale)
        filter_country = country_filter or _resolve_country(intent.get("country"))
        if filter_country:
            mask = df["Country"].str.lower() == filter_country.lower()
            if mask.sum() > 0:
                df = df[mask]

        # Geocodifica città di riferimento (una volta sola)
        city_coords = None
        city_ref = intent.get("city_reference")
        if city_ref:
            city_coords = geocode_city(city_ref)

        # Prepara feature profilo per MLP
        profile_feats = profile_to_features(intent)

        # Pre-calcola embedding query una volta sola (non dentro il loop)
        query_emb = None
        query_text = intent_to_query_text(intent)
        if query_text:
            query_emb = self._get_embedder().encode(query_text, normalize_embeddings=True)

        results = []
        for orig_idx, row in df.iterrows():
            # --- Embedding score ---
            pos = idx_map.get(orig_idx)
            if pos is not None and query_emb is not None:
                resort_emb = self.embeddings[pos]
                embed_score = float(np.dot(query_emb, resort_emb))  # cosine (embeddings normalizzati)
            else:
                embed_score = 0.5  # resort senza embedding → score neutro

            # --- MLP score ---
            resort_feats = resort_to_features(row, self.stats)
            combined = np.concatenate([profile_feats, resort_feats]).reshape(1, -1)
            combined_scaled = self.scaler.transform(combined)
            mlp_raw = float(self.mlp.predict(combined_scaled)[0])
            mlp_score = float(np.clip(mlp_raw, 0.0, 10.0)) / 10.0  # normalizza 0-1

            # --- Distanza ---
            dist_score = 0.0
            dist_km_value: Optional[float] = None
            if city_coords and pd.notna(row.get("Latitude")) and pd.notna(row.get("Longitude")):
                dist_km_value = haversine_km(city_coords[0], city_coords[1],
                                             float(row["Latitude"]), float(row["Longitude"]))
                # Più vicino = punteggio più alto; 0km → 1.0, 1500km → 0.0
                dist_score = float(max(0.0, 1.0 - dist_km_value / 1500.0))

            # --- Score ibrido finale ---
            if city_coords:
                # city_reference presente: distanza domina (ALPHA_DIST_GEO = 0.50)
                final = (ALPHA_EMBED_GEO * embed_score
                         + ALPHA_MLP_GEO   * mlp_score
                         + ALPHA_DIST_GEO  * dist_score)
            else:
                # Nessuna città: solo embedding + MLP
                final = ALPHA_EMBED * embed_score + ALPHA_MLP * mlp_score

            results.append({
                "resort_id":        int(orig_idx),
                "Resort":           row["Resort"],
                "Country":          row["Country"],
                "HighestPoint":     int(row["HighestPoint"]),
                "LowestPoint":      int(row["LowestPoint"]),
                "DayPassPriceAdult": float(row["DayPassPriceAdult"]),
                "TotalSlope":       float(row["TotalSlope"]),
                "BeginnerSlope":    float(row["BeginnerSlope"]),
                "IntermediateSlope": float(row["IntermediateSlope"]),
                "DifficultSlope":   float(row["DifficultSlope"]),
                "Snowparks":        row["Snowparks"],
                "NightSki":         row["NightSki"],
                "TotalLifts":       int(row["TotalLifts"]),
                "LiftCapacity":     int(row["LiftCapacity"]),
                "SnowCannons":      int(row["SnowCannons"]),
                "Latitude":         float(row["Latitude"]) if pd.notna(row.get("Latitude")) else None,
                "Longitude":        float(row["Longitude"]) if pd.notna(row.get("Longitude")) else None,
                "distance_km":      round(dist_km_value, 1) if dist_km_value is not None else None,
                "scores": {
                    "embedding": round(embed_score, 4),
                    "mlp":       round(mlp_score, 4),
                    "distance":  round(dist_score, 4),
                    "final":     round(final, 4),
                },
            })

        results.sort(key=lambda x: x["scores"]["final"], reverse=True)
        return results[:top_k]


def _resolve_country(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return COUNTRY_ALIASES.get(raw.lower(), raw)
