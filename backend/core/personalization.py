"""
Componente 5 — Personalizzazione Implicita (lato backend)

Responsabilità:
  1. Costruisce un contesto testuale dal profilo utente anonimo da iniettare
     nel prompt LLM (NLU + spiegazioni) per personalizzare le risposte.
  2. Applica una strategia epsilon-greedy di exploration: con probabilità ε
     sostituisce l'ultimo resort della lista top-K con uno che esce dal profilo
     abituale, segnalato con flag "is_exploration" e una spiegazione.

Il profilo utente vive nel localStorage del browser (frontend) e viene inviato
al backend come dizionario opzionale ad ogni richiesta.

Uso:
    from backend.core.personalization import PersonalizationEngine
    engine = PersonalizationEngine()

    # Arricchisce i risultati con possibile resort di exploration
    results = engine.apply_exploration(ranked_resorts, all_df, user_profile)

    # Genera il contesto da iniettare nel prompt LLM
    ctx = engine.build_llm_context(user_profile)
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Probabilità di exploration per sessione
EPSILON = 0.20

# Quante sessioni di storia considerare per dominanza profilo
HISTORY_WINDOW = 10

# ---------------------------------------------------------------------------
# Schema del profilo utente (specchio del tipo TypeScript lato frontend)
# ---------------------------------------------------------------------------
#
# {
#   "level_counts":      {"beginner": int, "intermediate": int, "expert": int},
#   "budget_counts":     {"low": int, "medium": int, "high": int},
#   "country_counts":    {"Italy": int, "France": int, ...},
#   "experience_counts": {"family": int, "apres_ski": int, "sport": int, "panoramic": int},
#   "snowpark_pref":     true | false | null,
#   "night_ski_pref":    true | false | null,
#   "query_history":     [{"query": str, "timestamp": str}, ...],   # ultimi N
#   "session_count":     int,
#   "last_updated":      str   # ISO timestamp
# }

# ---------------------------------------------------------------------------
# Testi italiani per exploration reason
# ---------------------------------------------------------------------------

_EXPLORATION_REASONS = {
    "country": "Fuori dalla tua zona abituale — potrebbe sorprenderti.",
    "level":   "Stazione con caratteristiche diverse dal tuo profilo — vale la pena scoprirla.",
    "budget":  "Proposta fuori dal tuo range solito — il rapporto qualità-prezzo è ottimo.",
    "default": "Una proposta insolita rispetto alle tue ricerche precedenti — aperta all'esplorazione.",
}


# ---------------------------------------------------------------------------
# PersonalizationEngine
# ---------------------------------------------------------------------------

class PersonalizationEngine:
    """
    Gestisce la personalizzazione implicita lato backend.
    Stateless: riceve il profilo dal frontend ad ogni richiesta.
    """

    def __init__(self, epsilon: float = EPSILON) -> None:
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def apply_exploration(
        self,
        ranked_resorts: List[Dict[str, Any]],
        all_df: pd.DataFrame,
        user_profile: Optional[Dict[str, Any]],
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Applica epsilon-greedy exploration all'ultimo resort della lista.

        Con probabilità ε sostituisce ranked_resorts[-1] con un resort
        "sorprendente" (diverso dal profilo dominante dell'utente) estratto
        da all_df, escludendo i resort già presenti in ranked_resorts.

        Args:
            ranked_resorts: output di HybridScorer.rank() (già arricchiti con explanation)
            all_df:         DataFrame completo dei resort (per il candidato exploration)
            user_profile:   profilo anonimo dal localStorage (None = sessione anonima)
            intent:         intent corrente (per escludere country filter dall'exploration)

        Returns:
            lista (stessa lunghezza) con eventuale resort di exploration all'ultima posizione
        """
        if not ranked_resorts:
            return ranked_resorts

        # Nessun profilo abbastanza ricco → nessuna exploration
        if user_profile is None or user_profile.get("session_count", 0) < 3:
            return ranked_resorts

        # Lancio del dado epsilon-greedy
        if random.random() > self.epsilon:
            return ranked_resorts

        # Trova resort di exploration
        already_shown = {r["Resort"] for r in ranked_resorts}
        candidate, reason_key = self._find_exploration_candidate(
            all_df, user_profile, intent, already_shown
        )

        if candidate is None:
            return ranked_resorts

        # Sostituisce l'ultimo resort
        result = list(ranked_resorts[:-1])
        candidate["is_exploration"] = True
        candidate["exploration_reason"] = _EXPLORATION_REASONS.get(reason_key, _EXPLORATION_REASONS["default"])
        result.append(candidate)
        return result

    def build_llm_context(self, user_profile: Optional[Dict[str, Any]]) -> str:
        """
        Costruisce una stringa di contesto dal profilo utente da iniettare
        nel system prompt dell'LLM (NLU o explainer).

        Restituisce stringa vuota se profilo assente o insufficiente.
        """
        if not user_profile or user_profile.get("session_count", 0) < 2:
            return ""

        parts = []

        dominant_level = _dominant(user_profile.get("level_counts", {}))
        if dominant_level:
            _LEVEL_IT = {"beginner": "principiante", "intermediate": "intermedio", "expert": "esperto"}
            parts.append(f"livello prevalente: {_LEVEL_IT.get(dominant_level, dominant_level)}")

        dominant_budget = _dominant(user_profile.get("budget_counts", {}))
        if dominant_budget:
            _BUDGET_IT = {"low": "basso", "medium": "medio", "high": "alto"}
            parts.append(f"budget tipico: {_BUDGET_IT.get(dominant_budget, dominant_budget)}")

        dominant_exp = _dominant(user_profile.get("experience_counts", {}))
        if dominant_exp:
            _EXP_IT = {
                "family":    "famiglie con bambini",
                "apres_ski": "après-ski",
                "sport":     "sci sportivo",
                "panoramic": "panorami alpini",
            }
            parts.append(f"esperienza preferita: {_EXP_IT.get(dominant_exp, dominant_exp)}")

        top_countries = _top_n(user_profile.get("country_counts", {}), n=2)
        if top_countries:
            parts.append(f"zone frequentate: {', '.join(top_countries)}")

        sp = user_profile.get("snowpark_pref")
        if sp is True:
            parts.append("interesse per snowpark")

        ns = user_profile.get("night_ski_pref")
        if ns is True:
            parts.append("interesse per sci notturno")

        if not parts:
            return ""

        sessions = user_profile.get("session_count", 0)
        return (
            f"[Contesto utente — {sessions} ricerche precedenti: {'; '.join(parts)}. "
            f"Tieni conto di questo profilo nelle raccomandazioni, senza menzionarlo esplicitamente.]"
        )

    # ------------------------------------------------------------------
    # Logica interna exploration
    # ------------------------------------------------------------------

    def _find_exploration_candidate(
        self,
        all_df: pd.DataFrame,
        user_profile: Dict[str, Any],
        intent: Optional[Dict[str, Any]],
        already_shown: set,
    ):
        """
        Trova un resort "sorprendente" estraendo il profilo dominante e cercando
        qualcosa di diverso. Restituisce (resort_dict, reason_key) o (None, None).
        """
        dominant_country = _dominant(user_profile.get("country_counts", {}))
        dominant_budget  = _dominant(user_profile.get("budget_counts", {}))

        # Strategia 1: resort in paese diverso da quello dominante
        if dominant_country:
            mask = (
                (all_df["Country"].str.lower() != dominant_country.lower()) &
                (~all_df["Resort"].isin(already_shown))
            )
            pool = all_df[mask]
            if len(pool) > 0:
                row = pool.sample(1, random_state=None).iloc[0]
                return _row_to_dict(row, all_df.index.name), "country"

        # Strategia 2: resort con fascia prezzo diversa dal budget dominante
        if dominant_budget:
            price_ranges = {
                "low":    (0, 35),
                "medium": (35, 55),
                "high":   (55, 9999),
            }
            lo, hi = price_ranges.get(dominant_budget, (0, 9999))
            # Cerca in fascia adiacente
            mask = (
                ((all_df["DayPassPriceAdult"] < lo) | (all_df["DayPassPriceAdult"] > hi)) &
                (~all_df["Resort"].isin(already_shown))
            )
            pool = all_df[mask]
            if len(pool) > 0:
                row = pool.sample(1, random_state=None).iloc[0]
                return _row_to_dict(row, all_df.index.name), "budget"

        # Fallback: resort casuale non già mostrato
        mask = ~all_df["Resort"].isin(already_shown)
        pool = all_df[mask]
        if len(pool) > 0:
            row = pool.sample(1, random_state=None).iloc[0]
            return _row_to_dict(row, all_df.index.name), "default"

        return None, None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _dominant(counts: Dict[str, int]) -> Optional[str]:
    """Restituisce la chiave con il conteggio più alto, o None se vuoto."""
    if not counts:
        return None
    return max(counts, key=counts.get)


def _top_n(counts: Dict[str, int], n: int = 2) -> List[str]:
    """Restituisce le top-n chiavi per conteggio."""
    return sorted(counts, key=counts.get, reverse=True)[:n]


def _row_to_dict(row: pd.Series, index_name) -> Dict[str, Any]:
    """Converte una riga DataFrame nel formato dict usato dal sistema."""
    def safe_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    lat = row.get("Latitude")
    lon = row.get("Longitude")

    return {
        "resort_id":         safe_int(row.name),
        "Resort":            str(row["Resort"]),
        "Country":           str(row["Country"]),
        "HighestPoint":      safe_int(row["HighestPoint"]),
        "LowestPoint":       safe_int(row["LowestPoint"]),
        "DayPassPriceAdult": safe_float(row["DayPassPriceAdult"]),
        "TotalSlope":        safe_float(row["TotalSlope"]),
        "BeginnerSlope":     safe_float(row["BeginnerSlope"]),
        "IntermediateSlope": safe_float(row["IntermediateSlope"]),
        "DifficultSlope":    safe_float(row["DifficultSlope"]),
        "Snowparks":         str(row["Snowparks"]),
        "NightSki":          str(row["NightSki"]),
        "TotalLifts":        safe_int(row["TotalLifts"]),
        "LiftCapacity":      safe_int(row["LiftCapacity"]),
        "SnowCannons":       safe_int(row["SnowCannons"]),
        "Latitude":          float(lat) if pd.notna(lat) else None,
        "Longitude":         float(lon) if pd.notna(lon) else None,
        "scores":            {"embedding": 0.0, "mlp": 0.0, "distance": 0.0, "final": 0.0},
        "explanation":       None,
        "is_exploration":    False,
        "exploration_reason": None,
    }


# ---------------------------------------------------------------------------
# Funzione di utilità: aggiornamento profilo lato backend (per testing)
# ---------------------------------------------------------------------------

def update_profile(
    profile: Dict[str, Any],
    intent: Dict[str, Any],
    resorts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggiorna un profilo con i dati di una nuova ricerca.
    In produzione l'aggiornamento avviene nel frontend (localStorage).
    Questa funzione è fornita per testing e per il modulo di valutazione.
    """
    p = dict(profile)

    def inc(d, key):
        if key:
            d[key] = d.get(key, 0) + 1
        return d

    p["level_counts"]      = inc(dict(p.get("level_counts", {})),      intent.get("level"))
    p["budget_counts"]     = inc(dict(p.get("budget_counts", {})),      intent.get("budget"))
    p["experience_counts"] = inc(dict(p.get("experience_counts", {})),  intent.get("experience_type"))

    if resorts:
        countries = p.get("country_counts", {})
        for r in resorts:
            if not r.get("is_exploration"):
                countries[r["Country"]] = countries.get(r["Country"], 0) + 1
        p["country_counts"] = countries

    # Snowpark / night_ski: aggiornamento con majority voting semplice
    sp = intent.get("snowpark")
    if sp is not None:
        p["snowpark_pref"] = sp
    ns = intent.get("night_ski")
    if ns is not None:
        p["night_ski_pref"] = ns

    # Cronologia query (max 20 elementi)
    history = p.get("query_history", [])
    history = history[-19:]  # mantieni ultimi 19 + quello nuovo
    p["query_history"] = history

    p["session_count"] = p.get("session_count", 0) + 1

    return p


def empty_profile() -> Dict[str, Any]:
    """Restituisce un profilo vuoto (stato iniziale)."""
    return {
        "level_counts":      {},
        "budget_counts":     {},
        "country_counts":    {},
        "experience_counts": {},
        "snowpark_pref":     None,
        "night_ski_pref":    None,
        "query_history":     [],
        "session_count":     0,
        "last_updated":      None,
    }
