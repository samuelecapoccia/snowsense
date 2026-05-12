"""
Componente 6 — Valutazione qualità raccomandazioni (Scorer Evaluation)

Per ogni profilo sintetico in synthetic_profiles.json:
  1. Chiama POST /api/recommend con la query del profilo
  2. Verifica che almeno uno dei top-3 resort rispetti le aspettative:
     - Paese/zona corretti
     - Budget compatibile (prezzo ski pass)
     - Livello sciatore compatibile (distribuzione piste)
  3. Produce evaluation/scorer_results.csv
  4. Stampa accuratezza media finale

Esecuzione (con uvicorn già in esecuzione su :8000):
    .venv/bin/python evaluation/eval_scorer.py

Metriche:
  - country_match:  almeno 1 resort nel paese/zona attesa
  - budget_match:   almeno 1 resort nel range di prezzo atteso
  - level_match:    almeno 1 resort con distribuzione piste compatibile col livello
  - match_score:    media delle tre (0.0 – 1.0)
  - overall_pass:   match_score >= 0.67 (almeno 2/3 criteri soddisfatti)
"""

import json
import time
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL       = "http://localhost:8000/api/recommend"
PROFILES_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_profiles.json"
OUT_CSV       = Path(__file__).resolve().parent / "scorer_results.csv"
GEO_CSV       = Path(__file__).resolve().parents[1] / "data" / "european_ski_resorts_geo.csv"
STRATEGY      = "few_shot"
REQUEST_DELAY = 0.5   # secondi tra chiamate API (evita rate limit DeepSeek)

# ---------------------------------------------------------------------------
# Mapping zona → paesi attesi
# ---------------------------------------------------------------------------

ZONA_TO_COUNTRIES = {
    "italy_north":       {"Italy"},
    "italy_center_south":{"Italy"},
    "austria":           {"Austria"},
    "france":            {"France"},
    "no_preference":     None,   # None = qualsiasi paese accettato
}

# ---------------------------------------------------------------------------
# Budget range (€)
# ---------------------------------------------------------------------------

BUDGET_RANGE = {
    "low":    (0,   35),
    "medium": (35,  55),
    "high":   (55, 999),
}

# Tolleranza: ±5€ per non penalizzare resort borderline
BUDGET_TOLERANCE = 5

# ---------------------------------------------------------------------------
# Compatibilità livello (basata su distribuzione piste)
# ---------------------------------------------------------------------------

def level_compatible(resort: dict, expected_level: str) -> bool:
    """
    True se la distribuzione piste del resort è compatibile col livello atteso.
    Usa i campi BeginnerSlope, IntermediateSlope, DifficultSlope, TotalSlope.
    """
    total = max(float(resort.get("TotalSlope", 1)), 1)
    beg   = float(resort.get("BeginnerSlope", 0)) / total
    inter = float(resort.get("IntermediateSlope", 0)) / total
    diff  = float(resort.get("DifficultSlope", 0)) / total

    if expected_level == "beginner":
        # Almeno 20% piste facili, piste difficili < 50%
        return beg >= 0.20 and diff < 0.50

    elif expected_level == "intermediate":
        # Almeno 20% piste intermedie, non dominato da difficili
        return inter >= 0.20

    elif expected_level == "expert":
        # Almeno 15% piste difficili, O resort grande (>80km totali)
        return diff >= 0.15 or total >= 80

    return True   # livello sconosciuto → non penalizzare


# ---------------------------------------------------------------------------
# Valutazione singolo profilo
# ---------------------------------------------------------------------------

def evaluate_profile(profile: dict) -> dict:
    p      = profile["profile"]
    intent = profile["ground_truth_intent"]
    query  = profile["query_it"]

    # Paesi attesi dalla zona
    zona            = p.get("zona", "no_preference")
    expected_countries = ZONA_TO_COUNTRIES.get(zona)   # None = qualsiasi

    # Se ground_truth_intent ha country esplicito, ha priorità
    gt_country = intent.get("country")
    if gt_country:
        expected_countries = {gt_country}

    expected_level  = p.get("level")
    expected_budget = p.get("budget")

    # Chiama API
    try:
        resp = requests.post(
            API_URL,
            json={"query": query, "strategy": STRATEGY, "user_profile": None},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {
            "profile_id":         profile["id"],
            "query":              query[:80],
            "zona":               zona,
            "expected_countries": str(expected_countries),
            "returned_countries": "ERROR",
            "expected_level":     expected_level,
            "expected_budget":    expected_budget,
            "country_match":      0,
            "budget_match":       0,
            "level_match":        0,
            "match_score":        0.0,
            "overall_pass":       False,
            "status":             f"ERROR: {exc}",
        }

    if data.get("status") == "clarification_needed":
        return {
            "profile_id":         profile["id"],
            "query":              query[:80],
            "zona":               zona,
            "expected_countries": str(expected_countries),
            "returned_countries": "CLARIFICATION_REQUESTED",
            "expected_level":     expected_level,
            "expected_budget":    expected_budget,
            "country_match":      0,
            "budget_match":       0,
            "level_match":        0,
            "match_score":        0.0,
            "overall_pass":       False,
            "status":             "clarification_needed",
        }

    resorts = data.get("resorts", [])
    returned_countries = [r["Country"] for r in resorts]

    # ── Country match ──────────────────────────────────────────
    if expected_countries is None:
        country_match = 1   # no_preference → sempre ok
    else:
        country_match = int(any(r["Country"] in expected_countries for r in resorts))

    # ── Budget match ───────────────────────────────────────────
    if expected_budget and expected_budget in BUDGET_RANGE:
        lo, hi = BUDGET_RANGE[expected_budget]
        budget_match = int(any(
            (lo - BUDGET_TOLERANCE) <= r["DayPassPriceAdult"] <= (hi + BUDGET_TOLERANCE)
            for r in resorts
        ))
    else:
        budget_match = 1   # budget non specificato → non penalizzare

    # ── Level match ────────────────────────────────────────────
    if expected_level:
        level_match = int(any(level_compatible(r, expected_level) for r in resorts))
    else:
        level_match = 1

    match_score  = (country_match + budget_match + level_match) / 3
    overall_pass = match_score >= 0.67

    return {
        "profile_id":         profile["id"],
        "query":              query[:80],
        "zona":               zona,
        "expected_countries": ", ".join(expected_countries) if expected_countries else "any",
        "returned_countries": ", ".join(returned_countries),
        "expected_level":     expected_level,
        "expected_budget":    expected_budget,
        "country_match":      country_match,
        "budget_match":       budget_match,
        "level_match":        level_match,
        "match_score":        round(match_score, 3),
        "overall_pass":       overall_pass,
        "status":             "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Verifica server attivo
    try:
        requests.get("http://localhost:8000/api/health", timeout=5)
    except Exception:
        print("ERRORE: server non raggiungibile su :8000. Avvia uvicorn prima.")
        sys.exit(1)

    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)

    print(f"Valutazione scorer su {len(profiles)} profili sintetici...\n")

    results = []
    for i, profile in enumerate(profiles, 1):
        row = evaluate_profile(profile)
        results.append(row)
        status_icon = "✓" if row["overall_pass"] else "✗"
        print(f"[{i:2d}/{len(profiles)}] {status_icon} {profile['id']}  "
              f"score={row['match_score']:.2f}  "
              f"country={row['country_match']} budget={row['budget_match']} level={row['level_match']}  "
              f"→ {row['returned_countries'][:50]}")
        time.sleep(REQUEST_DELAY)

    # Salva CSV
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    # Summary
    total        = len(df)
    passed       = df["overall_pass"].sum()
    avg_score    = df["match_score"].mean()
    avg_country  = df["country_match"].mean()
    avg_budget   = df["budget_match"].mean()
    avg_level    = df["level_match"].mean()
    clarif       = (df["status"] == "clarification_needed").sum()

    print("\n" + "=" * 60)
    print("RISULTATI SCORER EVALUATION")
    print("=" * 60)
    print(f"Profili valutati:       {total}")
    print(f"Overall pass (≥2/3):    {passed}/{total}  ({100*passed/total:.1f}%)")
    print(f"Match score medio:      {avg_score:.3f}")
    print(f"  Country match:        {avg_country:.3f}  ({100*avg_country:.1f}%)")
    print(f"  Budget match:         {avg_budget:.3f}  ({100*avg_budget:.1f}%)")
    print(f"  Level match:          {avg_level:.3f}  ({100*avg_level:.1f}%)")
    print(f"Clarification richieste: {clarif}")
    print(f"\nReport: {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
