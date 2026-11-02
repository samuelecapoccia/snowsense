"""
Componente 4 — Generazione Spiegazioni

Per ciascuno dei top-3 resort restituiti dal Componente 3 genera una spiegazione
testuale personalizzata in italiano, ancorata alla query originale dell'utente.

Non una descrizione generica del resort: una risposta a "perché questo resort
è giusto per te adesso".

Uso:
    from backend.core.explainer import SnowSenseExplainer
    explainer = SnowSenseExplainer()
    results = explainer.explain(
        user_query="Cerco qualcosa vicino a Roma per bambini, budget basso",
        intent={"level": "beginner", "budget": "low", "city_reference": "Roma", ...},
        resorts=[...],   # output di HybridScorer.rank()
    )
    # results: lista di dict resort arricchiti con campo "explanation"
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_TOKENS = 300  # spiegazione concisa: 3-5 frasi per resort
TEMPERATURE = 0.4  # leggermente creativo ma coerente

# ---------------------------------------------------------------------------
# Dizionari di localizzazione per il prompt
# ---------------------------------------------------------------------------

_LEVEL_IT = {
    "beginner": "principiante",
    "intermediate": "intermedio",
    "expert": "esperto",
}
_BUDGET_IT = {
    "low": "basso (preferibilmente sotto i 35€)",
    "medium": "medio (35-55€)",
    "high": "alto (oltre 55€ non è un problema)",
}
_EXP_IT = {
    "family": "vacanza in famiglia con bambini",
    "apres_ski": "après-ski e vita notturna",
    "sport": "esperienza sportiva su piste impegnative",
    "panoramic": "paesaggi e panorami alpini",
}

# ---------------------------------------------------------------------------
# Costruzione del prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    user_query: str,
    intent: Dict[str, Any],
    resort: Dict[str, Any],
) -> str:
    """
    Costruisce il prompt per generare la spiegazione di un singolo resort.
    Include: query originale, preferenze estratte, dati strutturati del resort.
    """
    # Sezione preferenze utente
    prefs = []
    if intent.get("level"):
        prefs.append(
            f"- Livello sciatore: {_LEVEL_IT.get(intent['level'], intent['level'])}"
        )
    if intent.get("budget"):
        prefs.append(f"- Budget: {_BUDGET_IT.get(intent['budget'], intent['budget'])}")
    if intent.get("experience_type"):
        prefs.append(
            f"- Tipo esperienza cercata: {_EXP_IT.get(intent['experience_type'], intent['experience_type'])}"
        )
    if intent.get("city_reference"):
        prefs.append(f"- Città di partenza dell'utente: {intent['city_reference']}")
    if intent.get("snowpark") is True:
        prefs.append("- Snowpark: richiesto")
    if intent.get("night_ski") is True:
        prefs.append("- Sci notturno: richiesto")

    prefs_text = "\n".join(prefs) if prefs else "Nessuna preferenza specifica estratta."

    # Sezione dati resort
    snowpark_txt = "Sì" if str(resort.get("Snowparks", "")).lower() == "yes" else "No"
    nightski_txt = "Sì" if str(resort.get("NightSki", "")).lower() == "yes" else "No"

    dist_line = ""
    dist_km = _extract_distance_km(resort)
    if dist_km is not None:
        dist_line = f"- Distanza da {intent.get('city_reference', 'città di partenza')}: circa {dist_km:.0f} km\n"

    resort_data = (
        f"- Resort: {resort['Resort']} ({resort['Country']})\n"
        f"- Quota: {resort['LowestPoint']}m – {resort['HighestPoint']}m\n"
        f"- Piste: {resort['BeginnerSlope']}km facili, "
        f"{resort['IntermediateSlope']}km intermedie, "
        f"{resort['DifficultSlope']}km difficili "
        f"(totale {resort['TotalSlope']}km)\n"
        f"- Ski pass adulto: {resort['DayPassPriceAdult']}€\n"
        f"- Snowpark: {snowpark_txt}  |  Sci notturno: {nightski_txt}\n"
        f"- Impianti: {resort['TotalLifts']} (capacità {resort['LiftCapacity']} pers/ora)\n"
        f"{dist_line}"
    )

    return (
        f"Un utente ha scritto questa richiesta:\n"
        f'"{user_query}"\n\n'
        f"Preferenze estratte:\n{prefs_text}\n\n"
        f"Dati del resort consigliato:\n{resort_data}\n"
        f"Scrivi una spiegazione personalizzata in italiano (3-4 frasi) che risponda "
        f"a 'perché questo resort è la scelta giusta per te'. "
        f"Sii specifico: cita dati concreti del resort (km piste, prezzo, quota, distanza se disponibile). "
        f"Non usare frasi generiche. Non ripetere tutti i dati — seleziona quelli rilevanti per questa richiesta. "
        f"Non aggiungere prefissi tipo 'Certo!' o 'Ecco la spiegazione:'. Vai diretto."
    )


def _extract_distance_km(resort: Dict[str, Any]) -> Optional[float]:
    """
    Recupera la distanza in km dal campo scores se presente,
    oppure la calcola dal dist_score (approssimazione inversa: score = 1 - km/1500).
    """
    scores = resort.get("scores", {})
    dist_score = scores.get("distance")
    if dist_score is not None and dist_score != 0.5:
        # Inversione approssimata: score = max(0, 1 - km/1500)
        km = (1.0 - float(dist_score)) * 1500.0
        return round(km, 0) if 0 < km < 1500 else None
    return None


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class SnowSenseExplainer:
    """
    Genera spiegazioni personalizzate per i resort raccomandati.

    Input:
        user_query  — testo originale dell'utente
        intent      — dict estratto dal Componente 1 (SnowSenseNLU)
        resorts     — lista di dict dal Componente 3 (HybridScorer.rank())

    Output:
        lista di dict identici ai resort in input, arricchiti con campo "explanation"
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key.startswith("sk-..."):
            raise EnvironmentError(
                "DEEPSEEK_API_KEY non configurata. Copia .env.example in .env e inserisci la chiave."
            )
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def explain(
        self,
        user_query: str,
        intent: Dict[str, Any],
        resorts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Arricchisce ogni resort con una spiegazione testuale personalizzata.

        Args:
            user_query: testo originale dell'utente
            intent:     dict con le preferenze estratte (Componente 1)
            resorts:    lista di resort (output di HybridScorer.rank())

        Returns:
            stessa lista con campo "explanation" aggiunto a ogni dict
        """
        results = []
        for resort in resorts:
            explanation = self._generate_explanation(user_query, intent, resort)
            enriched = dict(resort)
            enriched["explanation"] = explanation
            results.append(enriched)
        return results

    # ------------------------------------------------------------------
    # Generazione singola spiegazione
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        user_query: str,
        intent: Dict[str, Any],
        resort: Dict[str, Any],
    ) -> str:
        prompt = _build_prompt(user_query, intent, resort)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sei un esperto di stazioni sciistiche europee. "
                            "Scrivi spiegazioni concise, personalizzate e in italiano. "
                            "Usa dati concreti. Sii diretto."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Spiegazione non disponibile: {exc}]"
