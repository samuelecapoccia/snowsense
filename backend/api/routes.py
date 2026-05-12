"""
API routes SnowSense.
Pipeline completa: NLU → Scoring → Explanation → Exploration.
"""

from typing import Optional, Union
from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import (
    RecommendRequest,
    RecommendResponse,
    ClarificationResponse,
    ExtractedIntent,
    ResortResult,
    ResortScores,
    HealthResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Messaggi di chiarimento localizzati
# ---------------------------------------------------------------------------

_MISSING_FIELD_IT = {
    "level":           "il tuo livello sciistico (principiante, intermedio o esperto)",
    "budget":          "il budget giornaliero per lo ski pass",
    "experience_type": "il tipo di esperienza che cerchi (famiglia, sportivo, après-ski, panorami)",
}

_CLARIFICATION_TMPL = (
    "Per consigliarti al meglio ho bisogno di sapere: {fields}. "
    "Puoi specificarlo nella prossima risposta?"
)


def _build_clarification_message(missing_fields: list) -> str:
    labels = [_MISSING_FIELD_IT.get(f, f) for f in missing_fields]
    if len(labels) == 1:
        return _CLARIFICATION_TMPL.format(fields=labels[0])
    return _CLARIFICATION_TMPL.format(fields=", ".join(labels[:-1]) + " e " + labels[-1])


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    scorer = getattr(request.app.state, "scorer", None)
    return HealthResponse(status="ok", scorer_loaded=scorer is not None)


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

@router.post(
    "/recommend",
    tags=["pipeline"],
    responses={
        200: {"description": "Raccomandazioni o richiesta di chiarimento"},
        422: {"description": "Query vuota o strategia non valida"},
        500: {"description": "Errore interno pipeline"},
    },
)
def recommend(
    body: RecommendRequest,
    request: Request,
) -> Union[RecommendResponse, ClarificationResponse]:
    """
    Pipeline completa SnowSense:
      1. NLU    — estrae intent dalla query
      2. Scoring — classifica i resort per l'intent
      3. Explanation — genera spiegazioni personalizzate
      4. Exploration — applica epsilon-greedy se profilo presente
    """
    nlu         = request.app.state.nlu
    scorer      = request.app.state.scorer
    explainer   = request.app.state.explainer
    personalizer = request.app.state.personalizer
    all_df      = request.app.state.all_df

    # ------------------------------------------------------------------
    # 1. NLU — estrazione intent
    # ------------------------------------------------------------------
    try:
        raw_intent = nlu.extract(body.query, strategy=body.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NLU error: {exc}")

    intent_schema = ExtractedIntent(**{k: raw_intent.get(k) for k in ExtractedIntent.model_fields})

    # ------------------------------------------------------------------
    # 2. Chiarimento — se query troppo vaga
    # Safety net: ignora needs_clarification se l'intent contiene almeno
    # un vincolo discriminante (indipendente dall'output LLM).
    # ------------------------------------------------------------------
    _DISCRIMINATING_FIELDS = ("level", "budget", "city_reference", "country",
                               "experience_type", "snowpark", "night_ski")
    has_any_constraint = any(raw_intent.get(f) is not None for f in _DISCRIMINATING_FIELDS)

    # Profilo presente con almeno 1 sessione → abbastanza contesto, non chiedere chiarimenti
    profile_has_context = (
        body.user_profile is not None and
        body.user_profile.get("session_count", 0) >= 1 and
        any(body.user_profile.get(k) for k in
            ("level_counts", "budget_counts", "country_counts", "experience_counts"))
    )

    if raw_intent.get("needs_clarification") and not has_any_constraint and not profile_has_context:
        missing = raw_intent.get("missing_fields") or ["level", "budget", "experience_type"]
        return ClarificationResponse(
            missing_fields=missing,
            message=_build_clarification_message(missing),
        )

    # ------------------------------------------------------------------
    # 3. Contesto profilo utente → inject nel sistema (non usato da NLU qui,
    #    ma disponibile per future iterazioni conversazionali)
    # ------------------------------------------------------------------
    _llm_ctx = personalizer.build_llm_context(body.user_profile)

    # ------------------------------------------------------------------
    # 4. Scoring — top-3 resort
    # ------------------------------------------------------------------
    try:
        ranked = scorer.rank(raw_intent, top_k=3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")

    if not ranked:
        raise HTTPException(status_code=404, detail="Nessun resort trovato per i criteri specificati.")

    # ------------------------------------------------------------------
    # 5. Explanation — spiegazioni personalizzate
    # ------------------------------------------------------------------
    try:
        ranked_explained = explainer.explain(
            user_query=body.query,
            intent=raw_intent,
            resorts=ranked,
        )
    except Exception as exc:
        # Explanation non bloccante: restituisce resort senza spiegazione
        ranked_explained = [{**r, "explanation": None} for r in ranked]

    # ------------------------------------------------------------------
    # 6. Exploration — epsilon-greedy (richiede profilo)
    # ------------------------------------------------------------------
    try:
        final_resorts = personalizer.apply_exploration(
            ranked_resorts=ranked_explained,
            all_df=all_df,
            user_profile=body.user_profile,
            intent=raw_intent,
        )
    except Exception:
        final_resorts = ranked_explained

    # ------------------------------------------------------------------
    # 7. Serializzazione response
    # ------------------------------------------------------------------
    resort_results = []
    for r in final_resorts:
        scores_raw = r.get("scores", {})
        resort_results.append(
            ResortResult(
                resort_id=r["resort_id"],
                Resort=r["Resort"],
                Country=r["Country"],
                HighestPoint=r["HighestPoint"],
                LowestPoint=r["LowestPoint"],
                DayPassPriceAdult=r["DayPassPriceAdult"],
                TotalSlope=r["TotalSlope"],
                BeginnerSlope=r["BeginnerSlope"],
                IntermediateSlope=r["IntermediateSlope"],
                DifficultSlope=r["DifficultSlope"],
                Snowparks=r["Snowparks"],
                NightSki=r["NightSki"],
                TotalLifts=r["TotalLifts"],
                LiftCapacity=r["LiftCapacity"],
                SnowCannons=r["SnowCannons"],
                Latitude=r.get("Latitude"),
                Longitude=r.get("Longitude"),
                distance_km=r.get("distance_km"),
                scores=ResortScores(
                    embedding=scores_raw.get("embedding", 0.0),
                    mlp=scores_raw.get("mlp", 0.0),
                    distance=scores_raw.get("distance", 0.0),
                    final=scores_raw.get("final", 0.0),
                ),
                explanation=r.get("explanation"),
                is_exploration=r.get("is_exploration", False),
                exploration_reason=r.get("exploration_reason"),
            )
        )

    return RecommendResponse(intent=intent_schema, resorts=resort_results)
