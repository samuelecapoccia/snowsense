"""
Pydantic schemas per request/response dell'API SnowSense.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query in linguaggio naturale dell'utente")
    strategy: str = Field(
        default="few_shot",
        description="Strategia NLU: 'zero_shot' | 'few_shot' | 'cot'",
    )
    user_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Profilo utente anonimo dal localStorage (opzionale)",
    )


# ---------------------------------------------------------------------------
# Intent estratto (Componente 1)
# ---------------------------------------------------------------------------

class ExtractedIntent(BaseModel):
    level:               Optional[str] = None
    budget:              Optional[str] = None
    city_reference:      Optional[str] = None
    country:             Optional[str] = None
    experience_type:     Optional[str] = None
    snowpark:            Optional[bool] = None
    night_ski:           Optional[bool] = None
    needs_clarification: bool = False
    missing_fields:      List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Resort con spiegazione
# ---------------------------------------------------------------------------

class ResortScores(BaseModel):
    embedding: float
    mlp:       float
    distance:  float
    final:     float


class ResortResult(BaseModel):
    resort_id:          int
    Resort:             str
    Country:            str
    HighestPoint:       int
    LowestPoint:        int
    DayPassPriceAdult:  float
    TotalSlope:         float
    BeginnerSlope:      float
    IntermediateSlope:  float
    DifficultSlope:     float
    Snowparks:          str
    NightSki:           str
    TotalLifts:         int
    LiftCapacity:       int
    SnowCannons:        int
    Latitude:           Optional[float] = None
    Longitude:          Optional[float] = None
    distance_km:        Optional[float] = None
    scores:             ResortScores
    explanation:        Optional[str] = None
    is_exploration:     bool = False
    exploration_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Response principale
# ---------------------------------------------------------------------------

class ClarificationResponse(BaseModel):
    """Restituita quando la query è troppo vaga per raccomandare."""
    status:         str = "clarification_needed"
    missing_fields: List[str]
    message:        str


class RecommendResponse(BaseModel):
    status:   str = "ok"
    intent:   ExtractedIntent
    resorts:  List[ResortResult]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:        str = "ok"
    scorer_loaded: bool
