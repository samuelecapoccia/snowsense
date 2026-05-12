"""
SnowSense — FastAPI entry point

Avvio:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /api/health        — stato del server
    POST /api/recommend     — pipeline completa
    GET  /docs              — Swagger UI
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.nlu import SnowSenseNLU
from backend.core.scoring import HybridScorer
from backend.core.explainer import SnowSenseExplainer
from backend.core.personalization import PersonalizationEngine

ROOT = Path(__file__).resolve().parent.parent
GEO_DATASET = ROOT / "data" / "european_ski_resorts_geo.csv"


# ---------------------------------------------------------------------------
# Lifespan — carica artefatti pesanti una volta sola all'avvio
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SnowSense] Inizializzazione componenti...")

    # NLU (leggero: solo client API)
    app.state.nlu = SnowSenseNLU()
    print("  [1/5] NLU pronto")

    # Scorer (pesante: carica embeddings .npy + mlp .pkl)
    app.state.scorer = HybridScorer()
    print("  [2/5] HybridScorer pronto")

    # Explainer (leggero: solo client API)
    app.state.explainer = SnowSenseExplainer()
    print("  [3/5] Explainer pronto")

    # Personalization engine (stateless)
    app.state.personalizer = PersonalizationEngine()
    print("  [4/5] PersonalizationEngine pronto")

    # DataFrame completo per exploration (riusato da scorer, caricato separato per personalizer)
    app.state.all_df = pd.read_csv(GEO_DATASET, index_col=0)
    print(f"  [5/5] Dataset caricato ({len(app.state.all_df)} resort)")

    print("[SnowSense] Pronto su http://localhost:8000")
    yield
    print("[SnowSense] Shutdown.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SnowSense API",
    description=(
        "Conversational Recommender System per stazioni sciistiche europee. "
        "Integra NLU (LLM), scoring ibrido (embedding + MLP + Haversine), "
        "generazione spiegazioni e personalizzazione implicita epsilon-greedy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permette al frontend React (localhost:3000/5173) di chiamare l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
