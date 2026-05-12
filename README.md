# SnowSense

**Conversational Recommender System per località sciistiche europee**
Corso: Sistemi Intelligenti per Internet — Roma Tre, Magistrale AI/ML

---

## Struttura del Progetto

```
SnowSense/
├── backend/
│   ├── api/          # FastAPI routes
│   ├── core/         # Logica business (NLU, scoring, spiegazioni)
│   ├── data/         # Accesso al dataset, geocoding
│   └── models/       # Modelli Pydantic / ML
├── frontend/         # React + TypeScript
├── data/
│   ├── european_ski_resorts.csv          # Dataset originale (Kaggle)
│   └── european_ski_resorts_enriched.csv # Con coordinate (generato da geocoding)
├── notebooks/        # Esperimenti, analisi EDA
├── evaluation/       # Profili sintetici, ground truth, risultati comparativi
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Inserisci le chiavi API in .env
```

## Avvio Backend

```bash
uvicorn backend.main:app --reload
```

## Componenti

| # | Componente | Stato |
|---|---|---|
| 1 | Conversational NLU + studio prompting | da fare |
| 2 | Arricchimento geografico (geocoding) | da fare |
| 3 | Motore di scoring (ibrido embedding + MLP) | da fare |
| 4 | Generazione spiegazioni | da fare |
| 5 | Personalizzazione implicita | da fare |
| 6 | Valutazione con profili sintetici | da fare |
