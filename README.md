<div align="center">

# ❄ SnowSense

**Conversational Recommender System per stazioni sciistiche europee**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--V3-FF6B35?style=flat-square)](https://deepseek.com)

*Corso: Sistemi Intelligenti per Internet — Roma Tre, Magistrale AI/ML*

</div>

---

## Cos'è SnowSense

SnowSense è un sistema di raccomandazione conversazionale che permette all'utente di descrivere in linguaggio naturale cosa cerca in una vacanza sciistica. Il sistema comprende la richiesta, chiede chiarimenti se necessario, raccomanda i 3 resort più adatti da un catalogo di 376 stazioni europee, e spiega in italiano perché ognuno è la scelta giusta.

```
"Cerco qualcosa vicino a Roma per bambini, spendo poco"
          ↓
┌─────────────────────────────────────────────────┐
│  level: beginner  budget: low  city: Roma        │
│  experience_type: family  country: Italy         │
└─────────────────────────────────────────────────┘
          ↓
  1. Roccaraso       — 160km, 32€/giorno, 30km piste facili
  2. Campo Felice    — 140km, 28€/giorno, 20km piste verdi
  3. Ovindoli        — 148km, 29€/giorno, piste adatte ai bimbi
```

---

## Architettura

Il sistema è organizzato in **5 componenti** che formano una pipeline sequenziale:

```
Query utente (testo libero in italiano)
        │
        ▼
┌───────────────────────────────┐
│  COMPONENTE 1 — NLU           │  DeepSeek-V3 (OpenAI-compatible API)
│  Estrazione intent strutturato│  Strategie: zero-shot / few-shot / CoT
└───────────────┬───────────────┘
                │  needs_clarification? → domanda di chiarimento
                ▼
┌───────────────────────────────┐
│  COMPONENTE 2 — Geo           │  Nominatim (OpenStreetMap, gratuito)
│  Calcolo distanza Haversine   │  Geocoding città + formula sferico
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  COMPONENTE 3 — Hybrid Scorer │  Embedding cosine + MLP + Distanza
│  Ranking multi-criterio       │  Opzione D — approccio ibrido
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  COMPONENTE 4 — Explainer     │  DeepSeek-V3 genera 3-4 frasi
│  Spiegazioni personalizzate   │  Ancorate alla query originale
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  COMPONENTE 5 — Personalization│  Profilo anonimo in localStorage
│  Epsilon-greedy exploration   │  ε = 0.20, nessun login richiesto
└───────────────────────────────┘
        │
        ▼
Top-3 resort ordinati con spiegazioni
```

---

## Componente 1 — Conversational NLU

Trasforma una query in linguaggio libero in un oggetto JSON strutturato, gestendo ambiguità e richiedendo chiarimenti quando necessario.

**Schema di output:**
```json
{
  "level":            "beginner" | "intermediate" | "expert" | null,
  "budget":           "low" | "medium" | "high" | null,
  "city_reference":   "Roma" | null,
  "country":          "Italy" | null,
  "experience_type":  "family" | "apres_ski" | "sport" | "panoramic" | null,
  "snowpark":         true | false | null,
  "night_ski":        true | false | null,
  "needs_clarification": false,
  "missing_fields":   []
}
```

**Studio comparativo di prompt engineering** su 50 profili sintetici con ground truth noto:

| Strategia | Level | Budget | Country | Experience | Night ski | **Overall** |
|-----------|-------|--------|---------|------------|-----------|-------------|
| Zero-shot | 92.0% | 98.0% | 100.0% | 98.0% | 83.3% | **95.9%** |
| Few-shot (5 esempi) | 94.0% | 98.0% | 100.0% | 100.0% | 83.3% | **96.5%** |
| Chain-of-Thought | 90.0% | 96.0% | 100.0% | 94.0% | 66.7% | **92.4%** |

La strategia **few-shot** vince sull'overall con 0% di false positive su city_reference e snowpark. Il CoT, pur ragionando esplicitamente, tende a sovra-interpretare la query per night_ski.

---

## Componente 2 — Arricchimento Geografico

Il dataset Kaggle non include coordinate. Lo script `backend/data/geocode_resorts.py` arricchisce ogni resort con latitudine e longitudine via **Nominatim (OpenStreetMap)**, senza API key.

Quando l'utente menziona una città di partenza, il sistema geocodifica la città al momento della query e calcola la distanza con la **formula di Haversine**:

```python
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlam/2)**2
    return R * 2 * arctan2(sqrt(a), sqrt(1 - a))
```

---

## Componente 3 — Hybrid Scoring Engine (Opzione D)

Il motore di ranking combina tre segnali:

### Embedding Similarity
Ogni resort viene descritto in italiano (*"Stazione sciistica in Austria, quota 2000m, piste prevalentemente intermedie, prezzo 45€, snowpark disponibile"*). Sia la descrizione del resort che il testo della query vengono trasformati in vettori con [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Il ranking usa la **cosine similarity**.

### MLP Score
Una rete neurale `[64, 32]` è addestrata su **18.800 coppie sintetiche** (50 profili × 376 resort), con label generata da una funzione rule-based deterministica. L'input è un vettore da 22 feature:
- **Profile (12):** level one-hot (3), budget one-hot (3), experience one-hot (4), snowpark ±1, night_ski ±1
- **Resort (10):** price norm, altitude norm, beginner/intermediate/difficult ratio, total slope norm, snowpark bool, night_ski bool, lift capacity norm, snow cannon norm

### Pesi Ibridi
| Scenario | Embedding | MLP | Distanza |
|----------|-----------|-----|----------|
| Senza città di partenza | 37.5% | 62.5% | — |
| Con città di partenza | 15.0% | 35.0% | **50.0%** |

Quando l'utente specifica da dove parte, la distanza domina: il resort geograficamente raggiungibile vince su quello "perfetto ma lontano".

**Risultati valutazione scorer** (50 profili con ground truth):

| Metrica | Valore |
|---------|--------|
| Overall pass rate | **98.0%** |
| Country match | 98.0% |
| Budget match | **100.0%** |
| Level match | **100.0%** |

---

## Componente 4 — Generazione Spiegazioni

Per ciascuno dei 3 resort, DeepSeek-V3 genera una spiegazione in italiano di 3-4 frasi, ancorata alla query originale e ai dati strutturati del resort. Non una descrizione generica — una risposta a *"perché questo resort è giusto per te adesso"*.

**Esempio:**
> *"Roccaraso è la scelta più adatta alla tua richiesta. Si trova a circa 160km da Roma, raggiungibile in meno di due ore. Con 30km di piste prevalentemente facili e intermediate è ideale per chi si avvicina allo sci con bambini. Il giornaliero adulto a 32€ rientra nel budget indicato."*

---

## Componente 5 — Personalizzazione Implicita

- Nessun login, nessun database, nessun account
- Il browser salva un **profilo anonimo in localStorage** aggiornato ad ogni ricerca
- Il profilo traccia: livello prevalente, budget tipico, paesi visitati, tipo esperienza, preferenze snowpark/notturno
- Alla ricerca successiva il profilo viene inviato come contesto all'API e iniettato nel prompt LLM
- **Strategia epsilon-greedy** (ε = 0.20): con probabilità 20%, l'ultimo resort della lista viene sostituito con uno che esce dal profilo abituale, segnalato con il badge *"Fuori dalla tua zona abituale — potrebbe sorprenderti"*

---

## Dataset

**[european_ski_resorts.csv](https://www.kaggle.com/datasets/thomasnibb/european-ski-resorts)** — 376 stazioni sciistiche europee (Kaggle)

| Colonna | Descrizione |
|---------|-------------|
| `Resort`, `Country` | Nome e paese |
| `HighestPoint`, `LowestPoint` | Quota min/max (m) |
| `DayPassPriceAdult` | Ski pass adulto (€) |
| `BeginnerSlope`, `IntermediateSlope`, `DifficultSlope`, `TotalSlope` | Km per difficoltà |
| `Snowparks`, `NightSki` | Yes/No |
| `TotalLifts`, `LiftCapacity`, `SnowCannons` | Infrastrutture |

Il file `european_ski_resorts_geo.csv` aggiunge `Latitude` e `Longitude` tramite geocoding automatico.

---

## Stack Tecnologico

| Layer | Tecnologie |
|-------|-----------|
| **Backend** | Python 3.13, FastAPI, Uvicorn |
| **LLM** | DeepSeek-V3 via OpenAI-compatible SDK |
| **Embeddings** | `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) |
| **ML** | scikit-learn MLP, NumPy, Pandas |
| **Geo** | geopy / Nominatim (OpenStreetMap) |
| **Frontend** | React 19, TypeScript, Vite |
| **Dataset** | european_ski_resorts.csv (Kaggle) |

---

## Struttura del Repository

```
SnowSense/
├── backend/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── api/
│   │   └── routes.py            # POST /recommend, GET /health
│   ├── core/
│   │   ├── nlu.py               # Componente 1 — NLU (3 strategie)
│   │   ├── scoring.py           # Componente 3 — Hybrid Scorer
│   │   ├── explainer.py         # Componente 4 — Generazione spiegazioni
│   │   └── personalization.py   # Componente 5 — Epsilon-greedy
│   ├── data/
│   │   ├── geocode_resorts.py   # Componente 2 — Geocoding one-time
│   │   └── build_scorer.py      # Build artefatti MLP + embeddings
│   └── models/
│       └── schemas.py           # Modelli Pydantic I/O
├── frontend/
│   └── src/
│       ├── App.tsx              # Layout principale, hero, snow animation
│       ├── components/
│       │   ├── SearchBar.tsx    # Campo di input principale
│       │   ├── ResortCard.tsx   # Card resort con glassmorphism
│       │   ├── SkeletonCard.tsx # Skeleton loader animato
│       │   └── ClarificationBubble.tsx  # Chat bubble per chiarimenti
│       ├── hooks/
│       │   └── useRecommend.ts  # State machine: idle→loading→results
│       ├── lib/
│       │   └── userProfile.ts   # Profilo utente in localStorage
│       └── types/
│           └── api.ts           # Tipi TypeScript per l'API
├── data/
│   ├── european_ski_resorts.csv          # Dataset originale
│   ├── european_ski_resorts_geo.csv      # Dataset + coordinate
│   ├── synthetic_profiles.json          # 50 profili sintetici per eval
│   └── scorer_artifacts/
│       ├── resort_embeddings.npy        # Embeddings pre-calcolati
│       ├── mlp_model.pkl                # MLP addestrato
│       ├── feature_scaler.pkl           # StandardScaler
│       └── resort_index.json            # Mapping index → riga dataset
├── evaluation/
│   ├── generate_synthetic_profiles.py   # Genera i 50 profili di test
│   ├── evaluate_nlu.py                  # Confronto zero-shot/few-shot/CoT
│   ├── eval_scorer.py                   # Valutazione Hybrid Scorer
│   ├── nlu_report.md                    # Risultati NLU
│   └── scorer_results.csv               # Risultati scorer
├── scripts/
│   ├── fix_coordinates.py               # Correzione manuale coordinate
│   └── verify_map.py                    # Mappa HTML per verifica visiva
├── requirements.txt
└── .env.example
```

---

## Setup e Avvio

### Prerequisiti
- Python 3.13+
- Node.js 20+
- Una chiave API [DeepSeek](https://platform.deepseek.com)

### 1. Backend

```bash
# Clona e installa dipendenze
git clone https://github.com/samuelecapoccia/snowsense.git
cd snowsense

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configura le variabili d'ambiente
cp .env.example .env
# Inserisci DEEPSEEK_API_KEY nel file .env

# (Solo prima volta) Costruisci gli artefatti del modello
python -m backend.data.build_scorer

# Avvia il server
uvicorn backend.main:app --reload --port 8000
```

L'API è disponibile su `http://localhost:8000`.  
Swagger UI: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Il frontend è disponibile su `http://localhost:5173`.

### 3. (Opzionale) Rigenera le coordinate geografiche

```bash
python -m backend.data.geocode_resorts
```

> Il file `european_ski_resorts_geo.csv` è già incluso nel repo — questo step è necessario solo se vuoi rigenerarlo.

---

## API Reference

### `POST /api/recommend`

Esegue la pipeline completa e restituisce raccomandazioni o una richiesta di chiarimento.

**Request:**
```json
{
  "query": "Cerco qualcosa vicino a Roma per bambini, spendo poco",
  "strategy": "few_shot",
  "user_profile": null
}
```

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `query` | `string` | — | Testo libero dell'utente in italiano |
| `strategy` | `"zero_shot"` \| `"few_shot"` \| `"cot"` | `"few_shot"` | Strategia NLU |
| `user_profile` | `object \| null` | `null` | Profilo anonimo dal localStorage |

**Response — raccomandazioni:**
```json
{
  "intent": { "level": "beginner", "budget": "low", "city_reference": "Roma", ... },
  "resorts": [
    {
      "Resort": "Roccaraso",
      "Country": "Italy",
      "DayPassPriceAdult": 32.0,
      "TotalSlope": 30.0,
      "distance_km": 162.3,
      "explanation": "Roccaraso è la scelta più adatta...",
      "is_exploration": false,
      "scores": { "embedding": 0.71, "mlp": 0.68, "distance": 0.89, "final": 0.81 }
    }
  ]
}
```

**Response — chiarimento:**
```json
{
  "missing_fields": ["level", "budget"],
  "message": "Per consigliarti al meglio ho bisogno di sapere: il tuo livello sciistico e il budget giornaliero."
}
```

### `GET /api/health`

```json
{ "status": "ok", "scorer_loaded": true }
```

---

## Valutazione

### NLU — Confronto strategie di prompting

Eseguito su 50 profili sintetici con ground truth noto, generati da `evaluation/generate_synthetic_profiles.py`.

```bash
python evaluation/evaluate_nlu.py
```

Output in `evaluation/nlu_report.md`.

### Hybrid Scorer

```bash
python evaluation/eval_scorer.py
```

Output in `evaluation/scorer_results.csv`.

---

## Autori

Progetto realizzato per il corso **Sistemi Intelligenti per Internet** — Laurea Magistrale in Intelligenza Artificiale, Università Roma Tre.

| Componente | Responsabile |
|---|---|
| Componente 1 — NLU e studio prompting | Samuele & Pavlo |
| Componente 2 — Arricchimento geografico | Samuele |
| Componente 3 — Motore di scoring ibrido | Pavlo |
| Componente 4 — Generazione spiegazioni | Samuele & Pavlo |
| Componente 5 — Personalizzazione | Samuele |
| Componente 6 — Profili sintetici e valutazione | Samuele & Pavlo |
| Frontend React + TypeScript | Samuele & Pavlo |

---

## Licenza

Dataset originale: [European Ski Resorts — Kaggle](https://www.kaggle.com/datasets/thomasnibb/european-ski-resorts) (licenza CC BY 4.0)
