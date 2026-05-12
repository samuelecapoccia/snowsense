# SnowSense — Technical Report

**Corso:** Sistemi Intelligenti per Internet — Roma Tre, Magistrale AI/ML  
**Sistema:** Conversational Recommender System per stazioni sciistiche europee  
**Versione:** 1.0.0

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Component 1 — Conversational NLU](#4-component-1--conversational-nlu)
5. [Component 2 — Geographic Enrichment](#5-component-2--geographic-enrichment)
6. [Component 3 — Hybrid Scoring Engine](#6-component-3--hybrid-scoring-engine)
7. [Component 4 — Explanation Generation](#7-component-4--explanation-generation)
8. [Component 5 — Implicit Personalization](#8-component-5--implicit-personalization)
9. [Component 6 — Synthetic Profiles & Evaluation](#9-component-6--synthetic-profiles--evaluation)
10. [API Layer](#10-api-layer)
11. [Frontend](#11-frontend)
12. [Dependencies](#12-dependencies)
13. [Environment Variables](#13-environment-variables)
14. [Startup Instructions](#14-startup-instructions)
15. [Build Artifacts](#15-build-artifacts)

---

## 1. System Overview

SnowSense is a conversational recommender system for European ski resorts. The user describes in natural language what they are looking for — the system understands the intent, asks clarifying questions if needed, ranks the 372 available resorts, and explains each recommendation in personalized Italian prose.

### Pipeline

```
User query (natural language)
        ↓
[Component 1] NLU — intent extraction via LLM (zero-shot / few-shot / CoT)
        ↓
[Clarification] — if query is too vague, ask for missing critical fields
        ↓
[Component 2] Geographic enrichment — Haversine distance from city reference
        ↓
[Component 3] Hybrid scoring — cosine embedding similarity + MLP + distance
        ↓
[Component 4] Explanation generation — personalized Italian prose via LLM
        ↓
[Component 5] Epsilon-greedy exploration — inject one "surprise" resort if profile present
        ↓
JSON response → React frontend
```

### Key Technical Claims

| Property | Value |
|---|---|
| LLM provider | DeepSeek-V3 (OpenAI-compatible API) |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| ML model | `sklearn.MLPRegressor` — hidden layers (64, 32), ReLU, Adam |
| Dataset | 376 European ski resorts (Kaggle) + Nominatim geocoding |
| Training samples | 18,800 (50 synthetic profiles × 376 resorts) |
| Exploration strategy | Epsilon-greedy (ε = 0.20) |
| Personalization storage | Browser `localStorage` — no login, no server-side state |
| NLU study | Comparative evaluation: zero-shot vs. few-shot vs. chain-of-thought |

---

## 2. Repository Structure

```
SnowSense/
├── backend/
│   ├── main.py                     # FastAPI entrypoint, lifespan, CORS
│   ├── api/
│   │   └── routes.py               # POST /recommend, GET /health
│   ├── core/
│   │   ├── nlu.py                  # Component 1 — NLU
│   │   ├── scoring.py              # Component 3 — HybridScorer
│   │   ├── explainer.py            # Component 4 — Explanation generation
│   │   └── personalization.py      # Component 5 — Exploration + LLM context
│   ├── data/
│   │   ├── geocode_resorts.py      # Component 2 — Nominatim geocoding script
│   │   └── build_scorer.py         # Component 3 — Artifact build script
│   └── models/
│       └── schemas.py              # Pydantic v2 request/response schemas
├── data/
│   ├── european_ski_resorts.csv    # Original dataset (Kaggle)
│   ├── european_ski_resorts_geo.csv # Dataset + Latitude/Longitude
│   ├── synthetic_profiles.json     # 50 synthetic user profiles (Component 6)
│   └── scorer_artifacts/
│       ├── resort_embeddings.npy   # Pre-computed resort embeddings (376 × 384)
│       ├── mlp_model.pkl           # Trained MLPRegressor
│       ├── feature_scaler.pkl      # StandardScaler for MLP input
│       ├── resort_index.json       # Mapping: array position → DataFrame index
│       └── build_summary.json      # Build metadata + MLP metrics
├── frontend/
│   ├── index.html                  # Outfit font (Google Fonts), meta tags
│   └── src/
│       ├── App.tsx                 # Root: idle hero (2-col) / active (centered)
│       ├── App.css                 # All styles — glassmorphism, animations
│       ├── components/
│       │   ├── SearchBar.tsx       # Query textarea + suggestion chips
│       │   ├── ResortCard.tsx      # Resort result card (featured: horizontal layout)
│       │   ├── SkeletonCard.tsx    # Loading placeholder cards
│       │   └── ClarificationBubble.tsx  # Chat-bubble clarification UI
│       ├── hooks/
│       │   └── useRecommend.ts     # State machine: idle/loading/clarification/results/error
│       ├── lib/
│       │   └── userProfile.ts      # localStorage profile: load/save/update/getContext
│       └── types/
│           └── api.ts              # TypeScript types matching backend schemas
├── requirements.txt
└── .env                            # API keys (not committed)
```

---

## 3. Dataset

### Source

**File:** `data/european_ski_resorts.csv`  
**Origin:** Kaggle — European Ski Resorts dataset  
**Rows:** 376 resorts, 18 columns + 2 added (Latitude, Longitude)

### Schema

| Column | Type | Description |
|---|---|---|
| `Resort` | str | Resort name |
| `Country` | str | Country name (English) |
| `HighestPoint` | int | Summit altitude (m) |
| `LowestPoint` | int | Base altitude (m) |
| `DayPassPriceAdult` | float | Adult day ski pass (€) |
| `BeginnerSlope` | float | Easy piste km |
| `IntermediateSlope` | float | Intermediate piste km |
| `DifficultSlope` | float | Hard piste km |
| `TotalSlope` | float | Total piste km |
| `Snowparks` | str | "Yes" / "No" |
| `NightSki` | str | "Yes" / "No" |
| `SurfaceLifts` | int | Surface lift count |
| `ChairLifts` | int | Chair lift count |
| `GondolaLifts` | int | Gondola lift count |
| `TotalLifts` | int | Total lift count |
| `LiftCapacity` | int | Persons per hour |
| `SnowCannons` | int | Snow cannon count |
| `Latitude` | float | Added via Nominatim geocoding |
| `Longitude` | float | Added via Nominatim geocoding |

### Countries Represented (18)

Italy, France, Austria, Switzerland, Germany, Spain, Andorra, Norway, Sweden, Finland, Poland, Bulgaria, Slovenia, Slovakia, Czech Republic, United Kingdom, Southern Russia, Siberia.

---

## 4. Component 1 — Conversational NLU

**File:** `backend/core/nlu.py`

### Purpose

Transforms a free-text Italian query into a structured intent dictionary with 9 fields, using the DeepSeek-V3 LLM.

### Intent Schema

```json
{
  "level":               "beginner" | "intermediate" | "expert" | null,
  "budget":              "low" | "medium" | "high" | null,
  "city_reference":      "<departure city>" | null,
  "country":             "<preferred country>" | null,
  "experience_type":     "family" | "apres_ski" | "sport" | "panoramic" | null,
  "snowpark":            true | false | null,
  "night_ski":           true | false | null,
  "needs_clarification": true | false,
  "missing_fields":      ["level", "budget", "experience_type"]
}
```

### Value Mapping (embedded in system prompt)

| Field | Value | Trigger words |
|---|---|---|
| `level` | `beginner` | mai sciato, prima volta, principiante |
| `level` | `intermediate` | sciatore discreto, intermedio |
| `level` | `expert` | esperto, piste nere, agonistico |
| `budget` | `low` | economico, spendo poco, <35€ |
| `budget` | `medium` | nella media, 35-55€ |
| `budget` | `high` | alto, non importa, lusso, >55€ |
| `experience_type` | `family` | bambini, famiglia |
| `experience_type` | `apres_ski` | vita notturna, festa, après-ski |
| `experience_type` | `sport` | piste difficili, km, tecnico |
| `experience_type` | `panoramic` | paesaggi, panorami, natura |
| `snowpark` | `true` | snowpark, freestyle, tricks |
| `needs_clarification` | `true` | ONLY when zero discriminating constraints |

---

### Class: `SnowSenseNLU`

```python
class SnowSenseNLU:
    def __init__(self, model: str = "deepseek-chat") -> None
    def extract(self, query: str, strategy: str = "few_shot") -> Dict[str, Any]
    def _zero_shot_messages(self, query: str) -> tuple[list, int, bool]
    def _few_shot_messages(self, query: str) -> tuple[list, int, bool]
    def _cot_messages(self, query: str) -> tuple[list, int, bool]
    def _call_llm(self, messages: List[Dict], max_tokens: int, force_json: bool) -> str
    def _parse_json(self, text: str) -> Dict[str, Any]
```

**`extract(query, strategy)`**
- `query`: free text in Italian
- `strategy`: `"zero_shot"` | `"few_shot"` | `"cot"`
- Returns: dict matching intent schema above
- Raises `ValueError` for invalid strategy

**`_call_llm(messages, max_tokens, force_json)`**
- Prepends system message (SCHEMA_DESCRIPTION) to all strategies
- `force_json=True` → adds `response_format: {"type": "json_object"}` (zero-shot, few-shot)
- `force_json=False` → free text output (CoT: model reasons then writes JSON)
- `temperature=0.0` for full reproducibility

**`_parse_json(text)`**
- Calls `_extract_json_block()` — finds first balanced `{...}` in text
- On parse failure: returns `_empty_intent()` with `_parse_error=True` and `_raw` (first 500 chars)
- Calls `_normalize_intent()` — validates enum values, converts booleans

---

### LLM Configuration

| Parameter | Value |
|---|---|
| Base URL | `https://api.deepseek.com` |
| Model | `deepseek-chat` (DeepSeek-V3) |
| Temperature | 0.0 |
| max_tokens (zero-shot / few-shot) | 512 |
| max_tokens (CoT) | 1024 |
| response_format | `json_object` (zero-shot, few-shot only) |

---

### Strategy 1: Zero-Shot

**System message (all strategies):**
```
Sei un assistente per stazioni sciistiche europee.
[SCHEMA_DESCRIPTION — see full text below]
```

**SCHEMA_DESCRIPTION (full text):**
```
Estrai le preferenze sciistiche dell'utente nel seguente formato JSON.
Usa null per ogni campo NON menzionato esplicitamente — non inventare valori.

{
  "level":            "beginner" | "intermediate" | "expert" | null,
  "budget":           "low" | "medium" | "high" | null,
  "city_reference":   "<città di partenza dell'utente>" | null,
  "country":          "<paese preferito es. Italy, France, Austria, Switzerland>" | null,
  "experience_type":  "family" | "apres_ski" | "sport" | "panoramic" | null,
  "snowpark":         true | false | null,
  "night_ski":        true | false | null,
  "needs_clarification": true | false,
  "missing_fields":   ["<campi critici mancanti>"]
}

Dizionario valori:
- level:           mai sciato / principiante = beginner | discreto / intermedio = intermediate | esperto / piste nere = expert
- budget:          economico / spendo poco / <35€ = low | nella media / 35-55€ = medium | alto / non importa / lusso / >55€ = high
- experience_type: bambini / famiglia = family | vita notturna / festa / après-ski = apres_ski | piste difficili / km / tecnico = sport | paesaggi / panorami / natura = panoramic
- snowpark:        true se menziona snowpark / freestyle / tricks; false se lo esclude; null se non detto
- night_ski:       true se menziona sci notturno / sciare la sera; null se non detto
- needs_clarification: true SOLO se la query non contiene NESSUN vincolo discriminante.
  Un vincolo discriminante è qualsiasi campo non null: level, budget, city_reference, country, experience_type, snowpark=true/false, night_ski=true/false.
  Se l'utente ha fornito ALMENO UN vincolo specifico → needs_clarification: false, anche se level/budget/experience_type sono null.
  Esempi: "voglio sciare" → true (zero vincoli). "sci notturno, non importa il paese" → false (night_ski=true è un vincolo). "vicino a Roma" → false (city_reference presente).
- missing_fields:  lista solo i campi critici mancanti (level, budget, experience_type), solo se needs_clarification è true
```

**User message (zero-shot):**
```
Query utente:
{query}
```

- `max_tokens`: 512
- `force_json`: True

---

### Strategy 2: Few-Shot

**User message (few-shot):** Prepends 5 example query→JSON pairs before the target query.

**Few-shot examples (verbatim):**

```
Query: "Siamo 4 amici che vogliono sciare in Austria la settimana prossima. Siamo tutti bravi, non ci spaventa niente. Non abbiamo problemi di budget."
JSON: {"level": "expert", "budget": "high", "city_reference": null, "country": "Austria", "experience_type": "sport", "snowpark": null, "night_ski": null, "needs_clarification": false, "missing_fields": []}

Query: "Prima esperienza sulla neve con mio figlio di 5 anni, abitiamo a Genova. Preferirei non spendere troppo."
JSON: {"level": "beginner", "budget": "low", "city_reference": "Genova", "country": null, "experience_type": "family", "snowpark": null, "night_ski": null, "needs_clarification": false, "missing_fields": []}

Query: "Voglio andare a sciare."
JSON: {"level": null, "budget": null, "city_reference": null, "country": null, "experience_type": null, "snowpark": null, "night_ski": null, "needs_clarification": true, "missing_fields": ["level", "budget", "experience_type"]}

Query: "Sono un freestyle rider, cerco il miglior snowpark europeo. Soldi non sono un problema, voglio anche lo sci notturno."
JSON: {"level": "expert", "budget": "high", "city_reference": null, "country": null, "experience_type": "sport", "snowpark": true, "night_ski": true, "needs_clarification": false, "missing_fields": []}

Query: "Weekend in montagna con mio marito, piste non troppo difficili, budget medio. Preferirei la Francia."
JSON: {"level": "intermediate", "budget": "medium", "city_reference": null, "country": "France", "experience_type": "panoramic", "snowpark": null, "night_ski": null, "needs_clarification": false, "missing_fields": []}
```

Then:
```
Ora estrai l'intent di questa query:
Query: "{query}"
```

- `max_tokens`: 512
- `force_json`: True

**Note:** These 5 examples are disjoint from the 50 evaluation profiles to avoid data leakage.

---

### Strategy 3: Chain-of-Thought (CoT)

**User message (CoT):**

```
Prima di rispondere, ragiona esplicitamente su ogni punto:

1. LIVELLO: La query descrive l'esperienza sciistica? (mai sciato/prima volta = beginner, sciatore discreto = intermediate, esperto/piste nere/agonistico = expert, non detto = null)
2. BUDGET: Ci sono indicatori di prezzo? (economico/poco/limitato = low, nella media/normale = medium, alto/non importa/lusso = high, non detto = null)
3. CITTÀ: L'utente menziona da dove parte o dove abita? (stringa o null)
4. PAESE: Si preferisce un paese specifico? (Italy/France/Austria/Switzerland/... o null)
5. TIPO ESPERIENZA: Cosa cerca principalmente? (bambini/famiglia = family; serata/festa = apres_ski; piste difficili/km/tecnico = sport; panorami/natura = panoramic; non chiaro = null)
6. SNOWPARK: Si menziona snowpark, freestyle o tricks? (true/false/null)
7. SCI NOTTURNO: Si menziona sciare di sera o di notte? (true/false/null)
8. CHIARIMENTO: La query contiene almeno un vincolo specifico (night_ski, snowpark, country, city_reference, level, budget, experience_type)?
   Se sì → needs_clarification: false. Solo se ZERO vincoli → needs_clarification: true.

Scrivi prima il ragionamento (breve, una riga per punto), poi il JSON finale.

Query utente:
{query}
```

- `max_tokens`: 1024
- `force_json`: False (model outputs free reasoning text + JSON block; `_extract_json_block()` locates the JSON)

---

### Helper Functions

```python
def _extract_json_block(text: str) -> str
    # Finds first balanced {…} in text. Raises ValueError if none found.

def _normalize_intent(raw: Dict[str, Any]) -> Dict[str, Any]
    # Validates enums: level ∈ {beginner,intermediate,expert,None}
    # Validates: budget ∈ {low,medium,high,None}
    # Validates: experience_type ∈ {family,apres_ski,sport,panoramic,None}
    # Normalizes booleans: "true"/"sì"/"1" → True; "false"/"no"/"0" → False; else None
    # Normalizes string fields: "null"/"none"/"" → None

def _empty_intent() -> Dict[str, Any]
    # Returns all-None/False intent (used as fallback on parse error)
```

---

### Clarification Logic (in routes.py)

`needs_clarification` from the LLM is **overridden** by two safety conditions:

1. **Has any constraint**: if any of `(level, budget, city_reference, country, experience_type, snowpark, night_ski)` is non-null → do NOT ask for clarification
2. **Profile has context**: if `user_profile.session_count >= 1` and at least one count dict is non-empty → do NOT ask for clarification (system has enough past context)

**Clarification message template:**
```
Per consigliarti al meglio ho bisogno di sapere: {fields}.
Puoi specificarlo nella prossima risposta?
```

Missing field labels:
- `level` → "il tuo livello sciistico (principiante, intermedio o esperto)"
- `budget` → "il budget giornaliero per lo ski pass"
- `experience_type` → "il tipo di esperienza che cerchi (famiglia, sportivo, après-ski, panorami)"

---

## 5. Component 2 — Geographic Enrichment

**File:** `backend/data/geocode_resorts.py`

### Purpose

One-time script that adds `Latitude` and `Longitude` columns to the dataset by querying Nominatim (OpenStreetMap). Output: `data/european_ski_resorts_geo.csv`.

### Configuration

| Parameter | Value |
|---|---|
| Geocoder | Nominatim (geopy) |
| User agent | `snowsense_geocoder_roma3` |
| Rate limit | 1.1 s between requests (Nominatim policy: 1 req/s) |
| Max retries per query | 3 |
| Request timeout | 10 s |
| Checkpoint interval | Every 10 resorts |

### Key Functions

```python
def geocode_resort(
    geocoder: Nominatim,
    resort: str,
    country: str
) -> Tuple[Optional[float], Optional[float]]
    # Tries multiple name variants for each resort.
    # Returns (lat, lon) or (None, None) on failure.

def _name_candidates(name: str) -> List[str]
    # Generates query variants from resort name.
    # Handles: parenthetical subnames "(Grandvalira)", slash-separated segments,
    # suffix stripping ("Glacier", "Gletscher", "ski resort", etc.).
    # Returns deduplicated list ordered from most to least specific.

def _is_indoor_or_dry(name: str) -> bool
    # Returns True for indoor/dry-slope facilities (no geographic coordinates).
    # Keywords: "indoor ski", "dry slope", "snow arena", "snowworld", "skidome"
```

### Country Normalization

Some dataset country names are not recognized by Nominatim:

| Dataset value | Nominatim query value |
|---|---|
| `Southern Russia` | `Russia` |
| `Siberia` | `Russia` |
| `United Kingdom` | `UK` |

### Execution

```bash
python -m backend.data.geocode_resorts
```

The script is **resumable**: on re-run it reads the existing output file and skips already-geocoded resorts.

---

## 6. Component 3 — Hybrid Scoring Engine

**Files:** `backend/core/scoring.py`, `backend/data/build_scorer.py`

### Architecture (Option D — Hybrid)

The final score is a weighted combination of three sub-scores:

#### Without city reference (no geographic constraint):
```
final = 0.375 × embed_score + 0.625 × mlp_score
```

#### With city reference (geographic constraint active):
```
final = 0.15 × embed_score + 0.35 × mlp_score + 0.50 × dist_score
```

All three sub-scores are in [0, 1].

---

### Sub-score 1: Embedding Similarity

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional, multilingual)
- **Resort text** (`resort_to_text(row)`): Italian description, e.g.:
  ```
  Stazione sciistica in Austria. Quota massima 1980m, quota minima 740m.
  Piste: 30.0km facili, 81.0km intermedie, 4.0km difficili (totale 115.0km).
  Ski pass adulto 52.0€. Snowpark disponibile. Impianti: 49 con capacità 75398 persone/ora.
  ```
- **Query text** (`intent_to_query_text(intent)`): Italian sentence built from intent fields, e.g.:
  ```
  Sciatore principiante. Vacanza in famiglia con bambini, piste facili. Budget economico (meno di 35€). Preferenza paese: Italy.
  ```
- **Scoring:** cosine similarity = dot product (both embeddings L2-normalized at build time)
- **Pre-computation:** resort embeddings computed once in `build_scorer.py`, stored as `resort_embeddings.npy`
- **Query embedding:** computed at inference time per request

---

### Sub-score 2: MLP Regressor

**Architecture:** `sklearn.neural_network.MLPRegressor`

| Hyperparameter | Value |
|---|---|
| Hidden layers | (64, 32) |
| Activation | ReLU |
| Solver | Adam |
| Learning rate | 0.001 |
| Max iterations | 300 |
| Early stopping | True |
| Validation fraction | 0.10 |
| Patience (n_iter_no_change) | 20 |
| random_state | 42 |

**Input:** 22-dimensional vector = profile features (12) + resort features (10)

#### Profile features (12 dimensions)

```python
def profile_to_features(intent: Dict[str, Any]) -> np.ndarray  # shape: (12,)
```

| Dimensions | Encoding |
|---|---|
| 0-2 | `level`: one-hot [beginner, intermediate, expert] — `[0,0,0]` if null |
| 3-5 | `budget`: one-hot [low, medium, high] — `[0,0,0]` if null |
| 6-9 | `experience_type`: one-hot [family, apres_ski, sport, panoramic] — `[0,0,0,0]` if null |
| 10 | `snowpark`: +1.0 (yes) / -1.0 (no) / 0.0 (null) |
| 11 | `night_ski`: +1.0 (yes) / -1.0 (no) / 0.0 (null) |

#### Resort features (10 dimensions)

```python
def resort_to_features(row: pd.Series, stats: Dict[str, float]) -> np.ndarray  # shape: (10,)
```

| Index | Feature | Normalization |
|---|---|---|
| 0 | `DayPassPriceAdult` | ÷ max_price |
| 1 | `HighestPoint` | ÷ max_highest |
| 2 | `BeginnerSlope / TotalSlope` | ratio 0-1 |
| 3 | `IntermediateSlope / TotalSlope` | ratio 0-1 |
| 4 | `DifficultSlope / TotalSlope` | ratio 0-1 |
| 5 | `TotalSlope` | ÷ max_slope, clipped at 1.0 |
| 6 | `Snowparks` | 1.0 if "yes", else 0.0 |
| 7 | `NightSki` | 1.0 if "yes", else 0.0 |
| 8 | `LiftCapacity` | ÷ max_lift_cap |
| 9 | `SnowCannons` | ÷ max_cannons |

Dataset statistics (computed at load time by `compute_dataset_stats(df)`):

```python
{
  "max_price":    float(df["DayPassPriceAdult"].max()),
  "max_highest":  float(df["HighestPoint"].max()),
  "max_slope":    float(df["TotalSlope"].max()),
  "max_lift_cap": float(df["LiftCapacity"].max()),
  "max_cannons":  float(df["SnowCannons"].max()),
}
```

**MLP output normalization:** raw prediction ÷ 10.0 → mlp_score ∈ [0, 1]

---

### Sub-score 3: Distance (Haversine)

```python
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float
    # R = 6371.0 km (Earth radius)
    # Standard haversine formula using numpy trigonometry
```

**Distance to score:**
```python
dist_score = max(0.0, 1.0 - dist_km / 1500.0)
# 0 km → 1.0 (best), 1500 km → 0.0 (cutoff)
```

City geocoding at query time:
```python
def geocode_city(city: str) -> Optional[tuple]
    # Uses Nominatim with user_agent="snowsense_scorer"
    # 1.1 s sleep after call (rate limit)
    # Returns (lat, lon) or None on failure
```

---

### Training Data Generation

```python
def rule_based_score(intent: Dict[str, Any], row: pd.Series) -> float
    # Returns a deterministic score in [0, 10] based on rules.
    # Used as training labels for the MLP.
    # Base score: 5.0
```

**Scoring rules (additive/subtractive from base 5.0):**

| Condition | Δ score |
|---|---|
| `level=beginner`, high beginner ratio | +3.0 × beg_ratio |
| `level=beginner`, difficult ratio > 40% | −1.5 |
| `level=expert`, high difficult ratio | +3.0 × diff_ratio |
| `level=expert`, TotalSlope < 30 km | −1.0 |
| `level=intermediate` | +2.0 × inter_ratio |
| `budget=low`, price ≤ 35€ | +2.0 |
| `budget=low`, price > 55€ | −3.0 |
| `budget=low`, 45€ < price ≤ 55€ | −1.0 |
| `budget=medium`, 35€ ≤ price ≤ 55€ | +2.0 |
| `budget=medium`, price > 65€ | −1.5 |
| `budget=high`, price ≥ 50€ | +1.0 |
| `snowpark=true`, snowpark present | +1.5 |
| `snowpark=true`, snowpark absent | −2.0 |
| `snowpark=false`, snowpark present | −0.5 |
| `night_ski=true`, night ski present | +1.5 |
| `night_ski=true`, night ski absent | −2.0 |
| `experience=sport` | +min(TotalSlope,300)/100 + diff_ratio |
| `experience=family` | +2.0×beg_ratio + min(lifts,50)/50 |
| `experience=panoramic` | +min(HighestPoint,3500)/1750 |
| `experience=apres_ski` | +min(LiftCapacity,150000)/75000 |

Final score clipped to [0.0, 10.0].

**Training dataset size:** 50 profiles × 376 resorts = **18,800 samples**  
**Train/test split:** 85% / 15% (random_state=42)

---

### Build Artifacts

```python
def main() -> None
    # 1. Loads dataset and synthetic_profiles.json
    # 2. Generates training data (profile × resort pairs)
    # 3. Trains MLP + StandardScaler
    # 4. Computes resort embeddings (sentence-transformers)
    # 5. Saves all artifacts to data/scorer_artifacts/
```

**Output files:**

| File | Content |
|---|---|
| `resort_embeddings.npy` | Float32 array, shape (376, 384) |
| `mlp_model.pkl` | Pickled `MLPRegressor` |
| `feature_scaler.pkl` | Pickled `StandardScaler` |
| `resort_index.json` | List of 376 DataFrame index values |
| `build_summary.json` | Build metadata, MLP RMSE/MAE metrics |

---

### Class: `HybridScorer`

```python
class HybridScorer:
    def __init__(self) -> None
        # Loads: dataset, embeddings .npy, mlp .pkl, scaler .pkl, resort_index .json
        # sentence-transformers loaded lazily on first call

    def rank(
        self,
        intent: Dict[str, Any],
        top_k: int = 3,
        country_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]
        # Returns top_k resorts sorted by final score descending.
        # Each dict contains all resort fields + scores dict + distance_km.

    def _get_embedder(self) -> SentenceTransformer
        # Lazy singleton for sentence-transformers model
```

**`rank()` return type per resort:**
```python
{
    "resort_id": int,
    "Resort": str,
    "Country": str,
    "HighestPoint": int,
    "LowestPoint": int,
    "DayPassPriceAdult": float,
    "TotalSlope": float,
    "BeginnerSlope": float,
    "IntermediateSlope": float,
    "DifficultSlope": float,
    "Snowparks": str,   # "Yes" / "No"
    "NightSki": str,    # "Yes" / "No"
    "TotalLifts": int,
    "LiftCapacity": int,
    "SnowCannons": int,
    "Latitude": Optional[float],
    "Longitude": Optional[float],
    "distance_km": Optional[float],
    "scores": {
        "embedding": float,  # cosine similarity [0,1]
        "mlp":       float,  # MLP prediction normalized [0,1]
        "distance":  float,  # distance score [0,1]
        "final":     float,  # weighted sum
    }
}
```

---

## 7. Component 4 — Explanation Generation

**File:** `backend/core/explainer.py`

### Purpose

For each of the top-3 ranked resorts, generate a personalized Italian explanation (3-4 sentences) anchored to the user's original query. Not a generic resort description — a direct answer to "why is this resort right for you, now".

### LLM Configuration

| Parameter | Value |
|---|---|
| Model | `deepseek-chat` (DeepSeek-V3) |
| Max tokens | 300 |
| Temperature | 0.4 |

### Class: `SnowSenseExplainer`

```python
class SnowSenseExplainer:
    def __init__(self, model: str = "deepseek-chat") -> None

    def explain(
        self,
        user_query: str,
        intent: Dict[str, Any],
        resorts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]
        # Adds "explanation" field to each resort dict.
        # On LLM error: explanation = "[Spiegazione non disponibile: {exc}]"

    def _generate_explanation(
        self,
        user_query: str,
        intent: Dict[str, Any],
        resort: Dict[str, Any],
    ) -> str
```

### Prompt Structure (per resort)

**System message:**
```
Sei un esperto di stazioni sciistiche europee. Scrivi spiegazioni concise, personalizzate e in italiano. Usa dati concreti. Sii diretto.
```

**User message (built by `_build_prompt()`):**
```
Un utente ha scritto questa richiesta:
"{user_query}"

Preferenze estratte:
- Livello sciatore: {level_it}
- Budget: {budget_it}
- Tipo esperienza cercata: {exp_it}
- Città di partenza dell'utente: {city_reference}
- Snowpark: richiesto       [if snowpark=True]
- Sci notturno: richiesto   [if night_ski=True]

Dati del resort consigliato:
- Resort: {Resort} ({Country})
- Quota: {LowestPoint}m – {HighestPoint}m
- Piste: {BeginnerSlope}km facili, {IntermediateSlope}km intermedie, {DifficultSlope}km difficili (totale {TotalSlope}km)
- Ski pass adulto: {DayPassPriceAdult}€
- Snowpark: Sì|No  |  Sci notturno: Sì|No
- Impianti: {TotalLifts} (capacità {LiftCapacity} pers/ora)
- Distanza da {city_reference}: circa {dist_km} km   [if city_reference present]

Scrivi una spiegazione personalizzata in italiano (3-4 frasi) che risponda a 'perché questo resort è la scelta giusta per te'. Sii specifico: cita dati concreti del resort (km piste, prezzo, quota, distanza se disponibile). Non usare frasi generiche. Non ripetere tutti i dati — seleziona quelli rilevanti per questa richiesta. Non aggiungere prefissi tipo 'Certo!' o 'Ecco la spiegazione:'. Vai diretto.
```

### Distance Recovery

`_extract_distance_km(resort)` recovers distance from the scoring sub-score via inverse formula:
```python
km = (1.0 - dist_score) * 1500.0
```
Returns `None` if `dist_score == 0.5` (neutral/missing) or result is out of range [0, 1500].

### Localization Dictionaries

```python
_LEVEL_IT  = {"beginner": "principiante", "intermediate": "intermedio", "expert": "esperto"}
_BUDGET_IT = {"low": "basso (preferibilmente sotto i 35€)", "medium": "medio (35-55€)", "high": "alto (oltre 55€ non è un problema)"}
_EXP_IT    = {"family": "vacanza in famiglia con bambini", "apres_ski": "après-ski e vita notturna", "sport": "esperienza sportiva su piste impegnative", "panoramic": "paesaggi e panorami alpini"}
```

---

## 8. Component 5 — Implicit Personalization

**Files:** `backend/core/personalization.py`, `frontend/src/lib/userProfile.ts`

### Architecture

- **No login, no account, no server-side state**
- Profile lives in browser `localStorage` under key `"snowsense_user_profile"`
- Sent as optional `user_profile` field in every API request
- Backend is **stateless**: receives profile, uses it, returns results — never stores it

### Profile Schema

```typescript
interface UserProfile {
  level_counts:      Partial<Record<"beginner"|"intermediate"|"expert", number>>;
  budget_counts:     Partial<Record<"low"|"medium"|"high", number>>;
  country_counts:    Record<string, number>;
  experience_counts: Partial<Record<"family"|"apres_ski"|"sport"|"panoramic", number>>;
  snowpark_pref:     boolean | null;   // last explicit value
  night_ski_pref:    boolean | null;
  query_history:     Array<{query: string, timestamp: string}>;  // max 20
  session_count:     number;
  last_updated:      string | null;   // ISO timestamp
}
```

### Frontend Profile Functions (`frontend/src/lib/userProfile.ts`)

```typescript
function emptyProfile(): UserProfile          // Returns zeroed profile
function loadProfile(): UserProfile           // Reads from localStorage, merges with empty
function saveProfile(profile: UserProfile): void  // Writes to localStorage with last_updated
function clearProfile(): void                 // Removes key from localStorage

function updateProfile(
  profile: UserProfile,
  intent: Intent,
  resorts: ResortResult[],
  query: string,
): UserProfile
  // Updates all count fields from intent
  // Increments country_counts from non-exploration resorts only
  // Updates snowpark_pref / night_ski_pref (last explicit value wins)
  // Appends to query_history (max 20 entries, rolling window)
  // Increments session_count

function getProfileContext(profile: UserProfile): UserProfile | null
  // Returns null if session_count < 2 (insufficient context)
  // Otherwise returns the full profile for the API payload

function dominant<T>(counts: Partial<Record<T, number>>): T | null
function topN<T>(counts, n: number): T[]
```

Profile is sent to the API when `session_count >= 2`. Backend activates personalization context when `session_count >= 1` (for clarification bypass) or `>= 2` (for LLM context injection).

---

### Backend Personalization (`backend/core/personalization.py`)

**Class: `PersonalizationEngine`**

```python
class PersonalizationEngine:
    def __init__(self, epsilon: float = 0.20) -> None

    def apply_exploration(
        self,
        ranked_resorts: List[Dict[str, Any]],
        all_df: pd.DataFrame,
        user_profile: Optional[Dict[str, Any]],
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]
        # Epsilon-greedy: with probability ε (0.20), replaces ranked_resorts[-1]
        # with a "surprising" resort from the full dataset.
        # No-op if: user_profile is None, or session_count < 3.

    def build_llm_context(self, user_profile: Optional[Dict[str, Any]]) -> str
        # Returns Italian context string to inject into LLM prompts.
        # Format: "[Contesto utente — N ricerche precedenti: livello; budget; zona; ...]"
        # Returns "" if session_count < 2.
```

### Epsilon-Greedy Exploration

**ε = 0.20** (20% chance per session after ≥ 3 sessions)

**Replacement strategy (priority order):**
1. **Different country**: find resort from non-dominant country → `reason: "country"` → text: *"Fuori dalla tua zona abituale — potrebbe sorprenderti."*
2. **Different budget tier**: find resort outside dominant budget range → `reason: "budget"` → text: *"Proposta fuori dal tuo range solito — il rapporto qualità-prezzo è ottimo."*
3. **Random fallback**: any resort not already shown → `reason: "default"` → text: *"Una proposta insolita rispetto alle tue ricerche precedenti — aperta all'esplorazione."*

Exploration resorts are flagged: `is_exploration=True`, `exploration_reason=<string>`.

**Important:** exploration resorts are excluded from `country_counts` updates in `updateProfile()` to prevent distorting the dominant profile.

### LLM Context Injection

When profile is sufficiently rich (≥ 2 sessions), `build_llm_context()` returns:
```
[Contesto utente — 5 ricerche precedenti: livello prevalente: intermedio; budget tipico: medio; esperienza preferita: sport; zone frequentate: Austria, France; interesse per snowpark. Tieni conto di questo profilo nelle raccomandazioni, senza menzionarlo esplicitamente.]
```

This string is built but currently available for future iterative use; it is computed by the route handler but not yet injected back into the NLU prompt (the variable `_llm_ctx` is prepared but the NLU is stateless per-call).

---

## 9. Component 6 — Synthetic Profiles & Evaluation

### Purpose

Provides ground truth for offline evaluation of both the NLU component (prompt strategy comparison) and the scoring component (resort ranking quality).

### Profile Structure

**File:** `data/synthetic_profiles.json` — 50 profiles

Each profile:
```json
{
  "id": "P001",
  "profile": {
    "level": "beginner",
    "budget": "low",
    "zona": "italy_center_south",
    "experience_type": "family",
    "city_reference": "Roma",
    "snowpark": false,
    "night_ski": false,
    "budget_range_eur": "<35€"
  },
  "query_it": "Ciao, siamo una famiglia con due bambini di 6 e 9 anni, partiamo da Roma. Non abbiamo mai sciato, cerchiamo qualcosa di economico e non troppo lontano. Dove ci consigliate?",
  "ground_truth_intent": {
    "level": "beginner",
    "budget": "low",
    "city_reference": "Roma",
    "experience_type": "family",
    "snowpark": null,
    "night_ski": null
  }
}
```

### Profile Dimensions (5)

| Dimension | Values |
|---|---|
| Level | beginner / intermediate / expert |
| Budget | low (<35€) / medium (35-55€) / high (>55€) |
| Zona | italy_north / italy_center_south / france / austria / no_preference |
| Experience | family / apres_ski / sport / panoramic |
| City reference | with / without |

### NLU Evaluation Metrics

For each of the 50 profiles × 3 strategies:

| Metric | Measurement |
|---|---|
| Exact match | Categorical fields: `level`, `budget`, `experience_type`, `country` |
| Binary accuracy | Boolean fields: `snowpark`, `night_ski` (null = correct if null in ground truth) |
| Clarification accuracy | `needs_clarification` match |
| Average accuracy | Mean across all non-null ground truth fields |

**Comparative table format:**

| Strategy | Level acc. | Budget acc. | Exp. acc. | Boolean acc. | Overall |
|---|---|---|---|---|---|
| Zero-shot | x% | x% | x% | x% | x% |
| Few-shot | x% | x% | x% | x% | x% |
| CoT | x% | x% | x% | x% | x% |

---

## 10. API Layer

**Files:** `backend/main.py`, `backend/api/routes.py`, `backend/models/schemas.py`

### Startup (lifespan)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlu         = SnowSenseNLU()
    app.state.scorer      = HybridScorer()       # loads embeddings + MLP
    app.state.explainer   = SnowSenseExplainer()
    app.state.personalizer = PersonalizationEngine()
    app.state.all_df      = pd.read_csv(GEO_DATASET, index_col=0)
    yield
```

All heavy objects are loaded once at startup and shared across requests via `app.state`.

### CORS

```python
allow_origins=["http://localhost:3000", "http://localhost:5173"]
```

### Endpoints

#### `GET /api/health`

**Response `HealthResponse`:**
```json
{"status": "ok", "scorer_loaded": true}
```

#### `POST /api/recommend`

**Request `RecommendRequest`:**
```json
{
  "query": "Cerco qualcosa vicino a Roma per bambini, budget basso",
  "strategy": "few_shot",
  "user_profile": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | str | required | Natural language query (min 1 char) |
| `strategy` | str | `"few_shot"` | `"zero_shot"` \| `"few_shot"` \| `"cot"` |
| `user_profile` | dict \| null | `null` | Anonymized localStorage profile |

**Response A — Results `RecommendResponse`:**
```json
{
  "status": "ok",
  "intent": {
    "level": "beginner",
    "budget": "low",
    "city_reference": "Roma",
    "country": null,
    "experience_type": "family",
    "snowpark": null,
    "night_ski": null,
    "needs_clarification": false,
    "missing_fields": []
  },
  "resorts": [
    {
      "resort_id": 42,
      "Resort": "Roccaraso",
      "Country": "Italy",
      "HighestPoint": 2140,
      "LowestPoint": 1236,
      "DayPassPriceAdult": 32.0,
      "TotalSlope": 30.0,
      "BeginnerSlope": 15.0,
      "IntermediateSlope": 12.0,
      "DifficultSlope": 3.0,
      "Snowparks": "No",
      "NightSki": "No",
      "TotalLifts": 22,
      "LiftCapacity": 18000,
      "SnowCannons": 200,
      "Latitude": 41.8491,
      "Longitude": 14.0657,
      "distance_km": 163.5,
      "scores": {
        "embedding": 0.7412,
        "mlp": 0.8231,
        "distance": 0.8910,
        "final": 0.8671
      },
      "explanation": "Roccaraso è la scelta più adatta...",
      "is_exploration": false,
      "exploration_reason": null
    }
  ]
}
```

**Response B — Clarification `ClarificationResponse`:**
```json
{
  "status": "clarification_needed",
  "missing_fields": ["level", "budget", "experience_type"],
  "message": "Per consigliarti al meglio ho bisogno di sapere: il tuo livello sciistico (principiante, intermedio o esperto), il budget giornaliero per lo ski pass e il tipo di esperienza che cerchi (famiglia, sportivo, après-ski, panorami). Puoi specificarlo nella prossima risposta?"
}
```

### Pipeline Execution Order

1. **NLU** → `nlu.extract(body.query, strategy=body.strategy)`
2. **Clarification check** → return `ClarificationResponse` if needed
3. **LLM context** → `personalizer.build_llm_context(body.user_profile)` (currently prepared, available for injection)
4. **Scoring** → `scorer.rank(raw_intent, top_k=3)`
5. **Explanation** → `explainer.explain(query, intent, ranked)` — non-blocking (fallback: `None`)
6. **Exploration** → `personalizer.apply_exploration(ranked_explained, all_df, user_profile, intent)`
7. **Serialization** → build `RecommendResponse`

---

## 11. Frontend

**Stack:** React 19.2.5 + TypeScript 6.0.2 + Vite 8.0.10, pure vanilla CSS (no Tailwind)

### State Machine (`useRecommend.ts`)

```typescript
type AppState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'clarification'; message: string; missing: string[] }
  | { phase: 'results'; data: ApiResponse; query: string }
  | { phase: 'error'; message: string };
```

### Components

| Component | Purpose |
|---|---|
| `App.tsx` | Root layout. `hero--idle` (2-column) / `hero--active` (centered). Snowflake animation layer (60 deterministic flakes). Decorative resort cards. |
| `SearchBar.tsx` | Auto-resizing textarea. 5 suggestion chips (populate only, no auto-submit). |
| `ResortCard.tsx` | Full resort card. Rank #1: horizontal 2-column layout. Others: vertical. Exploration badge. Slope bar (beginner/intermediate/difficult). |
| `SkeletonCard.tsx` | Animated loading placeholder, mirrors `ResortCard` layout. |
| `ClarificationBubble.tsx` | Chat-bubble style, contains inline reply input. |

### Design System

| Property | Value |
|---|---|
| Font | Outfit (Google Fonts), weights 300–800 |
| Background | `#F8FAFC` (white-ice) |
| Primary accent | `#1E3A5F` (deep blue) |
| Secondary accent | `#7EC8E3` (sky blue) |
| Text | `#1A1A2E` |
| Cards | `rgba(255,255,255,0.7)` + `backdrop-filter: blur(12px)` (glassmorphism) |
| Border radius | 20-24px (cards), 12px (buttons), 99px (chips/badges) |
| Typography | `font-variant-numeric: tabular-nums` on all numeric data |
| Scroll | `scroll-behavior: smooth` |
| Animations | CSS custom properties (`--dur`, `--delay`, `--drift`, `--rot`, `--enter-delay`, `--float-delay`) |

### Snowflake Animation

60 deterministic flakes: `left`, `size`, `dur`, `delay`, `drift`, `op` computed from index at module load. `delay` is **negative** (flakes already mid-fall on page load). Two-animation architecture: `decoFadeIn` (opacity only) on `.deco-card-wrapper` + `floatCard` (transform only) on `.deco-card`, to prevent CSS animation vs inline style transform conflicts.

---

## 12. Dependencies

### Backend (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.0 | Web framework |
| `uvicorn[standard]` | 0.30.6 | ASGI server |
| `pandas` | 2.2.3 | Dataset management |
| `numpy` | 1.26.4 | Numerical operations |
| `geopy` | 2.4.1 | Nominatim geocoding, Haversine |
| `sentence-transformers` | 3.1.1 | Multilingual embedding model |
| `scikit-learn` | 1.5.2 | MLPRegressor, StandardScaler |
| `torch` | 2.4.1 | sentence-transformers backend |
| `openai` | 1.51.0 | DeepSeek API client (OpenAI-compatible) |
| `python-dotenv` | 1.0.1 | `.env` loading |
| `httpx` | 0.27.2 | Async HTTP client |
| `pydantic` | 2.9.2 | Request/response validation |

### Frontend (`package.json`)

| Package | Version | Purpose |
|---|---|---|
| `react` | ^19.2.5 | UI framework |
| `react-dom` | ^19.2.5 | DOM renderer |
| `vite` | ^8.0.10 | Dev server + bundler |
| `typescript` | ~6.0.2 | Type safety |
| `@vitejs/plugin-react` | ^6.0.1 | React HMR |

No external UI component libraries, no Tailwind — all styling is custom vanilla CSS.

---

## 13. Environment Variables

**File:** `.env` (not committed — copy from `.env.example`)

| Variable | Required | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | API key for DeepSeek-V3 (NLU + Explainer). Format: `sk-...` |
| `GOOGLE_MAPS_API_KEY` | No | Not used by default (geocoding uses Nominatim) |

The backend reads `.env` automatically via `python-dotenv` in both `nlu.py` and `explainer.py`.

---

## 14. Startup Instructions

### Prerequisites

- Python ≥ 3.11
- Node.js ≥ 18
- DeepSeek API key

### Backend Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY=sk-your-key-here

# 3. (First time only) Geocode the dataset — ~7 minutes
python -m backend.data.geocode_resorts
# Output: data/european_ski_resorts_geo.csv

# 4. (First time only) Build scorer artifacts — ~3-5 minutes
python -m backend.data.build_scorer
# Output: data/scorer_artifacts/*.npy, *.pkl, *.json

# 5. Start server
uvicorn backend.main:app --reload --port 8000
```

**Startup output:**
```
[SnowSense] Inizializzazione componenti...
  [1/5] NLU pronto
  [2/5] HybridScorer pronto
  [3/5] Explainer pronto
  [4/5] PersonalizationEngine pronto
  [5/5] Dataset caricato (376 resort)
[SnowSense] Pronto su http://localhost:8000
```

API documentation: http://localhost:8000/docs (Swagger UI)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Verify Installation

```bash
curl -s http://localhost:8000/api/health
# {"status":"ok","scorer_loaded":true}

curl -s -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Piste nere in Austria, budget alto", "strategy": "few_shot"}' \
  | python3 -m json.tool | head -40
```

---

## 15. Build Artifacts

Located in `data/scorer_artifacts/` after running `build_scorer.py`:

### `resort_embeddings.npy`
- Shape: `(376, 384)` — float32
- 384-dimensional sentence-transformer embeddings, L2-normalized
- Row order matches `resort_index.json`
- Generated from Italian text descriptions via `resort_to_text(row)`

### `mlp_model.pkl`
- Pickled `sklearn.neural_network.MLPRegressor`
- Architecture: Linear(22→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)
- Input: 22 features (12 profile + 10 resort, StandardScaler normalized)
- Output: float in [0, 10] representing match quality

### `feature_scaler.pkl`
- Pickled `sklearn.preprocessing.StandardScaler`
- Fit on all 18,800 training samples
- Applied to MLP input at inference time

### `resort_index.json`
- List of 376 integers: DataFrame index values
- Maps `embeddings[i]` → `df.loc[resort_index[i]]`

### `build_summary.json`
- Training metadata: n_resorts, n_profiles, n_training_samples
- MLP metrics: RMSE and MAE on held-out 15% test set (scale 0-10)
- Embedding model name and dimension
- Alpha weights used in scoring

---

*Report generated from source code — SnowSense v1.0.0*
