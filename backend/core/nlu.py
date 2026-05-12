"""
Componente 1 — Conversational NLU
Estrae l'intent utente da una query in linguaggio naturale in tre strategie:
    - zero_shot:  solo la descrizione del task
    - few_shot:   5 esempi query → JSON prima della query reale
    - cot:        Chain-of-Thought: ragionamento passo per passo prima del JSON

Uso:
    from backend.core.nlu import SnowSenseNLU
    nlu = SnowSenseNLU()
    result = nlu.extract("Cerco qualcosa vicino a Roma per bambini", strategy="few_shot")
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Costanti di configurazione
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"   # DeepSeek-V3: ottimo per NLU, molto economico
MAX_TOKENS_SHORT = 512            # zero-shot, few-shot
MAX_TOKENS_COT = 1024             # chain-of-thought (ragionamento + JSON)
TEMPERATURE = 0.0                 # determinismo massimo per confronto riproducibile

# ---------------------------------------------------------------------------
# Schema e istruzioni (condivisi dalle tre strategie)
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """\
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
- missing_fields:  lista solo i campi critici mancanti (level, budget, experience_type), solo se needs_clarification è true\
"""

# ---------------------------------------------------------------------------
# Esempi few-shot (NON sovrapposti con i 50 profili di test)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "query": "Siamo 4 amici che vogliono sciare in Austria la settimana prossima. Siamo tutti bravi, non ci spaventa niente. Non abbiamo problemi di budget.",
        "json": {
            "level": "expert",
            "budget": "high",
            "city_reference": None,
            "country": "Austria",
            "experience_type": "sport",
            "snowpark": None,
            "night_ski": None,
            "needs_clarification": False,
            "missing_fields": [],
        },
    },
    {
        "query": "Prima esperienza sulla neve con mio figlio di 5 anni, abitiamo a Genova. Preferirei non spendere troppo.",
        "json": {
            "level": "beginner",
            "budget": "low",
            "city_reference": "Genova",
            "country": None,
            "experience_type": "family",
            "snowpark": None,
            "night_ski": None,
            "needs_clarification": False,
            "missing_fields": [],
        },
    },
    {
        "query": "Voglio andare a sciare.",
        "json": {
            "level": None,
            "budget": None,
            "city_reference": None,
            "country": None,
            "experience_type": None,
            "snowpark": None,
            "night_ski": None,
            "needs_clarification": True,
            "missing_fields": ["level", "budget", "experience_type"],
        },
    },
    {
        "query": "Sono un freestyle rider, cerco il miglior snowpark europeo. Soldi non sono un problema, voglio anche lo sci notturno.",
        "json": {
            "level": "expert",
            "budget": "high",
            "city_reference": None,
            "country": None,
            "experience_type": "sport",
            "snowpark": True,
            "night_ski": True,
            "needs_clarification": False,
            "missing_fields": [],
        },
    },
    {
        "query": "Weekend in montagna con mio marito, piste non troppo difficili, budget medio. Preferirei la Francia.",
        "json": {
            "level": "intermediate",
            "budget": "medium",
            "city_reference": None,
            "country": "France",
            "experience_type": "panoramic",
            "snowpark": None,
            "night_ski": None,
            "needs_clarification": False,
            "missing_fields": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Chain-of-Thought: istruzioni ragionamento
# ---------------------------------------------------------------------------

COT_REASONING_STEPS = """\
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

Scrivi prima il ragionamento (breve, una riga per punto), poi il JSON finale.\
"""


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class SnowSenseNLU:
    """
    Estrae intent da query sciistiche in italiano via LLM.

    Strategie disponibili: "zero_shot", "few_shot", "cot"
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

    def extract(self, query: str, strategy: str = "few_shot") -> Dict[str, Any]:
        """
        Estrae l'intent da una query usando la strategia specificata.

        Args:
            query:    testo libero dell'utente in italiano
            strategy: "zero_shot" | "few_shot" | "cot"

        Returns:
            dict con i campi dello schema (valori None dove non inferibili)
        """
        builders = {
            "zero_shot": self._zero_shot_messages,
            "few_shot": self._few_shot_messages,
            "cot": self._cot_messages,
        }
        if strategy not in builders:
            raise ValueError(f"Strategia non valida: {strategy!r}. Usa: {list(builders)}")

        messages, max_tok, force_json = builders[strategy](query)
        raw = self._call_llm(messages, max_tokens=max_tok, force_json=force_json)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Costruttori dei prompt per le tre strategie
    # ------------------------------------------------------------------

    def _zero_shot_messages(self, query: str):
        messages = [{"role": "user", "content": f"Query utente:\n{query}"}]
        return messages, MAX_TOKENS_SHORT, True   # True = forza JSON output

    def _few_shot_messages(self, query: str):
        examples_text = "\n\n".join(
            f'Query: "{ex["query"]}"\nJSON: {json.dumps(ex["json"], ensure_ascii=False)}'
            for ex in FEW_SHOT_EXAMPLES
        )
        content = (
            f"Esempi:\n\n{examples_text}\n\n"
            f"Ora estrai l'intent di questa query:\n"
            f"Query: \"{query}\""
        )
        messages = [{"role": "user", "content": content}]
        return messages, MAX_TOKENS_SHORT, True

    def _cot_messages(self, query: str):
        content = f"{COT_REASONING_STEPS}\n\nQuery utente:\n{query}"
        messages = [{"role": "user", "content": content}]
        # CoT: il modello ragiona in testo libero poi scrive il JSON → non forziamo json_object
        return messages, MAX_TOKENS_COT, False

    # ------------------------------------------------------------------
    # Chiamata LLM
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: List[Dict],
        max_tokens: int,
        force_json: bool = True,
    ) -> str:
        """Chiama l'API DeepSeek (OpenAI-compatible) e restituisce il testo grezzo."""
        system_msg = {"role": "system", "content": f"Sei un assistente per stazioni sciistiche europee.\n{SCHEMA_DESCRIPTION}"}
        all_messages = [system_msg] + messages

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            messages=all_messages,
        )
        # response_format json_object garantisce JSON valido per zero-shot e few-shot
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Parsing risposta → dict
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Estrae il JSON dalla risposta del modello.
        Gestisce sia risposte JSON pure (zero-shot/few-shot con prefill)
        sia risposte con ragionamento + JSON (CoT).
        """
        # Cerca il blocco JSON nell'output (primo { ... } bilanciato)
        try:
            json_str = _extract_json_block(text)
            result = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # Fallback: restituisce struttura vuota con flag errore
            result = _empty_intent()
            result["_parse_error"] = True
            result["_raw"] = text[:500]
            return result

        # Normalizzazione dei valori
        result = _normalize_intent(result)
        return result


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Estrae il primo oggetto JSON bilanciato dal testo."""
    start = text.find("{")
    if start == -1:
        raise ValueError("Nessun oggetto JSON trovato")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("JSON non bilanciato")


def _normalize_intent(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizza e valida i valori del JSON estratto."""
    VALID_LEVEL = {"beginner", "intermediate", "expert", None}
    VALID_BUDGET = {"low", "medium", "high", None}
    VALID_EXPERIENCE = {"family", "apres_ski", "sport", "panoramic", None}

    def norm_str(v, valid_set):
        if v is None:
            return None
        v = str(v).lower().strip()
        return v if v in valid_set else None

    def norm_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        if str(v).lower() in ("true", "1", "yes", "sì", "si"):
            return True
        if str(v).lower() in ("false", "0", "no"):
            return False
        return None

    def norm_str_field(v):
        if v is None or str(v).lower() in ("null", "none", ""):
            return None
        return str(v).strip()

    return {
        "level":               norm_str(raw.get("level"), VALID_LEVEL),
        "budget":              norm_str(raw.get("budget"), VALID_BUDGET),
        "city_reference":      norm_str_field(raw.get("city_reference")),
        "country":             norm_str_field(raw.get("country")),
        "experience_type":     norm_str(raw.get("experience_type"), VALID_EXPERIENCE),
        "snowpark":            norm_bool(raw.get("snowpark")),
        "night_ski":           norm_bool(raw.get("night_ski")),
        "needs_clarification": bool(raw.get("needs_clarification", False)),
        "missing_fields":      raw.get("missing_fields", []),
    }


def _empty_intent() -> Dict[str, Any]:
    return {
        "level": None,
        "budget": None,
        "city_reference": None,
        "country": None,
        "experience_type": None,
        "snowpark": None,
        "night_ski": None,
        "needs_clarification": False,
        "missing_fields": [],
    }
