"""
Componente 1 — Script di Valutazione NLU
Testa le tre strategie di prompting (zero-shot, few-shot, chain-of-thought)
sui 50 profili sintetici e produce una tabella comparativa di accuratezza.

Esecuzione:
    python -m evaluation.evaluate_nlu                  # esegue tutto (150 chiamate API)
    python -m evaluation.evaluate_nlu --limit 10       # testa solo i primi 10 profili
    python -m evaluation.evaluate_nlu --recompute      # ricarica risultati salvati, ricalcola metriche

Output:
    evaluation/nlu_raw_results.json     risultati grezzi (predizioni + ground truth)
    evaluation/nlu_report.md            tabella comparativa in Markdown
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "data" / "synthetic_profiles.json"
RAW_RESULTS_PATH = ROOT / "evaluation" / "nlu_raw_results.json"
REPORT_PATH = ROOT / "evaluation" / "nlu_report.md"

sys.path.insert(0, str(ROOT))
from backend.core.nlu import SnowSenseNLU

# ---------------------------------------------------------------------------
# Campi valutati e loro tipologia
# ---------------------------------------------------------------------------

# (campo_gt, campo_predetto, tipo)
# tipo: "categorical" | "string_ci" | "boolean"
EVAL_FIELDS: List[Tuple[str, str, str]] = [
    ("level",           "level",           "categorical"),
    ("budget",          "budget",          "categorical"),
    ("city_reference",  "city_reference",  "string_ci"),
    ("country",         "country",         "string_ci"),
    ("experience_type", "experience_type", "categorical"),
    ("snowpark",        "snowpark",        "boolean"),
    ("night_ski",       "night_ski",       "boolean"),
]

STRATEGIES = ["zero_shot", "few_shot", "cot"]
STRATEGY_LABELS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "Chain-of-Thought"}
API_DELAY = 0.5   # secondi tra chiamate (evita rate limit)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def field_match(gt_value: Any, pred_value: Any, field_type: str) -> Optional[bool]:
    """
    Confronta un valore predetto con il ground truth.
    Restituisce:
        True   → corretto
        False  → sbagliato
        None   → ground truth è null (campo non menzionato nella query) → skip
    """
    if gt_value is None:
        return None  # campo non valutato per questo profilo

    if pred_value is None:
        return False  # il modello non ha estratto un campo presente nel GT

    if field_type == "categorical":
        return str(gt_value).lower() == str(pred_value).lower()

    if field_type == "string_ci":
        # Match case-insensitive; accetta anche match parziale (es. "Roma" in "Roma, Italy")
        gt_norm = str(gt_value).lower().strip()
        pred_norm = str(pred_value).lower().strip()
        return gt_norm in pred_norm or pred_norm in gt_norm

    if field_type == "boolean":
        return bool(gt_value) == bool(pred_value)

    return False


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    results: lista di entry con struttura
        { "id", "strategy", "ground_truth", "prediction", "matches" }
    Restituisce metriche per strategia e per campo.
    """
    metrics: Dict[str, Any] = {}

    for strategy in STRATEGIES:
        strat_results = [r for r in results if r["strategy"] == strategy]

        field_stats: Dict[str, Dict] = {}
        for _, field, _ in EVAL_FIELDS:
            correct = sum(
                1 for r in strat_results
                if r["matches"].get(field) is True
            )
            total = sum(
                1 for r in strat_results
                if r["matches"].get(field) is not None
            )
            false_pos = sum(
                1 for r in strat_results
                if r["matches"].get(field) is None
                and r["prediction"].get(field) is not None
            )
            null_gt_total = sum(
                1 for r in strat_results
                if r["matches"].get(field) is None
            )
            field_stats[field] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total > 0 else None,
                "false_positives": false_pos,
                "null_gt_total": null_gt_total,
                "fp_rate": false_pos / null_gt_total if null_gt_total > 0 else None,
            }

        # Overall accuracy = media delle accuracy per campo (solo campi con ≥1 esempio valutato)
        field_accs = [
            v["accuracy"] for v in field_stats.values()
            if v["accuracy"] is not None
        ]
        overall = sum(field_accs) / len(field_accs) if field_accs else None

        # Parse error rate
        parse_errors = sum(1 for r in strat_results if r["prediction"].get("_parse_error"))

        metrics[strategy] = {
            "fields": field_stats,
            "overall_accuracy": overall,
            "parse_errors": parse_errors,
            "total_profiles": len(strat_results),
        }

    return metrics


# ---------------------------------------------------------------------------
# Rendering tabella
# ---------------------------------------------------------------------------

def render_markdown_table(metrics: Dict[str, Any]) -> str:
    """Genera la tabella comparativa in Markdown."""
    field_names = [f for _, f, _ in EVAL_FIELDS]
    col_labels = {
        "level": "Level",
        "budget": "Budget",
        "city_reference": "City ref.",
        "country": "Country",
        "experience_type": "Experience",
        "snowpark": "Snowpark",
        "night_ski": "Night ski",
    }

    header = "| Strategy | " + " | ".join(col_labels[f] for f in field_names) + " | Overall | Parse err |"
    sep    = "|---|" + "---|" * (len(field_names) + 2)

    rows = []
    for strategy in STRATEGIES:
        m = metrics[strategy]
        cells = []
        for f in field_names:
            acc = m["fields"][f]["accuracy"]
            cells.append(f"{acc*100:.1f}%" if acc is not None else "—")
        overall = m["overall_accuracy"]
        overall_str = f"**{overall*100:.1f}%**" if overall is not None else "—"
        parse_str = str(m["parse_errors"]) if m["parse_errors"] else "0"
        label = STRATEGY_LABELS[strategy]
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {overall_str} | {parse_str} |")

    return "\n".join([header, sep] + rows)


def render_fp_table(metrics: Dict[str, Any]) -> str:
    """Tabella false-positive rate (hallucination): modello inventa valori non richiesti."""
    field_names = [f for _, f, _ in EVAL_FIELDS]
    header = "| Strategy | " + " | ".join(f.replace("_", " ").title() for f in field_names) + " |"
    sep    = "|---|" + "---|" * len(field_names)

    rows = []
    for strategy in STRATEGIES:
        m = metrics[strategy]
        cells = []
        for f in field_names:
            rate = m["fields"][f]["fp_rate"]
            cells.append(f"{rate*100:.1f}%" if rate is not None else "—")
        rows.append(f"| {STRATEGY_LABELS[strategy]} | " + " | ".join(cells) + " |")

    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_evaluation(profiles: List[Dict], nlu: SnowSenseNLU, limit: Optional[int]) -> List[Dict]:
    """Esegue le chiamate API per tutte le strategie su tutti i profili."""
    subset = profiles[:limit] if limit else profiles
    total_calls = len(subset) * len(STRATEGIES)
    results: List[Dict] = []
    call_n = 0

    for profile in subset:
        pid = profile["id"]
        query = profile["query_it"]
        gt = profile["ground_truth_intent"]

        for strategy in STRATEGIES:
            call_n += 1
            print(f"  [{call_n:3d}/{total_calls}] {pid} / {strategy}...", end=" ", flush=True)

            try:
                prediction = nlu.extract(query, strategy=strategy)
                had_error = prediction.get("_parse_error", False)
            except Exception as exc:
                print(f"ERRORE: {exc}")
                prediction = {"_parse_error": True, "_exception": str(exc)}
                had_error = True

            # Calcola i match campo per campo
            matches: Dict[str, Optional[bool]] = {}
            for gt_field, pred_field, field_type in EVAL_FIELDS:
                gt_val = gt.get(gt_field)
                pred_val = prediction.get(pred_field)
                matches[pred_field] = field_match(gt_val, pred_val, field_type)

            results.append({
                "id": pid,
                "strategy": strategy,
                "query": query,
                "ground_truth": gt,
                "prediction": prediction,
                "matches": matches,
            })

            status = "PARSE_ERR" if had_error else "OK"
            correct = sum(1 for v in matches.values() if v is True)
            evaluated = sum(1 for v in matches.values() if v is not None)
            print(f"{status}  {correct}/{evaluated} campi corretti")

            time.sleep(API_DELAY)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Valutazione NLU — SnowSense")
    parser.add_argument("--limit", type=int, default=None, help="Testa solo i primi N profili")
    parser.add_argument("--recompute", action="store_true", help="Ricarica risultati salvati e ricalcola metriche")
    parser.add_argument("--model", type=str, default=None, help="Modello LLM da usare")
    args = parser.parse_args()

    # Carica profili
    print(f"Caricamento profili: {PROFILES_PATH}")
    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"  {len(profiles)} profili caricati.")

    if args.recompute and RAW_RESULTS_PATH.exists():
        print(f"Ricarico risultati da: {RAW_RESULTS_PATH}")
        with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
            results = json.load(f)
        print(f"  {len(results)} predizioni caricate.")
    else:
        # Inizializza NLU
        kwargs = {"model": args.model} if args.model else {}
        nlu = SnowSenseNLU(**kwargs)
        n_profiles = args.limit or len(profiles)
        n_calls = n_profiles * len(STRATEGIES)
        print(f"\nAvvio valutazione: {n_profiles} profili × {len(STRATEGIES)} strategie = {n_calls} chiamate API")
        print(f"Modello: {nlu.model}\n")

        results = run_evaluation(profiles, nlu, limit=args.limit)

        # Salva risultati grezzi
        RAW_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nRisultati grezzi salvati: {RAW_RESULTS_PATH}")

    # Calcola metriche
    metrics = compute_metrics(results)

    # Stampa riepilogo a console
    print("\n" + "=" * 70)
    print("RISULTATI CONFRONTO STRATEGIE NLU — SnowSense")
    print("=" * 70)
    print("\nAccuratezza per campo e strategia:")
    print()

    field_names = [f for _, f, _ in EVAL_FIELDS]
    for strategy in STRATEGIES:
        m = metrics[strategy]
        print(f"  {STRATEGY_LABELS[strategy]}:")
        for f in field_names:
            acc = m["fields"][f]["accuracy"]
            total = m["fields"][f]["total"]
            fp = m["fields"][f]["false_positives"]
            acc_str = f"{acc*100:5.1f}%" if acc is not None else "    —"
            print(f"    {f:<20} acc={acc_str}  (n={total}, false_pos={fp})")
        overall = m["overall_accuracy"]
        print(f"    {'OVERALL':<20} acc={overall*100:5.1f}%" if overall else "")
        print(f"    Parse errors: {m['parse_errors']}")
        print()

    # Genera report Markdown
    acc_table = render_markdown_table(metrics)
    fp_table = render_fp_table(metrics)

    n_profiles_tested = len(set(r["id"] for r in results))
    model_used = results[0]["prediction"].get("_model", "—") if results else "—"

    report_md = f"""\
# SnowSense — Valutazione NLU: Confronto Strategie di Prompting

**Profili testati:** {n_profiles_tested} / 50
**Strategie:** Zero-shot, Few-shot (5 esempi), Chain-of-Thought

---

## Accuratezza per Campo

{acc_table}

*Overall = media delle accuracy sui campi con ground truth non-null.*

---

## False Positive Rate (Hallucination)

Percentuale di volte in cui il modello predice un valore per un campo
che l'utente **non ha menzionato** (ground truth = null).

{fp_table}

---

## Note Metodologiche

- **Exact match** per campi categorici (level, budget, experience\\_type)
- **Case-insensitive substring match** per campi stringa (city\\_reference, country)
- **Exact match** per booleani (snowpark, night\\_ski), valutato solo dove ground truth ≠ null
- I campi con ground truth null vengono esclusi dall'accuracy ma conteggiati nei false positive
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report salvato: {REPORT_PATH}")
    print()
    print(acc_table)


if __name__ == "__main__":
    main()
