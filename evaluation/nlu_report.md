# SnowSense — Valutazione NLU: Confronto Strategie di Prompting

**Profili testati:** 50 / 50
**Strategie:** Zero-shot, Few-shot (5 esempi), Chain-of-Thought

---

## Accuratezza per Campo

| Strategy | Level | Budget | City ref. | Country | Experience | Snowpark | Night ski | Overall | Parse err |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 92.0% | 98.0% | 100.0% | 100.0% | 98.0% | 100.0% | 83.3% | **95.9%** | 0 |
| Few-shot | 94.0% | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% | 83.3% | **96.5%** | 0 |
| Chain-of-Thought | 90.0% | 96.0% | 100.0% | 100.0% | 94.0% | 100.0% | 66.7% | **92.4%** | 0 |

*Overall = media delle accuracy sui campi con ground truth non-null.*

---

## False Positive Rate (Hallucination)

Percentuale di volte in cui il modello predice un valore per un campo
che l'utente **non ha menzionato** (ground truth = null).

| Strategy | Level | Budget | City Reference | Country | Experience Type | Snowpark | Night Ski |
|---|---|---|---|---|---|---|---|
| Zero-shot | — | — | 0.0% | 11.1% | — | 0.0% | 2.3% |
| Few-shot | — | — | 0.0% | 11.1% | — | 0.0% | 0.0% |
| Chain-of-Thought | — | — | 0.0% | 11.1% | — | 0.0% | 2.3% |

---

## Note Metodologiche

- **Exact match** per campi categorici (level, budget, experience\_type)
- **Case-insensitive substring match** per campi stringa (city\_reference, country)
- **Exact match** per booleani (snowpark, night\_ski), valutato solo dove ground truth ≠ null
- I campi con ground truth null vengono esclusi dall'accuracy ma conteggiati nei false positive
