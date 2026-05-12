"""
Componente 6 — Valutazione con Profili Sintetici
Script one-time: genera 50 profili utente diversificati e le loro query in linguaggio naturale.

Esecuzione:
    python -m evaluation.generate_synthetic_profiles

Output:
    data/synthetic_profiles.json

Struttura di ogni profilo:
    - id: identificativo unico (P001..P050)
    - profile: dimensioni strutturate (livello, budget, zona, esperienza, città)
    - query_it: query realistica in italiano scritta come un utente reale
    - ground_truth_intent: quello che il NLU *deve* estrarre dalla query
      (serve come ground truth per il confronto zero-shot / few-shot / CoT)

Note sui campi ground_truth_intent:
    - Se la query non menziona un campo, il valore è null
    - I campi booleani (snowpark, night_ski) sono null se non menzionati
    - Questo rispecchia il comportamento atteso del NLU: non inferire ciò che non è detto
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_profiles.json"

# ---------------------------------------------------------------------------
# Dati strutturati dei 50 profili
# ---------------------------------------------------------------------------
# Dimensioni:
#   level:           beginner | intermediate | expert
#   budget:          low (<35€) | medium (35-55€) | high (>55€)
#   zona:            italy_north | italy_center_south | france | austria | no_preference
#   experience_type: family | apres_ski | sport | panoramic
#   city_reference:  città di partenza (stringa) oppure null
#   snowpark:        true | false | null
#   night_ski:       true | false | null

PROFILES_RAW = [

    # ── BEGINNER + FAMILY (8 profili) ────────────────────────────────────────

    {
        "id": "P001",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "italy_center_south", "experience_type": "family",
            "city_reference": "Roma", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Ciao, siamo una famiglia con due bambini di 6 e 9 anni, partiamo da Roma. "
            "Non abbiamo mai sciato, cerchiamo qualcosa di economico e non troppo lontano. "
            "Dove ci consigliate?"
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": "Roma", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P002",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "italy_center_south", "experience_type": "family",
            "city_reference": "Napoli", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Vorremmo portare i bambini sulla neve per la prima volta, abitiamo a Napoli. "
            "Budget limitato, preferiamo piste facili e un posto tranquillo."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": "Napoli", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P003",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_north", "experience_type": "family",
            "city_reference": "Milano", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Siamo una famiglia milanese, bambini piccoli, vogliamo un weekend sulla neve. "
            "Prima volta sugli sci per i piccoli, budget medio. "
            "Qualcosa di raggiungibile da Milano senza troppo viaggio."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Milano", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P004",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_north", "experience_type": "family",
            "city_reference": "Torino", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerco una stazione sciistica per bambini, siamo di Torino. "
            "I ragazzi non hanno mai sciato, vogliamo una settimana bianca tranquilla. "
            "Spesa nella media."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Torino", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P005",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "no_preference", "experience_type": "family",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Dove posso portare la famiglia a sciare spendendo poco? "
            "Siamo principianti assoluti, cerchiamo piste per bambini e prezzi accessibili."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": None, "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P006",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "austria", "experience_type": "family",
            "city_reference": "Vienna", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Siamo a Vienna con due bambini, stiamo cercando una stazione sciistica "
            "adatta per chi inizia da zero. Budget medio, vogliamo stare in Austria."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Vienna", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P007",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_north", "experience_type": "family",
            "city_reference": "Bologna", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Famiglia con bambini da Bologna cerca stazione sci con piste verdi e blu. "
            "Siamo alle prime armi, prezzo nella media. "
            "Possibilmente non troppo lontano dall'Emilia."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Bologna", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P008",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "italy_center_south", "experience_type": "family",
            "city_reference": "Firenze", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Partiamo da Firenze, prima volta sulla neve con i bambini. "
            "Cerchiamo qualcosa di economico, piste adatte ai principianti, "
            "atmosfera familiare."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": "Firenze", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── BEGINNER + PANORAMIC (5 profili) ─────────────────────────────────────

    {
        "id": "P009",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_north", "experience_type": "panoramic",
            "city_reference": "Venezia", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sono alle prime armi con lo sci ma cerco soprattutto paesaggi mozzafiato. "
            "Parto da Venezia, budget medio. Mi interessa più la bellezza del posto "
            "che le piste difficili."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Venezia", "experience_type": "panoramic",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P010",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_center_south", "experience_type": "panoramic",
            "city_reference": "Roma", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Non so sciare bene ma voglio godermi la montagna e i panorami. "
            "Sono a Roma, budget medio, qualcosa di scenico non troppo lontano."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Roma", "experience_type": "panoramic",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P011",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "italy_north", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerco una stazione sciistica in Italia del nord con bei panorami, "
            "sono principiante e ho un budget limitato."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": None, "experience_type": "panoramic",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P012",
        "profile": {
            "level": "beginner", "budget": "low",
            "zona": "italy_center_south", "experience_type": "panoramic",
            "city_reference": "Perugia", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Primo anno sugli sci, parto da Perugia. Cerco qualcosa di economico "
            "ma con un bel paesaggio montano. Non mi interessa la difficoltà delle piste."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "low",
            "city_reference": "Perugia", "experience_type": "panoramic",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P013",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "france", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Voglio sciare in Francia, sono alle prime armi. "
            "Mi interessa una stazione con paesaggi alpini belli, budget nella media. "
            "Non cerco piste nere."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": None, "experience_type": "panoramic",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── BEGINNER + APRES-SKI / SPORT (3 profili) ─────────────────────────────

    {
        "id": "P014",
        "profile": {
            "level": "beginner", "budget": "high",
            "zona": "austria", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": True,
        },
        "query_it": (
            "Voglio imparare a sciare ma per me è importante anche la vita notturna. "
            "Cerco una stazione in Austria con buon après-ski, non mi importa il budget. "
            "Ci sarà anche qualcosa di sera?"
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "Austria",
            "snowpark": None, "night_ski": True,
        },
    },
    {
        "id": "P015",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": "Milano", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sono un principiante motivato, voglio migliorare velocemente. "
            "Cerco una stazione vicino a Milano con buone scuole sci e molte piste blu. "
            "Budget medio."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": "Milano", "experience_type": "sport",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P016",
        "profile": {
            "level": "beginner", "budget": "medium",
            "zona": "france", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Prima settimana bianca in Francia con gli amici, siamo tutti alle prime armi. "
            "Vogliamo sciare di giorno e divertirci la sera. Budget medio."
        ),
        "ground_truth_intent": {
            "level": "beginner", "budget": "medium",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── INTERMEDIATE + FAMILY (5 profili) ────────────────────────────────────

    {
        "id": "P017",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_center_south", "experience_type": "family",
            "city_reference": "Roma", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciamo da qualche anno, cerchiamo una stazione per tutta la famiglia "
            "vicino a Roma. Piste rosse per noi adulti e qualcosa di più facile per i bambini. "
            "Budget nella media."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Roma", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P018",
        "profile": {
            "level": "intermediate", "budget": "low",
            "zona": "italy_center_south", "experience_type": "family",
            "city_reference": "Napoli", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerchiamo qualcosa di economico da Napoli, siamo intermedi ma con bambini piccoli. "
            "Non vogliamo spendere troppo, basta che ci siano piste variegate."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "low",
            "city_reference": "Napoli", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P019",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "family",
            "city_reference": "Milano", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Famiglia con esperienza media da Milano, cerchiamo una stazione con "
            "piste per tutti i livelli. Non troppo lontano, settimana di vacanza invernale."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Milano", "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P020",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "no_preference", "experience_type": "family",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerchiamo la migliore stazione sciistica per famiglie in Europa, "
            "budget non è un problema. Vogliamo qualità, servizi per bambini, "
            "e piste per tutti i livelli."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "family",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P021",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "family",
            "city_reference": "Bologna", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Da Bologna, sci in famiglia, livello intermedio. "
            "Bambini adolescenti che sciamo insieme. Budget medio, preferibilmente in Italia."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Bologna", "experience_type": "family",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── INTERMEDIATE + SPORT (8 profili) ─────────────────────────────────────

    {
        "id": "P022",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": "Milano", "snowpark": True, "night_ski": False,
        },
        "query_it": (
            "Sciatore intermedio da Milano, cerco una stazione con snowpark e "
            "piste rosse tecniche. Budget elevato, voglio qualità e km di piste. "
            "Interessato anche allo snowpark."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": "Milano", "experience_type": "sport",
            "snowpark": True, "night_ski": None,
        },
    },
    {
        "id": "P023",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": "Torino", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciatore di livello medio da Torino, voglio una settimana sulle piste rosse. "
            "Budget nella media, cerco tanti km di piste e buon innevamento."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Torino", "experience_type": "sport",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P024",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "austria", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Voglio una settimana sciistica seria in Austria, sono intermedio e "
            "voglio migliorare su piste rosse lunghe. Budget alto, preferisco "
            "stazioni con molti km di piste."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "Austria",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P025",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "france", "experience_type": "sport",
            "city_reference": None, "snowpark": True, "night_ski": False,
        },
        "query_it": (
            "Cerco una grande stazione in Francia, livello intermedio, "
            "snowpark per fare qualche trick. Budget non è un problema, "
            "voglio il meglio della Valle Bianca o simili."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "France",
            "snowpark": True, "night_ski": None,
        },
    },
    {
        "id": "P026",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_center_south", "experience_type": "sport",
            "city_reference": "Roma", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciatore discreto da Roma, voglio piste rosse senza stare troppo lontano. "
            "Budget medio, cerco una stazione italiana con buona skiarea."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Roma", "experience_type": "sport",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P027",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Livello intermedio, cerco una stazione italiana del nord con almeno "
            "50km di piste rosse e buon innevamento artificiale. Budget medio."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": None, "experience_type": "sport",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P028",
        "profile": {
            "level": "intermediate", "budget": "low",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerco una stazione in Italia del nord economica ma con buone piste "
            "per un livello intermedio. Vorrei sciare su rosse e qualche nera, "
            "ma senza spendere troppo."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "low",
            "city_reference": None, "experience_type": "sport",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P029",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "no_preference", "experience_type": "sport",
            "city_reference": None, "snowpark": True, "night_ski": True,
        },
        "query_it": (
            "Sciatore intermedio in cerca della stazione europea più completa: "
            "voglio snowpark, sci notturno e tanti km di piste. Budget illimitato."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "snowpark": True, "night_ski": True,
        },
    },

    # ── INTERMEDIATE + APRES-SKI (5 profili) ─────────────────────────────────

    {
        "id": "P030",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "austria", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": True,
        },
        "query_it": (
            "Gruppo di amici, sciamo bene ma vogliamo soprattutto divertirci la sera. "
            "Austria è la meta perfetta per l'après-ski, budget alto, vogliamo "
            "la stazione più festosa possibile."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "Austria",
            "snowpark": None, "night_ski": True,
        },
    },
    {
        "id": "P031",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "france", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerco una stazione francese con bella vita notturna e buone piste rosse. "
            "Budget elevato, voglio un posto alla moda dove sciare e poi godersi la serata."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P032",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "italy_north", "experience_type": "apres_ski",
            "city_reference": "Milano", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Da Milano cerco una stazione con ottimo après-ski, sciamo a livello "
            "intermedio. Budget alto, vogliamo lusso e divertimento serale."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": "Milano", "experience_type": "apres_ski",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P033",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "no_preference", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Voglio una settimana di sci e divertimento, livello intermedio, "
            "budget medio. Non ho preferenze geografiche, cerco un buon mix "
            "di piste e vita sociale."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": None, "experience_type": "apres_ski",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P034",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "austria", "experience_type": "apres_ski",
            "city_reference": "Vienna", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Siamo a Vienna, sciatori medi, vogliamo un weekend con buon après-ski "
            "in Austria. Budget nella media."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Vienna", "experience_type": "apres_ski",
            "country": "Austria",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── INTERMEDIATE + PANORAMIC (5 profili) ─────────────────────────────────

    {
        "id": "P035",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciatore intermedio in cerca di paesaggi spettacolari sulle Alpi italiane. "
            "Budget medio, cerco una stazione con viste mozzafiato e piste rosse."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": None, "experience_type": "panoramic",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P036",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "italy_north", "experience_type": "panoramic",
            "city_reference": "Torino", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Da Torino cerco una stazione con scenari alpini straordinari, "
            "piste rosse per un livello intermedio, budget nella norma."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": "Torino", "experience_type": "panoramic",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P037",
        "profile": {
            "level": "intermediate", "budget": "high",
            "zona": "no_preference", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Voglio la stazione sciistica più bella d'Europa dal punto di vista paesaggistico. "
            "Sono un buon sciatore, il budget non è un limite. Voglio panorami indimenticabili."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "high",
            "city_reference": None, "experience_type": "panoramic",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P038",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "france", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Cerco una stazione francese con panorami alpini da cartolina, "
            "piste rosse, budget medio."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": None, "experience_type": "panoramic",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P039",
        "profile": {
            "level": "intermediate", "budget": "medium",
            "zona": "austria", "experience_type": "panoramic",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Stazione austriaca con viste sulle Alpi, sciatore intermedio, "
            "budget nella media. Voglio che ogni discesa sia una cartolina."
        ),
        "ground_truth_intent": {
            "level": "intermediate", "budget": "medium",
            "city_reference": None, "experience_type": "panoramic",
            "country": "Austria",
            "snowpark": None, "night_ski": None,
        },
    },

    # ── EXPERT + SPORT (8 profili) ────────────────────────────────────────────

    {
        "id": "P040",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "austria", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sci agonistico, cerco la stazione con le piste nere più impegnative in Austria. "
            "Budget illimitato, voglio dislivello massimo e snowcannons per garantire "
            "innevamento."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "Austria",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P041",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "france", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Voglio fare fuoripista in Francia, sono un esperto. "
            "Cerco alta quota, tanta neve, piste nere serie. Budget alto."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P042",
        "profile": {
            "level": "expert", "budget": "medium",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciatore esperto cerca piste nere impegnative in Italia del nord. "
            "Non ho budget illimitato ma voglio piste di qualità e buon dislivello."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "medium",
            "city_reference": None, "experience_type": "sport",
            "country": "Italy",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P043",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "austria", "experience_type": "sport",
            "city_reference": "Vienna", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Da Vienna, sciatore di alto livello, cerco la stazione con il "
            "maggior dislivello in Austria. Budget non è un problema."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": "Vienna", "experience_type": "sport",
            "country": "Austria",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P044",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "no_preference", "experience_type": "sport",
            "city_reference": None, "snowpark": True, "night_ski": False,
        },
        "query_it": (
            "Esperto di freestyle, cerco la stazione europea con il miglior snowpark "
            "e piste tecniche. Budget alto, non importa dove si trova."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "snowpark": True, "night_ski": None,
        },
    },
    {
        "id": "P045",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "austria", "experience_type": "sport",
            "city_reference": None, "snowpark": False, "night_ski": True,
        },
        "query_it": (
            "Cercate stazioni in Austria con sci notturno e piste nere. "
            "Sciatore esperto, budget alto, voglio sciare anche dopo il tramonto."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "Austria",
            "snowpark": None, "night_ski": True,
        },
    },
    {
        "id": "P046",
        "profile": {
            "level": "expert", "budget": "medium",
            "zona": "italy_north", "experience_type": "sport",
            "city_reference": "Torino", "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Sciatore esperto da Torino, cerco piste nere con buon dislivello "
            "sulle Alpi italiane. Budget medio, voglio sfida tecnica."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "medium",
            "city_reference": "Torino", "experience_type": "sport",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P047",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "france", "experience_type": "sport",
            "city_reference": None, "snowpark": True, "night_ski": False,
        },
        "query_it": (
            "Esperto che ama lo sci freestyle, voglio la stazione francese con "
            "snowpark di livello mondiale e piste nere. Budget alto."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "sport",
            "country": "France",
            "snowpark": True, "night_ski": None,
        },
    },

    # ── EXPERT + APRES-SKI (3 profili) ───────────────────────────────────────

    {
        "id": "P048",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "austria", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": True,
        },
        "query_it": (
            "Sciatore esperto, voglio una stazione austriaca dove sciare su nere "
            "di giorno e fare festa la notte. Budget alto, sci notturno indispensabile."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "Austria",
            "snowpark": None, "night_ski": True,
        },
    },
    {
        "id": "P049",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "france", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": False,
        },
        "query_it": (
            "Esperto, cerco la stazione francese più glam: piste difficili di giorno, "
            "champagne e musica di sera. Budget illimitato."
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "country": "France",
            "snowpark": None, "night_ski": None,
        },
    },
    {
        "id": "P050",
        "profile": {
            "level": "expert", "budget": "high",
            "zona": "no_preference", "experience_type": "apres_ski",
            "city_reference": None, "snowpark": False, "night_ski": True,
        },
        "query_it": (
            "Cerco la stazione europea definitiva: piste nere lunghe, "
            "après-ski leggendario, sci notturno. Budget non è un limite. "
            "Dove andate se volete il massimo?"
        ),
        "ground_truth_intent": {
            "level": "expert", "budget": "high",
            "city_reference": None, "experience_type": "apres_ski",
            "snowpark": None, "night_ski": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Validazione
# ---------------------------------------------------------------------------

VALID_LEVELS = {"beginner", "intermediate", "expert"}
VALID_BUDGETS = {"low", "medium", "high"}
VALID_ZONE = {"italy_north", "italy_center_south", "france", "austria", "no_preference"}
VALID_EXPERIENCE = {"family", "apres_ski", "sport", "panoramic"}
BUDGET_RANGE = {"low": "<35€", "medium": "35–55€", "high": ">55€"}


def validate_profile(p: dict) -> None:
    pid = p["id"]
    pr = p["profile"]
    assert pr["level"] in VALID_LEVELS, f"{pid}: livello non valido"
    assert pr["budget"] in VALID_BUDGETS, f"{pid}: budget non valido"
    assert pr["zona"] in VALID_ZONE, f"{pid}: zona non valida"
    assert pr["experience_type"] in VALID_EXPERIENCE, f"{pid}: experience non valida"
    assert isinstance(p["query_it"], str) and len(p["query_it"]) > 20, f"{pid}: query troppo corta"
    assert isinstance(p["ground_truth_intent"], dict), f"{pid}: ground_truth_intent mancante"


# ---------------------------------------------------------------------------
# Arricchimento (aggiunge campi derivati utili)
# ---------------------------------------------------------------------------

def enrich(p: dict) -> dict:
    budget_key = p["profile"]["budget"]
    p["profile"]["budget_range_eur"] = BUDGET_RANGE[budget_key]
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Validazione {len(PROFILES_RAW)} profili...")
    for p in PROFILES_RAW:
        validate_profile(p)
    print("  OK — tutti i profili sono validi.")

    enriched = [enrich(p) for p in PROFILES_RAW]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Salvati {len(enriched)} profili in: {OUTPUT_PATH}")

    # Statistiche distribuzione
    from collections import Counter
    levels = Counter(p["profile"]["level"] for p in enriched)
    budgets = Counter(p["profile"]["budget"] for p in enriched)
    zones = Counter(p["profile"]["zona"] for p in enriched)
    experiences = Counter(p["profile"]["experience_type"] for p in enriched)
    with_city = sum(1 for p in enriched if p["profile"]["city_reference"])

    print("\nDistribuzione profili:")
    print(f"  Livello:     {dict(levels)}")
    print(f"  Budget:      {dict(budgets)}")
    print(f"  Zona:        {dict(zones)}")
    print(f"  Esperienza:  {dict(experiences)}")
    print(f"  Con città:   {with_city}/{len(enriched)}")


if __name__ == "__main__":
    main()
