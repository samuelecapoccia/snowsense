"""
Componente 2 — Arricchimento Geografico
Script one-time: aggiunge latitudine e longitudine a ogni resort tramite Nominatim (OpenStreetMap).

Esecuzione:
    python -m backend.data.geocode_resorts

Output:
    data/european_ski_resorts_geo.csv

Note:
- Nominatim richiede max 1 richiesta/secondo (rispettato con time.sleep)
- Lo script è riprendibile: salta i resort già geocodificati nel file di output
- Per i nomi composti (es. "Soldeu/El Tarter (Grandvalira)") usa il primo segmento come fallback
"""

import re
import time
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

INPUT_CSV = Path(__file__).resolve().parents[2] / "data" / "european_ski_resorts.csv"
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "data" / "european_ski_resorts_geo.csv"

NOMINATIM_USER_AGENT = "snowsense_geocoder_roma3"
REQUEST_DELAY = 1.1        # secondi tra richieste (policy Nominatim: 1 req/s)
MAX_RETRIES = 3
TIMEOUT = 10

# Alcuni nomi di "country" nel dataset non sono riconosciuti da Nominatim — mappa corretta
COUNTRY_MAP = {
    "Southern Russia": "Russia",
    "Siberia": "Russia",
    "United Kingdom": "UK",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pulizia del nome resort per la query
# ---------------------------------------------------------------------------

def _is_indoor_or_dry(name: str) -> bool:
    """Restituisce True per impianti indoor/dry slope — non hanno coordinate utili."""
    lower = name.lower()
    return any(kw in lower for kw in ("indoor ski", "dry slope", "snow arena", "snowworld", "skidome"))


def _name_candidates(name: str) -> List[str]:
    """
    Genera varianti del nome resort da provare in ordine crescente di generalità.
    Es: "Pitztal Glacier-Pitztaler Gletscher-" → ["Pitztal Glacier", "Pitztaler Gletscher", "Pitztal"]
    """
    # Pulizia base
    cleaned = re.sub(r"\(.*?\)", "", name).strip(" -")

    # Segmenti separati da / o zero-width space o – (en-dash)
    segments = [s.strip(" -") for s in re.split(r"[/\u200b–]", cleaned) if s.strip(" -")]

    candidates: List[str] = []

    # 1. Ogni segmento intero
    for seg in segments:
        if seg:
            candidates.append(seg)

    # 2. Prima parola significativa di ogni segmento (per nomi come "Rauriser Hochalmbahnen-Rauris" → "Rauris")
    for seg in segments:
        words = seg.split()
        if len(words) > 1:
            candidates.append(words[-1])   # ultima parola (spesso il nome del paese/luogo)
            candidates.append(words[0])    # prima parola

    # 3. Rimuovi suffissi generici (Glacier, Gletscher, ski resort, ecc.)
    suffixes = re.compile(
        r"\b(glacier|gletscher|ski resort|ski area|skicenter|skiwelt|hochalmbahnen|lifts|bahnen)\b",
        re.IGNORECASE,
    )
    for seg in segments:
        stripped = suffixes.sub("", seg).strip(" -")
        if stripped and stripped not in candidates:
            candidates.append(stripped)

    # Deduplica mantenendo ordine
    seen: Set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_resort(geocoder: Nominatim, resort: str, country: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Tenta di geocodificare un resort provando tutte le varianti del nome.
    Restituisce (lat, lon) oppure (None, None) se fallisce.
    """
    if _is_indoor_or_dry(resort):
        return None, None

    mapped_country = COUNTRY_MAP.get(country, country)
    name_variants = _name_candidates(resort)
    queries = [f"{n}, {mapped_country}" for n in name_variants]

    for query in queries:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                location = geocoder.geocode(query, timeout=TIMEOUT)
                if location:
                    return location.latitude, location.longitude
                break  # risposta valida ma nessun risultato → prossima query
            except GeocoderTimedOut:
                log.warning("Timeout '%s' (tentativo %d/%d)", query, attempt, MAX_RETRIES)
                time.sleep(REQUEST_DELAY * attempt)
            except GeocoderServiceError as exc:
                log.warning("Errore servizio '%s': %s", query, exc)
                time.sleep(REQUEST_DELAY * 2)
                break
        time.sleep(REQUEST_DELAY)

    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Lettura dataset: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, index_col=0)
    total = len(df)

    # Ripresa: se l'output esiste, salta i resort già processati
    already_done: set[str] = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, index_col=0)
        already_done = set(existing.index[existing["Latitude"].notna()])
        log.info("Ripresa: %d resort già geocodificati, skip.", len(already_done))
        df = df.merge(
            existing[["Latitude", "Longitude"]],
            left_index=True, right_index=True, how="left"
        )
    else:
        df["Latitude"] = None
        df["Longitude"] = None

    geocoder = Nominatim(user_agent=NOMINATIM_USER_AGENT)

    to_process = df[df["Latitude"].isna()]
    log.info("Resort da geocodificare: %d / %d", len(to_process), total)

    failed: list[str] = []

    for i, (idx, row) in enumerate(to_process.iterrows(), 1):
        resort_name = str(row["Resort"])
        country = str(row["Country"])

        lat, lon = geocode_resort(geocoder, resort_name, country)

        if lat is not None:
            df.at[idx, "Latitude"] = lat
            df.at[idx, "Longitude"] = lon
            log.info("[%3d/%d] OK  %-50s → (%.4f, %.4f)", i, len(to_process), resort_name[:50], lat, lon)
        else:
            failed.append(resort_name)
            log.warning("[%3d/%d] FAIL %-50s", i, len(to_process), resort_name[:50])

        # Salvataggio incrementale ogni 10 resort
        if i % 10 == 0:
            df.to_csv(OUTPUT_CSV)
            log.info("Checkpoint salvato (%d/%d)", i, len(to_process))

        time.sleep(REQUEST_DELAY)

    # Salvataggio finale
    df.to_csv(OUTPUT_CSV)

    # Riepilogo
    geocoded = df["Latitude"].notna().sum()
    log.info("─" * 60)
    log.info("Completato: %d/%d resort geocodificati (%.1f%%)", geocoded, total, 100 * geocoded / total)
    if failed:
        log.warning("Resort non trovati (%d):", len(failed))
        for name in failed:
            log.warning("  - %s", name)
    log.info("Output salvato in: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
