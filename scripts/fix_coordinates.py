"""
Fix coordinate dei resort sciistici usando Google Maps Geocoding API.

Ri-geocodifica TUTTI i 376 resort usando Google Geocoding API.
Le coordinate restituite da Google vengono accettate senza validazione aggiuntiva.

Esecuzione:
    .venv/bin/python scripts/fix_coordinates.py

Prerequisito:
    GOOGLE_MAPS_API_KEY=... nel file .env

Output:
    data/european_ski_resorts_geo.csv  — CSV aggiornato (backup in .bak)
    data/coordinate_fixes_report.csv  — report before/after per ogni resort
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import googlemaps
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[1]
GEO_CSV  = ROOT / "data" / "european_ski_resorts_geo.csv"
REPORT   = ROOT / "data" / "coordinate_fixes_report.csv"
BAK      = Path(str(GEO_CSV) + ".bak")

REQUEST_DELAY = 0.05  # Google Geocoding: fino a 50 req/s sul piano gratuito

COUNTRY_MAP = {
    "Southern Russia": "Russia",
    "Siberia":         "Russia",
    "United Kingdom":  "UK",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pulizia nome resort
# ---------------------------------------------------------------------------

def _is_indoor_or_dry(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in (
        "indoor ski", "dry slope", "snow arena", "snowworld",
        "skidome", "snow dome", "indoorski",
    ))


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\(.*?\)", "", name)       # rimuove (Via Lattea) ecc.
    cleaned = cleaned.replace("\u200b", " ")      # zero-width space → spazio
    cleaned = re.sub(r"[-–/]+\s*$", "", cleaned) # trattini finali
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _first_segment(name: str) -> str:
    cleaned = _clean_name(name)
    seg = re.split(r"[/\u200b–-]", cleaned)[0].strip()
    return seg if seg else cleaned


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_resort(
    gmaps: googlemaps.Client,
    resort: str,
    country: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Geocodifica un resort con Google Maps API. Prova più varianti del nome."""
    if _is_indoor_or_dry(resort):
        return None, None

    mapped_country = COUNTRY_MAP.get(country, country)
    clean   = _clean_name(resort)
    segment = _first_segment(resort)

    queries = [
        f"{clean} ski resort, {mapped_country}",
        f"{clean}, {mapped_country}",
    ]
    if segment != clean:
        queries.append(f"{segment} ski resort, {mapped_country}")
        queries.append(f"{segment}, {mapped_country}")

    for query in queries:
        try:
            results = gmaps.geocode(query)
            if results:
                loc = results[0]["geometry"]["location"]
                return float(loc["lat"]), float(loc["lng"])
        except Exception as exc:
            log.warning("Errore query '%s': %s", query, exc)
        time.sleep(REQUEST_DELAY)

    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY non configurata. Inseriscila nel file .env"
        )

    gmaps = googlemaps.Client(key=api_key)

    # Backup (solo la prima volta)
    if not BAK.exists():
        import shutil
        shutil.copy(GEO_CSV, BAK)
        log.info("Backup creato: %s", BAK)

    df = pd.read_csv(GEO_CSV, index_col=0)
    log.info("Dataset: %d resort", len(df))

    report_rows = []
    fixed = unchanged = indoor_skip = missing_after = 0

    for idx, row in df.iterrows():
        resort  = str(row["Resort"])
        country = str(row["Country"])
        old_lat = row.get("Latitude")
        old_lon = row.get("Longitude")

        if _is_indoor_or_dry(resort):
            indoor_skip += 1
            report_rows.append({
                "Resort": resort, "Country": country, "Status": "INDOOR_SKIP",
                "Old_Lat": old_lat, "Old_Lon": old_lon,
                "New_Lat": None, "New_Lon": None,
            })
            continue

        new_lat, new_lon = geocode_resort(gmaps, resort, country)

        if new_lat is None:
            missing_after += 1
            status = "STILL_MISSING"
            log.warning("FAIL  %-55s [%s]", resort[:55], country)
            # Mantieni le vecchie coordinate se esistono
            new_lat, new_lon = (float(old_lat) if pd.notna(old_lat) else None,
                                float(old_lon) if pd.notna(old_lon) else None)
        else:
            changed = (
                pd.isna(old_lat) or pd.isna(old_lon) or
                abs(float(old_lat) - new_lat) > 0.01 or
                abs(float(old_lon) - new_lon) > 0.01
            )
            if changed:
                fixed += 1
                status = "FIXED"
                log.info("FIX   %-50s → (%.4f, %.4f)", resort[:50], new_lat, new_lon)
            else:
                unchanged += 1
                status = "OK"

        df.at[idx, "Latitude"]  = new_lat
        df.at[idx, "Longitude"] = new_lon

        report_rows.append({
            "Resort": resort, "Country": country, "Status": status,
            "Old_Lat": old_lat, "Old_Lon": old_lon,
            "New_Lat": new_lat, "New_Lon": new_lon,
        })

    df.to_csv(GEO_CSV)
    pd.DataFrame(report_rows).to_csv(REPORT, index=False)

    log.info("─" * 65)
    log.info("Totale:          %d", len(df))
    log.info("Aggiornate:      %d", fixed)
    log.info("Già corrette:    %d", unchanged)
    log.info("Indoor (skip):   %d", indoor_skip)
    log.info("Ancora mancanti: %d", missing_after)
    log.info("Report:          %s", REPORT)
    log.info("CSV aggiornato:  %s", GEO_CSV)


if __name__ == "__main__":
    main()
