"""
Mappa di verifica: mostra in rosso i resort modificati (FIXED),
in blu tutti gli altri. Permette di identificare visivamente coordinate errate.

Esecuzione:
    .venv/bin/python scripts/verify_map.py && open verify_map.html
"""

from pathlib import Path
import pandas as pd
import folium

ROOT = Path(__file__).resolve().parents[1]

geo    = pd.read_csv(ROOT / "data" / "european_ski_resorts_geo.csv", index_col=0)
report = pd.read_csv(ROOT / "data" / "coordinate_fixes_report.csv")

fixed_names = set(report[report["Status"] == "FIXED"]["Resort"])

m = folium.Map(location=[46.5, 10.0], zoom_start=5, tiles="CartoDB positron")

for _, row in geo.iterrows():
    if pd.isna(row.get("Latitude")) or pd.isna(row.get("Longitude")):
        continue

    is_fixed = row["Resort"] in fixed_names
    color    = "red" if is_fixed else "blue"
    label    = "MODIFICATO" if is_fixed else "OK"

    folium.Marker(
        location=[float(row["Latitude"]), float(row["Longitude"])],
        popup=folium.Popup(
            f"<b>{row['Resort']}</b><br>"
            f"{row['Country']}<br>"
            f"Lat: {row['Latitude']:.4f} | Lon: {row['Longitude']:.4f}<br>"
            f"<b style='color:{'red' if is_fixed else 'green'}'>{label}</b>",
            max_width=260,
        ),
        tooltip=f"{'⚠ ' if is_fixed else ''}{row['Resort']}",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

out = ROOT / "verify_map.html"
m.save(str(out))
print(f"Resort totali mappati: {len(geo.dropna(subset=['Latitude']))}")
print(f"Resort FIXED (rosso):  {len(fixed_names)}")
print(f"Mappa salvata: {out}")
