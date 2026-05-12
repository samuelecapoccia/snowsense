"""
Genera resort_map.html — mappa interattiva dei resort sciistici europei.
Usa: python generate_map.py
"""

import pandas as pd
import folium

CSV_PATH = "data/european_ski_resorts_geo.csv"
OUT_PATH = "resort_map.html"

df = pd.read_csv(CSV_PATH, index_col=0)

mapped = []
missing = []

for _, row in df.iterrows():
    lat = row.get("Latitude")
    lon = row.get("Longitude")
    if pd.notna(lat) and pd.notna(lon):
        mapped.append(row)
    else:
        missing.append(f"  - {row['Resort']} ({row['Country']})")

# Mappa centrata sull'Europa
m = folium.Map(location=[46.5, 10.0], zoom_start=5, tiles="CartoDB positron")

for row in mapped:
    folium.Marker(
        location=[float(row["Latitude"]), float(row["Longitude"])],
        popup=folium.Popup(
            f"<b>{row['Resort']}</b><br>"
            f"{row['Country']}<br>"
            f"Lat: {row['Latitude']:.4f} | Lon: {row['Longitude']:.4f}",
            max_width=220,
        ),
        tooltip=row["Resort"],
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

m.save(OUT_PATH)

print(f"Mappati:  {len(mapped)} resort → {OUT_PATH}")
print(f"Mancanti: {len(missing)} resort")
if missing:
    print("\nResort senza coordinate:")
    print("\n".join(missing))
