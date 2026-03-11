import pandas as pd
import numpy as np


n_rader = 200000
np.random.seed(42)

print(f"Starter generering av {n_rader} rader. Gjør klar Oppgavebehandling!")


start_dato = pd.to_datetime("2025-01-01")
tilfeldige_dager = np.random.randint(0, 365, n_rader)
dato_liste = start_dato + pd.to_timedelta(tilfeldige_dager, unit="D")

data = {
    "dato_fangst": dato_liste,
    "fisk_type": np.random.choice(
        [
            "Torsk",
            "Sei",
            "Laks",
            "Makrell",
            "Kveite",
            "Lyr",
            "Brosme",
            "Breiflabb",
            "Pale",
            "Flyndre",
            "Uer",
        ],
        n_rader,
    ),
    "dyp_fanget_meter": np.random.randint(5, 250, n_rader),
    "sluk_type": np.random.choice(
        [
            "Møresilda",
            "Stingsild",
            "Sluk",
            "Flue",
            "Jigg",
            "Rema-sluk",
            "Fireball",
            "Wobbler",
            "Spinner",
        ],
        n_rader,
    ),
    "vekt_kg": np.random.uniform(0.5, 30.0, n_rader).round(1),
    "omraade": np.random.choice(
        ["Lysefjorden", "Blia", "Krokeide", "Austevoll", "Frekhaug", "Sotra", "Askøy"],
        n_rader,
    ),
}

df = pd.DataFrame(data)
df["maaned"] = df["dato_fangst"].dt.month


def beregn_kompleks_drama(row):

    score = (row["vekt_kg"] * 0.045) + (row["dyp_fanget_meter"] * 0.0025)

    if row["fisk_type"] == "Makrell":
        if row["maaned"] in [7, 8, 9]:
            score -= 0.45
        elif row["maaned"] in [1, 2, 3]:
            score += 0.65

    if row["sluk_type"] == "Stingsild" and row["dyp_fanget_meter"] > 80:
        score -= 0.25

    if row["sluk_type"] == "Flue" and row["vekt_kg"] > 6.0:
        score += 0.35

    if row["omraade"] == "Austevoll":
        score -= 0.1

    score += np.random.normal(0, 0.3)

    return 1 if score > 0.85 else 0


print("Beregner drama_score for alle rader... (Dette tar noen sekunder på én kjerne)")
df["fisk_som_slapp"] = df.apply(beregn_kompleks_drama, axis=1)


df.to_csv("fiske_fangst.csv", index=False)

print("\n--- STATUS ---")
print(f"Datasettet er lagret. Filstørrelse: Ca. {(n_rader * 100) / 1024 / 1024:.1f} MB")
print(f"Antall som slapp: {df['fisk_som_slapp'].sum()} av {n_rader}")
print(f"Andel som slapp: {round((df['fisk_som_slapp'].sum() / n_rader) * 100, 1)}%")
