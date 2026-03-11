import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("fiske_fangst.csv")

le_fisk = LabelEncoder()
le_omraade = LabelEncoder()
le_sluk = LabelEncoder()


df["fisk_type_enc"] = le_fisk.fit_transform(df["fisk_type"])
df["omraade_enc"] = le_omraade.fit_transform(df["omraade"])
df["sluk_type_enc"] = le_sluk.fit_transform(df["sluk_type"])


X = df[
    [
        "maaned",
        "dyp_fanget_meter",
        "vekt_kg",
        "omraade_enc",
        "sluk_type_enc",
        "fisk_type_enc",
    ]
]
y = df["fisk_som_slapp"]


model = RandomForestClassifier(
    n_estimators=100, max_depth=6, n_jobs=-1, random_state=23
)
model.fit(X, y)

importances = model.feature_importances_
for i, v in enumerate(importances):
    print(f"Feature: {X.columns[i]}, Score: {v:.4f}")

joblib.dump(model, "fiske_modell.joblib")
joblib.dump(le_fisk, "le_fisk.joblib")
joblib.dump(le_omraade, "le_omraade.joblib")
joblib.dump(le_sluk, "le_sluk.joblib")

print("Ferdig! Ordbøkene er klare for fastAPI")
