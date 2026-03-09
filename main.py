from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

import numpy as np

app = FastAPI(title="Fiske-AI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

model = joblib.load("fiske_modell.joblib")
le_omraade = joblib.load("le_omraade.joblib")
le_sluk = joblib.load("le_sluk.joblib")
le_fisk = joblib.load("le_fisk.joblib")


class FangstRequest(BaseModel):
    maaned: int
    dyp_meter: int
    vekt_kg: float
    omraade: str
    sluk: str
    fisk_type: str


@app.get("/")
def home():
    return {"status": "Online", "message": "Jeg er klar!"}


@app.post("/predict")
def predict_catch(data: FangstRequest):
    try:

        omr_enc = le_omraade.transform([data.omraade])[0]
        sluk_enc = le_sluk.transform([data.sluk])[0]
        fisk_enc = le_fisk.transform([data.fisk_type])[0]

        input_features = np.array(
            [
                [
                    data.maaned,
                    data.dyp_meter,
                    data.vekt_kg,
                    omr_enc,
                    sluk_enc,
                    fisk_enc,
                ]
            ]
        )

        sjanse_verdi = model.predict_proba(input_features)[0][1]

        return {
            "sjanse_for_at_den_slipper": f"{round(sjanse_verdi * 100, 1)}%",
            "anbefaling": (
                "Sveiv rolig, den kan glippe"
                if sjanse_verdi > 0.7
                else "Fortsett slik, snart er den i håven!"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
