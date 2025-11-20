from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, List
import pandas as pd
import os

# =========================
# Config API
# =========================
app = FastAPI(
    title="prix prediction API",
    description="API très simple - les prévisions XGB / LSTM / GRU sur les commodités.",
   #  version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# recuperation des resultat de prediction 
# =========================
MODEL_FILES = {
    "xgb": "xgboost/xgb_future.csv",
    "lstm": "lstm/lstm_future.csv",
    "gru": "gru/gru_future.csv",
}


def _load_future_df(model: str) -> pd.DataFrame:
    """
    Charge le CSV de prédictions futures associé au modèle.
    
    """
    if model not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Modèle inconnu: {model}")

    file_path = MODEL_FILES[model]
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=500,
            detail=f"Fichier de prédictions futures introuvable pour le modèle '{model}': {file_path}",
        )

    try:
        df = pd.read_csv(file_path, parse_dates=["date"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la lecture du fichier {file_path}: {e}",
        )

    if "commodity" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=f"Colonne 'commodity' manquante dans {file_path}",
        )

    # 
    if "predicted" in df.columns:
        df["prediction"] = df["predicted"]
    elif "predicted_close" in df.columns:
        df["prediction"] = df["predicted_close"]
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Aucune colonne de prédiction ('predicted' ou 'predicted_close') trouvée dans {file_path}",
        )

    return df


# =========================
# Endpoints
# =========================

@app.get("/health", summary="Vérifier que l'API est en ligne")
def health_check():
    return {"status": "ok"}


@app.get("/models", summary="Liste des modèles disponibles")
def list_models():
    """Retourne la liste des modèles pour lesquels un fichier de prédictions existe."""
    available = []
    for model, path in MODEL_FILES.items():
        if os.path.exists(path):
            available.append(model)
    return {"models": available}


@app.get("/commodities", summary="Liste des commodities disponibles pour un modèle donné")
def list_commodities(
    model: Literal["xgb", "lstm", "gru"]
):
    df = _load_future_df(model)
    commodities: List[str] = sorted(df["commodity"].unique().tolist())
    return {
        "model": model,
        "commodities": commodities,
    }


@app.get(
    "/predict",
    summary="Obtenir les prévisions futures pour un modèle et une commodity",
)
def predict(
    model: Literal["xgb", "lstm", "gru"],
    commodity: str,
    horizon: int = Query(
        30,
        ge=1,
        le=365,
        description="Nombre de jours à renvoyer (max 365 selon tes scripts).",
    ),
):
    """
    Retourne les prévisions futures pour un modèle donné (xgb, lstm, gru),
    une commodity et un horizon (en jours).
    Les données viennent des fichiers *_future.csv générés lors de l'entraînement.
    """
    df = _load_future_df(model)
    sub = df[df["commodity"] == commodity].sort_values("date")

    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée trouvée pour la commodity '{commodity}' avec le modèle '{model}'.",
        )

    # Limiter au nombre de jours demandé
    sub = sub.head(horizon)

    return {
        "model": model,
        "commodity": commodity,
        "horizon_requested": horizon,
        "horizon_returned": len(sub),
        "dates": [d.strftime("%Y-%m-%d") for d in sub["date"]],
        "predictions": sub["prediction"].tolist(),
    }
