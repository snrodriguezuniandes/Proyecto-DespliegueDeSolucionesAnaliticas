from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Forecast - Mercalia")

# Cargar los 3 modelos al iniciar la API
modelos = {
    "19127": joblib.load("models/modelo_19127_colombia.pkl"),
    "294000": joblib.load("models/modelo_294000_Colombia.pkl"),
    "455151": joblib.load("models/modelo_455151_colombia.pkl"),
}

class PrediccionRequest(BaseModel):
    sku: str
    dia_semana: int
    mes: int
    semana_anio: int
    es_finde: int
    lag_7: float
    lag_14: float
    lag_30: float
    media_movil_7: float
    media_movil_30: float

@app.get("/")
def root():
    return {"mensaje": "API Forecast Mercalia - Grupo 4"}

@app.post("/predict")
def predecir(datos: PrediccionRequest):
    if datos.sku not in modelos:
        return {"error": f"SKU {datos.sku} no reconocido"}

    modelo = modelos[datos.sku]

    features = pd.DataFrame([{
        "dia_semana": datos.dia_semana,
        "mes": datos.mes,
        "semana_anio": datos.semana_anio,
        "es_finde": datos.es_finde,
        "lag_7": datos.lag_7,
        "lag_14": datos.lag_14,
        "lag_30": datos.lag_30,
        "media_movil_7": datos.media_movil_7,
        "media_movil_30": datos.media_movil_30,
    }])

    prediccion = modelo.predict(features)[0]

    return {"sku": datos.sku, "prediccion": float(prediccion)}