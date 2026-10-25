"""API de generación de corales.

Al iniciar resuelve el alias `champion` del modelo registrado en MLflow y carga esa versión.
No vuelve a consultar MLflow en cada solicitud; si cambia el alias hay que reiniciar el servicio.

Variables de entorno: MLFLOW_TRACKING_URI, MODEL_NAME (y opcionalmente MODEL_ALIAS).
Ejecutar: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow.pytorch
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, StrictInt

load_dotenv()
log = logging.getLogger("uvicorn.error")

MODEL_NAME = os.getenv("MODEL_NAME", "bach-chorales-equipoXX")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

state: dict = {"model": None, "name": MODEL_NAME, "version": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
        model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}", map_location="cpu")
        model.eval()
        state.update(model=model, version=str(version))
        log.info("Modelo cargado: %s versión %s (alias %s)", MODEL_NAME, version, MODEL_ALIAS)
    except Exception as e:  # el servicio sigue arriba y /health informa el problema
        state["error"] = f"{type(e).__name__}: {e}"
        log.error("No se pudo cargar %s@%s: %s", MODEL_NAME, MODEL_ALIAS, state["error"])
    yield


app = FastAPI(title="Bach chorales API", version="1.0", lifespan=lifespan)

Note = Annotated[StrictInt, Field(ge=0, le=127)]
Step = Annotated[list[Note], Field(min_length=4, max_length=4)]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"seed": [[74, 70, 65, 58]] * 32, "steps": 32, "temperature": 1.0}]
    })

    seed: Annotated[list[Step], Field(min_length=32, max_length=32,
                                      description="32 pasos x 4 voces (note0..note3), enteros 0-127")]
    steps: Annotated[StrictInt, Field(ge=1, le=64, description="Pasos nuevos a generar, sin contar el contexto")]
    temperature: Annotated[float, Field(ge=0.5, le=1.5)] = 1.0


class GenerateResponse(BaseModel):
    generated: list[list[int]]
    model: str
    version: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


def loaded_model():
    if state["model"] is None:
        raise HTTPException(503, detail=f"Modelo no disponible: {state['error']}")
    return state["model"]


@app.get("/health", response_model=HealthResponse)
def health():
    loaded_model()
    return HealthResponse(status="ok", model=state["name"], version=state["version"])


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = loaded_model()
    context = torch.tensor(req.seed, dtype=torch.long).unsqueeze(0)  # (1, 32, 4)
    with torch.no_grad():
        new = model.generate(context, steps=req.steps, temperature=req.temperature)
    return GenerateResponse(generated=new[0].tolist(), model=state["name"], version=state["version"])
