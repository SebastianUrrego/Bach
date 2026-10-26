"""Pruebas contra la API desplegada (modelo champion real cargado desde MLflow).

    pytest test/test_despliegue.py                       # usa la EC2
    BACH_API_URL=http://127.0.0.1:8000 pytest test/test_despliegue.py

Se omiten si la API no responde (por ejemplo, con el Learner Lab apagado).
"""

import os

import pytest
import requests

URL = os.getenv("BACH_API_URL", "http://3.229.168.127:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "bach-chorales-equipoF")

pytestmark = pytest.mark.despliegue


@pytest.fixture(scope="module")
def health():
    try:
        r = requests.get(f"{URL}/health", timeout=10)
    except requests.RequestException as e:
        pytest.skip(f"API no disponible en {URL}: {type(e).__name__}")
    return r


def post(payload):
    return requests.post(f"{URL}/generate", json=payload, timeout=60)


def test_health_champion(health):
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_NAME
    assert body["version"].isdigit()


@pytest.mark.parametrize("steps", [32, 33])
def test_generate_valida(health, seed, steps):
    r = post({"seed": seed, "steps": steps, "temperature": 1.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["generated"]) == steps
    assert all(len(row) == 4 and all(0 <= n <= 127 for n in row) for row in body["generated"])
    assert body["model"] == MODEL_NAME
    assert body["version"] == health.json()["version"]


def test_modelo_entrenado_respeta_los_registros(health, seed):
    """Con el modelo entrenado, cada voz debe quedar cerca de su registro (no notas al azar en 0-127)."""
    r = post({"seed": seed, "steps": 64, "temperature": 1.0})
    g = r.json()["generated"]
    registros = [(55, 86), (47, 79), (41, 74), (31, 71)]  # rango de train de cada voz ± 5 semitonos
    for v, (lo, hi) in enumerate(registros):
        notas = [row[v] for row in g if row[v] > 0]
        assert all(lo <= n <= hi for n in notas), f"voz {v}: {notas}"


@pytest.mark.parametrize("payload", [
    {"steps": 16},
    {"seed": [[60, 60, 60, 60]] * 31, "steps": 16},
    {"seed": [[128, 60, 60, 60]] * 32, "steps": 16},
    {"seed": [[60, 60, 60, 60]] * 32, "steps": 65},
    {"seed": [[60, 60, 60, 60]] * 32, "steps": 16, "temperature": 2},
], ids=["sin seed", "31 filas", "nota 128", "steps 65", "temperature 2"])
def test_entrada_invalida(health, payload):
    assert post(payload).status_code == 422
