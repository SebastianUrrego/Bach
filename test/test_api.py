"""Pruebas de la API con un modelo inyectado (sin red): contrato de entrada/salida y validación.

    pytest test/test_api.py
"""

import copy

import pytest

from conftest import SpyModel


def assert_generated(body: dict, steps: int) -> None:
    """generated es una matriz steps x 4 de enteros 0-127, sin el contexto."""
    g = body["generated"]
    assert len(g) == steps
    assert all(len(row) == 4 for row in g)
    assert all(isinstance(n, int) and 0 <= n <= 127 for row in g for n in row)


# --- GET /health -------------------------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "bach-chorales-prueba", "version": "7"}


def test_health_sin_modelo_responde_503(client_sin_modelo):
    r = client_sin_modelo.get("/health")
    assert r.status_code == 503
    assert "champion" in r.json()["detail"]


# --- POST /generate: solicitudes válidas -------------------------------------------------------------------

def test_generate_valida(client, seed):
    r = client.post("/generate", json={"seed": seed, "steps": 32, "temperature": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert_generated(body, 32)
    assert body["model"] == "bach-chorales-prueba"
    assert body["version"] == "7" and isinstance(body["version"], str)
    assert SpyModel.calls == 1  # el espía cuenta las llamadas: valida las pruebas de entradas inválidas


@pytest.mark.parametrize("steps", [1, 15, 16, 17, 33, 48, 64])
def test_generate_devuelve_los_pasos_pedidos(client, seed, steps):
    """Incluye longitudes no múltiplos de 16 (el último bloque se corta)."""
    r = client.post("/generate", json={"seed": seed, "steps": steps})
    assert r.status_code == 200
    assert_generated(r.json(), steps)


@pytest.mark.parametrize("temperature", [0.5, 1, 1.5])
def test_generate_temperaturas_limite(client, seed, temperature):
    r = client.post("/generate", json={"seed": seed, "steps": 16, "temperature": temperature})
    assert r.status_code == 200
    assert_generated(r.json(), 16)


def test_temperature_es_opcional(client, seed):
    r = client.post("/generate", json={"seed": seed, "steps": 8})
    assert r.status_code == 200
    assert_generated(r.json(), 8)


def test_silencios_en_el_contexto_son_validos(client, seed):
    seed[0] = [0, 0, 0, 0]
    r = client.post("/generate", json={"seed": seed, "steps": 4})
    assert r.status_code == 200


# --- POST /generate: entradas inválidas (422 antes de llamar al modelo) ------------------------------------

def _con(base, **cambios):
    payload = {"seed": copy.deepcopy(base), "steps": 16, "temperature": 1.0}
    payload.update(cambios)
    return payload


INVALIDAS = {
    "seed con 31 filas": lambda s: _con(s, seed=s[:31]),
    "seed con 33 filas": lambda s: _con(s, seed=s + s[:1]),
    "fila con 3 notas": lambda s: _con(s, seed=[s[0][:3]] + s[1:]),
    "fila con 5 notas": lambda s: _con(s, seed=[s[0] + [60]] + s[1:]),
    "nota 128": lambda s: _con(s, seed=[[128, 70, 65, 58]] + s[1:]),
    "nota negativa": lambda s: _con(s, seed=[[-1, 70, 65, 58]] + s[1:]),
    "nota decimal": lambda s: _con(s, seed=[[60.5, 70, 65, 58]] + s[1:]),
    "nota como texto": lambda s: _con(s, seed=[["60", 70, 65, 58]] + s[1:]),
    "nota booleana": lambda s: _con(s, seed=[[True, 70, 65, 58]] + s[1:]),
    "steps 0": lambda s: _con(s, steps=0),
    "steps 65": lambda s: _con(s, steps=65),
    "steps decimal": lambda s: _con(s, steps=16.5),
    "steps como texto": lambda s: _con(s, steps="16"),
    "temperature 0.4": lambda s: _con(s, temperature=0.4),
    "temperature 1.6": lambda s: _con(s, temperature=1.6),
    "sin seed": lambda s: {"steps": 16},
    "sin steps": lambda s: {"seed": s},
    "campo extra": lambda s: _con(s, top_k=5),
}


@pytest.mark.parametrize("caso", INVALIDAS)
def test_entrada_invalida_422_sin_llamar_al_modelo(client, seed, caso):
    r = client.post("/generate", json=INVALIDAS[caso](seed))
    assert r.status_code == 422, r.text
    assert SpyModel.calls == 0


def test_cuerpo_que_no_es_json(client):
    r = client.post("/generate", content=b"no es json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert SpyModel.calls == 0


def test_generate_sin_modelo_responde_503(client_sin_modelo, seed):
    r = client_sin_modelo.post("/generate", json={"seed": seed, "steps": 16})
    assert r.status_code == 503
