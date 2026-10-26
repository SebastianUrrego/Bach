"""Fixtures compartidas por las pruebas de la API."""

import pytest
import torch
from fastapi.testclient import TestClient

import api.main as api
from bach.data import COLUMNS, load_split
from bach.models import Seq2Seq


class SpyModel(Seq2Seq):
    """Seq2Seq real (sin entrenar) que cuenta cuántas veces se llama a generate."""

    calls = 0

    def generate(self, *args, **kwargs):
        SpyModel.calls += 1
        return super().generate(*args, **kwargs)


@pytest.fixture
def seed() -> list[list[int]]:
    """Primeros 32 pasos de chorale_229 (valid), como en el ejemplo de la guía."""
    return load_split("valid")[0][COLUMNS].iloc[:32].values.tolist()


@pytest.fixture
def client(monkeypatch):
    """Cliente de la API con un modelo local inyectado: no necesita MLflow ni la EC2.

    Sin `with TestClient(...)` no se ejecuta el lifespan, así que no se consulta MLflow.
    """
    torch.manual_seed(0)
    SpyModel.calls = 0
    monkeypatch.setitem(api.state, "model", SpyModel(cell="gru").eval())
    monkeypatch.setitem(api.state, "name", "bach-chorales-prueba")
    monkeypatch.setitem(api.state, "version", "7")
    return TestClient(api.app)


@pytest.fixture
def client_sin_modelo(monkeypatch):
    """API cuyo modelo no pudo cargarse al arrancar."""
    monkeypatch.setitem(api.state, "model", None)
    monkeypatch.setitem(api.state, "error", "RestException: alias champion no encontrado")
    return TestClient(api.app)
