"""Pruebas de la API (guía, sección 6): solicitud válida, steps=33 y entradas inválidas.

    python api/probar_api.py http://3.229.168.127:8000 data/jsb_chorales/valid/chorale_229.csv
"""

import sys
import time

import pandas as pd
import requests

base = sys.argv[1]
coro = pd.read_csv(sys.argv[2])
seed = coro[["note0", "note1", "note2", "note3"]].iloc[:32].values.tolist()

for _ in range(60):
    try:
        if requests.get(f"{base}/health", timeout=2).ok:
            break
    except requests.ConnectionError:
        pass
    time.sleep(2)
print("health:", requests.get(f"{base}/health").json())


def check(label, payload, steps=None):
    r = requests.post(f"{base}/generate", json=payload)
    if r.ok:
        g = r.json()["generated"]
        ok = len(g) == steps and all(len(s) == 4 and all(0 <= n <= 127 for n in s) for s in g)
        print(f"{label}: {r.status_code} filas={len(g)} formato_ok={ok} primeras={g[:2]}")
    else:
        detail = r.json()["detail"]
        msgs = [f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in detail] if isinstance(detail, list) else detail
        print(f"{label}: {r.status_code} {msgs}")


check("valida steps=32", {"seed": seed, "steps": 32, "temperature": 1.0}, 32)
check("steps=33", {"seed": seed, "steps": 33}, 33)
check("steps=1 T=0.5", {"seed": seed, "steps": 1, "temperature": 0.5}, 1)
check("seed de 31 filas", {"seed": seed[:31], "steps": 16})
check("nota 128", {"seed": [[128, 70, 65, 58]] + seed[1:], "steps": 16})
check("steps=65", {"seed": seed, "steps": 65})
check("steps=0", {"seed": seed, "steps": 0})
check("temperature=2", {"seed": seed, "steps": 16, "temperature": 2})
check("nota 60.5", {"seed": [[60.5, 70, 65, 58]] + seed[1:], "steps": 16})
check("sin steps", {"seed": seed})
