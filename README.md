# Bach — Generación de secuencias musicales con modelos recurrentes y atención

Laboratorio de **Procesamiento de Lenguaje Natural – 2026 S02** (Universidad Sergio Arboleda) · **Equipo F**

Se entrena un encoder-decoder para continuar corales de Bach: a partir de **32 pasos** (4 voces por paso,
notas 0–127, `0` = silencio) el modelo predice los **16 pasos siguientes**. Se comparan LSTM, GRU y GRU con
atención, más un cuarto experimento; el modelo se elige por **pérdida de validación** en **MLflow** y se
publica con una **API FastAPI**.

| Componente | Enlace / estado |
|------------|-----------------|
| MLflow (experimento `nlp-bach-equipoF`) | http://3.229.168.127:5000 — acceso restringido por IP (Security Group) |
| API FastAPI | http://3.229.168.127:8000/docs — pendiente de registrar `champion` |
| Modelo elegido | **E4** · run `042df735c704434f8e2cac7e58d922c0` |

| Fase | Estado |
|------|--------|
| 0. Entorno (EC2 + MLflow + S3) | ✅ |
| 1. Datos y ventanas | ✅ |
| 2. Modelos E1–E3 | ✅ |
| 3. E4, selección y test | ✅ |
| 4. Generación musical (2 MIDI) | ⏳ |
| 5. Registro `champion` y API | ⏳ código listo, falta registrar |
| 6. Informe final | ⏳ este README |

---

## 1. Datos

Se usa JSB Chorales en la versión de Géron, con sus particiones originales. Cada CSV tiene las columnas
`note0, note1, note2, note3`; cada fila es una semicorchea. `python -m bach.data` descarga los datos y
ejecuta estas comprobaciones:

| Partición | Corales | Pasos por coral (mín–máx) | Silencios | Stride | Ventanas X (n,32,4) / Y (n,16,4) |
|-----------|--------:|:----:|:----:|:----:|-----:|
| train | 229 | 100–516 | 0.1–0.2 % | 8 | 5 681 |
| valid | 76 | 128–576 | 0.8 % | 16 | 982 |
| test | 77 | 128–640 | 0.4 % | 16 | 1 011 |

- Vocabulario fijo 0–127 sin recodificar. Los valores reales van de 36 a 81 (más el `0`).
- Las ventanas de 48 pasos se cortan **dentro de cada coral** y se descartan los fragmentos incompletos.
  Ninguna ventana cruza corales ni particiones, y las mismas ventanas se usan en todos los experimentos.
- El silencio `0` es una clase más: no se usa como relleno (no hace falta, las ventanas tienen longitud fija)
  y **entra en la pérdida**.

## 2. Modelos y protocolo

| ID | Arquitectura | Qué compara |
|----|--------------|-------------|
| E1 | Encoder-decoder LSTM | Referencia |
| E2 | Encoder-decoder GRU | Celda recurrente, frente a E1 |
| E3 | Encoder-decoder GRU + atención aditiva (Bahdanau) | Efecto de la atención, frente a E2 |
| E4 | E2 con **hidden 256** | Un solo hiperparámetro sobre el mejor de E1–E3 |

Decisiones comunes ([`src/bach/models.py`](src/bach/models.py)):

- **Entrada de cada paso:** una tabla de embeddings compartida (16 dims por nota). Los 4 vectores se concatenan
  en el orden de las voces, así que cada paso es un vector de 64.
- **Estado inicial del decoder:** el estado final del encoder (en LSTM se transfieren `h` y `c`).
- **E3:** en cada paso del decoder se calcula `score = vᵀ tanh(W_q h + W_k e_j)` sobre las 32 salidas del encoder;
  el contexto resultante se concatena con la entrada del decoder. Todo lo demás es igual a E2.
- **Teacher forcing** en entrenamiento y validación: las entradas del decoder son el último paso de X seguido de
  `Y[0..14]`. En generación libre, cada entrada es la muestra anterior.
- **Salida:** logits `(batch, 16, 4, 128)`. La pérdida es `CrossEntropyLoss` sobre `(batch·16·4, 128)`, sin softmax previo.
- **Exactitud por nota:** notas acertadas con argmax. **Exactitud por paso:** pasos con las 4 notas correctas.
  Ambas se acumulan sobre toda la partición y se miden en modo evaluación, sin gradientes y con teacher forcing.

Hiperparámetros base, iguales en E1–E3:
- 1 capa unidireccional, hidden 128, embedding 16;
- Adam con lr 1e-3, batch 64;
- máximo 20 épocas, con parada temprana tras 4 épocas sin bajar la pérdida de validación;
- semilla 42 reiniciada en cada run.

Equipo de cómputo: CPU Intel Core i5-11400H (6 hilos de torch), torch 2.14.1+cpu, Windows 11.

Cada run guarda el **checkpoint de menor pérdida de validación**, lo recarga al terminar y calcula las
exactitudes **con ese checkpoint**, nunca con la última época. Todo queda en MLflow:

| Tipo | Qué se guarda |
|------|---------------|
| Parámetros | arquitectura, atención, capas, hidden, embedding, lr, batch, épocas máximas, semilla, contexto, horizonte |
| Métricas | pérdidas por época; mejor época, mejor pérdida de validación y exactitudes; parámetros entrenables; tiempo |
| Artefactos | `config.json`, `history.json`, curva de pérdidas, checkpoint, `hardware.json`; en E3, el mapa de atención |

## 3. Resultados E1–E4

Métricas de **validación** del mejor checkpoint de cada run:

| Run | Arquitectura | Hidden | Parámetros entrenables | Mejor época / épocas | Val loss | Exactitud nota | Exactitud paso | Tiempo |
|-----|--------------|-------:|-----------:|:----:|:----:|:----:|:----:|----:|
| E1 | LSTM | 128 | 266 752 | 20 / 20 | 0.7494 | 0.7896 | 0.5985 | 32 s |
| E2 | GRU | 128 | 217 088 | 19 / 20 | 0.6339 | 0.8158 | 0.6382 | 54 s |
| E3 | GRU + atención | 128 | 299 136 | 17 / 20 | 0.7181 | 0.7908 | 0.6058 | 123 s |
| **E4** | **GRU** | **256** | **628 224** | **12 / 16** | **0.6057** | **0.8217** | **0.6451** | 74 s |

![Pérdida por época de E1–E4](docs/img/comparacion_perdidas.png)

La línea continua es la validación y la punteada el entrenamiento; ★ marca la época del checkpoint guardado.
En las primeras épocas la pérdida de entrenamiento queda por encima de la de validación. La razón es que la de
entrenamiento es el promedio de toda la época, mientras el modelo todavía aprende, y la de validación se mide al
final de la época.

<table>
<tr><td><img src="docs/img/loss_E1.png" alt="Curva E1"></td><td><img src="docs/img/loss_E2.png" alt="Curva E2"></td></tr>
<tr><td><img src="docs/img/loss_E3.png" alt="Curva E3"></td><td><img src="docs/img/loss_E4.png" alt="Curva E4"></td></tr>
</table>

### E1 frente a E2: LSTM vs GRU

- **El GRU gana:** baja la pérdida de validación de 0.749 a 0.634 (−15 %) y sube la exactitud por paso de 0.599 a
  0.638, con un **19 % menos de parámetros** (el GRU tiene 3 compuertas por capa; el LSTM, 4 y una celda de memoria).
- **El LSTM no había convergido:** su mejor época fue la 20, la última permitida, y la curva seguía bajando. La
  comparación es justa con el protocolo común de 20 épocas, pero no muestra el techo del LSTM. Con más épocas la
  diferencia podría reducirse.
- **El GRU aprende más rápido por época:** en la época 4 ya tenía una pérdida de validación de 0.94; el LSTM
  necesitó 7 épocas para llegar a 0.92.

### E2 frente a E3: efecto de la atención

- **La atención no mejoró:** E3 termina en 0.718, frente a 0.634 de E2, y con **un 38 % más de parámetros**
  (W_q, W_k, v y una entrada al decoder 128 dimensiones más ancha).
- **E3 sobreajusta antes:** su mejor época es la 17; después la pérdida de entrenamiento sigue bajando
  (0.56 → 0.51) mientras la de validación sube (0.718 → 0.730).
- **Una posible explicación, no comprobada:** con un contexto de solo 32 pasos, el estado final del GRU ya
  resume bien el fragmento, y la atención agrega capacidad que aquí se traduce en sobreajuste más que en
  información nueva.
- **Costo:** el decoder con atención se ejecuta paso a paso y E3 tardó 2,3 veces más que E2.

### E4: por qué hidden 256 y qué mostró

E4 parte de **E2**, el mejor de E1–E3, y cambia **solo** la dimensión oculta de 128 a 256. Se entrena desde cero,
con las mismas ventanas, semilla y métricas.

- **Por qué este cambio:** E2 (el más pequeño) ganó y E3 (más grande, pero con atención) perdió. Faltaba saber si
  el problema de E3 era el tamaño o la atención. E4 separa esas dos variables: más capacidad, sin atención.
- **Qué mostró:** la pérdida de validación bajó de 0.634 a **0.606** (−4,4 %) y subieron las dos exactitudes.
  Entonces el peor resultado de E3 no se explica solo por tener más parámetros.
- **Comportamiento del entrenamiento:** E4 converge mucho antes (mejor época 12 frente a 19) y es el **único run
  que activó la parada temprana**, en la época 16. Desde la época 13 la pérdida de entrenamiento sigue bajando
  (0.49 → 0.41) y la de validación sube un poco: empieza a sobreajustar. El checkpoint guardado es el de la época 12.

### Modelo elegido: E4

**Criterio:** el que fija la guía, la menor pérdida de validación entre todos los runs comparables. E4 tiene la
menor (0.6057), además de las mejores exactitudes por nota (0.822) y por paso (0.645). La selección la hace
[`src/bach/select.py`](src/bach/select.py) leyendo los runs de MLflow.

Al interpretar la elección hay que tener en cuenta:
- **Una sola semilla:** la semilla 42 hace comparables las ejecuciones, pero con un solo entrenamiento por
  configuración no se puede afirmar que el GRU de 256 sea superior en general. La diferencia con E2 (0.028 de
  pérdida, 0.6 puntos de exactitud por nota) es pequeña y podría cambiar con otra semilla.
- **El tamaño:** E4 tiene **2,9 veces los parámetros de E2** (628 k frente a 217 k). La mejora tiene un costo en
  memoria y en tiempo de inferencia, aunque para la API en CPU sigue siendo un modelo pequeño (unos 2,5 MB).
- **La pérdida es una medida predictiva, no musical:** mide qué tan bien se predice la nota real con teacher
  forcing. No garantiza que la continuación generada libremente suene mejor; eso se analiza en la generación.

### Evaluación en test

Se evaluó **solo E4, después de elegirlo**, con su checkpoint de la época 12 (descargado del run en MLflow) y el
mismo protocolo de validación. Las métricas se registraron en su run, con la etiqueta `seleccionado=true`.
`test` no se usó para ninguna decisión de arquitectura, hiperparámetros ni época.

| Partición | Pérdida | Exactitud nota | Exactitud paso |
|-----------|:----:|:----:|:----:|
| valid | 0.6057 | 0.8217 | 0.6451 |
| **test** | **0.6256** | **0.8157** | **0.6268** |

Test queda un poco por debajo de validación (+0.02 de pérdida), lo esperado al medir sobre corales que no se
usaron para elegir el modelo.

### Mapa de atención de E3

Para el mapa se usó el mejor checkpoint de E3 (época 17), en **generación libre** (sin teacher forcing) de 16 pasos.
El contexto son los primeros 32 pasos de `chorale_229` (valid), con T = 1.0 y semilla 42.

![Mapa de atención de E3](docs/img/attention_E3.png)

- El peso se concentra en las **posiciones 23–24 del contexto** para casi todos los pasos generados, con picos de
  hasta 0.22 en los pasos 3 y 16. Una distribución uniforme daría 1/32 ≈ 0.03 a cada posición.
- Las **posiciones 1–6** reciben un peso prácticamente nulo en todos los pasos.
- Algunos pasos reparten parte del peso hacia el final del contexto (posiciones 25–32); por ejemplo, el paso 11
  también da un peso alto a las posiciones 30–32.
- El patrón cambia poco de un paso generado a otro: el decoder mira casi siempre a la misma zona del contexto.

Estas observaciones describen dónde está el peso. **No explican por qué el modelo predice lo que predice**: un peso
alto no implica que esa posición cause la predicción.

## 4. Generación musical

⏳ Pendiente: continuación de 2 corales de valid con E4 (≥ 32 pasos nuevos, bloques de 16, T = 1.0), exportada a
2 archivos MIDI con `pretty_midi`.

## 5. Registro del modelo y API

⏳ Pendiente: registrar E4 con `mlflow.pytorch.log_model` como `bach-chorales-equipoF`, con el alias `champion`.

El código de la API ya está listo y desplegado en la EC2 ([`api/main.py`](api/main.py)). Al arrancar resuelve
`champion` y carga esa versión desde MLflow. Expone `POST /generate` (`seed` 32×4, `steps` 1–64, `temperature`
0.5–1.5) y `GET /health`, y valida las entradas con Pydantic antes de llamar al modelo.

## 6. Cómo ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (en Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
copy .env.example .env            # y completar MLFLOW_TRACKING_URI
```

| Paso | Comando |
|------|---------|
| Verificar datos | `python -m bach.data` |
| Entrenar | `python -m bach.train E1` (E2, E3; E4: `python -m bach.train E4 --cell gru --no-attention --hidden 256 --change "hidden 128 -> 256 (base E2)"`) |
| Elegir y evaluar en test | `python -m bach.select` |
| Figuras del README | `python -m bach.report` |
| API local | `uvicorn api.main:app --port 8000` |

El despliegue en AWS (EC2 con MLflow y la API como servicios systemd, artefactos en S3) está documentado en
[`deploy/README.md`](deploy/README.md).

## 7. Estructura

```
Bach/
├── src/bach/
│   ├── data.py       # carga de CSV, verificación, ventanas 32→16, DataLoaders
│   ├── models.py     # Seq2Seq LSTM/GRU, atención aditiva, generación por bloques
│   ├── metrics.py    # pérdida y exactitudes
│   ├── train.py      # entrenamiento + MLflow
│   ├── select.py     # selección por val loss + test
│   ├── plots.py      # curvas y mapa de atención
│   └── report.py     # figuras del README desde MLflow
├── api/main.py       # FastAPI
├── deploy/           # instalación en EC2 (MLflow + API)
├── docs/img/         # figuras del README
└── outputs/resultados/  # resumen por run, tabla y selección
```

## 8. Uso de IA generativa

Se usó **Claude Code** (Anthropic, modelo Claude Opus) como asistente para:
- escribir y depurar el código de datos, modelos, entrenamiento, selección y API;
- preparar los scripts de despliegue en AWS;
- redactar este README.

El equipo revisó el código, ejecutó y verificó cada resultado (las métricas salen de los runs en MLflow) y tomó
las decisiones del laboratorio, como el cambio de E4. No se usaron modelos preentrenados ni servicios externos de
generación musical: los cuatro modelos se entrenaron desde cero con los datos del laboratorio.

## Referencias

- Géron, A. *JSB Chorales Dataset* — https://github.com/ageron/handson-ml2/tree/master/datasets/jsb_chorales
- Boulanger-Lewandowski, Bengio y Vincent (2012) — https://arxiv.org/abs/1206.6392
- Bahdanau, Cho y Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate* — https://arxiv.org/abs/1409.0473
- PyTorch, *Seq2seq con atención* — https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- MLflow Model Registry — https://mlflow.org/docs/latest/ml/model-registry/workflow/
- FastAPI — https://fastapi.tiangolo.com/tutorial/body/
- pretty_midi — https://craffel.github.io/pretty-midi/
