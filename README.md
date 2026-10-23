# Bach — Generación de secuencias musicales con modelos recurrentes y atención

Laboratorio de **Procesamiento de Lenguaje Natural – 2026 S02** (Universidad Sergio Arboleda).

> Estado: **en planificación**. Este README describe lo que se debe hacer y la hoja de ruta. Al final del laboratorio se convertirá en el informe con resultados.

## Objetivo

Continuar corales de Bach con modelos encoder-decoder. A partir de **32 pasos musicales** (4 voces por paso), el modelo predice los **16 pasos siguientes**. Se comparan tres arquitecturas y un cuarto experimento, se elige el modelo con **menor pérdida de validación** usando **MLflow**, y se publica con una **API en FastAPI** que recibe 32 pasos y devuelve una continuación.

| ID | Arquitectura | Propósito |
|----|--------------|-----------|
| E1 | Encoder-decoder LSTM | Referencia |
| E2 | Encoder-decoder GRU | Comparar celda recurrente con E1 |
| E3 | Encoder-decoder GRU + atención aditiva | Comparar con E2 para medir el efecto de la atención |
| E4 | Mejor de E1–E3 cambiando **un** hiperparámetro | Experimento adicional, entrenado desde cero |

## Lo que toca hacer

### 1. Datos

- Dataset: [`jsb_chorales`](https://github.com/ageron/handson-ml2/tree/master/datasets/jsb_chorales) (382 corales en CSV, cada paso = semicorchea, 4 enteros).
- Particiones originales: `train` 229, `valid` 76, `test` 77.
- Vocabulario fijo **0–127** sin recodificar; columnas en orden `note0, note1, note2, note3`.
- El silencio `0` es una categoría válida: **no** es padding y **sí** entra en la pérdida.
- Ventanas de 48 pasos dentro de cada coral (X = 32, Y = 16):
  - stride **8** en `train`, stride **16** en `valid` y `test`;
  - se descartan los fragmentos incompletos y ninguna ventana cruza corales ni particiones.
- Tensores: X `(batch, 32, 4)`, Y `(batch, 16, 4)`.

### 2. Modelos

- Embedding por nota (16 dims) y concatenación de las 4 voces en cada paso.
- El decoder se inicializa con el estado final del encoder (en LSTM, `h` y `c`).
- E3 agrega atención aditiva sobre los 32 estados del encoder.
- **Teacher forcing** en entrenamiento y validación: el decoder recibe el último paso de X y luego los primeros 15 pasos de Y.
- Salida: logits `(batch, 16, 4, 128)`.

**Configuración base:**
- 1 capa unidireccional, hidden 128, Adam con lr `1e-3`, batch 64, máximo 20 épocas;
- semilla 42 al inicio de cada run;
- early stopping con paciencia 4.

**Métricas** (con argmax, sin muestreo):
- pérdida: `CrossEntropyLoss` con logits `(batch*16*4, 128)`, sin softmax previo;
- **exactitud por nota**;
- **exactitud por paso completo** (las 4 notas correctas).

### 3. Experimentos y MLflow

- Un experimento propio con los runs E1–E4.
- **Parámetros:** arquitectura, atención, capas, hidden, embedding, lr, batch, épocas, semilla, contexto y horizonte.
- **Métricas:** pérdida de train y de validación por época, mejor época, mejor pérdida de validación y exactitudes del mejor checkpoint.
- **Artefactos:** configuración, curva de pérdidas, mejor checkpoint, número de parámetros entrenables, tiempo, hardware y, en E3, el mapa de atención.
- La selección se hace por pérdida de validación. Solo el modelo elegido se evalúa en `test`.

### 4. Generación musical

- Tomar los primeros 32 pasos de **2 corales de `valid`** y generar **≥ 32 pasos nuevos** de cada uno, sin teacher forcing.
- Generar en bloques de 16 pasos, usando como contexto de cada bloque los últimos 32 pasos.
- Muestrear con `softmax(logits / T)`, con T = 1.0 como referencia.
- Exportar **2 archivos MIDI** (contexto + continuación) con `pretty_midi`:
  - `0` representa silencio;
  - las notas iguales consecutivas se fusionan;
  - se usa un tempo común y documentado.
- Hacer el **mapa de atención de E3** (16 pasos generados × 32 de contexto, con los ejes rotulados).

### 5. Registro del modelo

- Guardar el mejor checkpoint con `mlflow.pytorch.log_model` y registrarlo en el **Model Registry** con el alias **`champion`**.
- Las clases del modelo deben estar en un módulo Python importable.
- Verificar que el modelo se pueda cargar desde un proceso distinto al notebook.

### 6. API (FastAPI)

| Endpoint | Función |
|----------|---------|
| `POST /generate` | Recibe `seed` (32×4, valores 0–127), `steps` (1–64) y `temperature` opcional (0.5–1.5, por defecto 1.0). Devuelve `generated` (`steps`×4), `model` y `version`. |
| `GET /health` | Devuelve `{"status": "ok", "model": ..., "version": ...}` |

- Cargar `champion` al iniciar el servicio, no en cada petición.
- Validar las entradas antes de llamar al modelo.
- Probar tres casos: una petición válida, `steps=33` y una entrada inválida.
- Desplegar en AWS para la sustentación, sin exponer credenciales.

## Hoja de ruta

### Fase 0 — Preparación del entorno
- [ ] Verificar el acceso al entorno de entrenamiento (GPU), a MLflow Tracking y al Model Registry
- [ ] Crear la estructura del repo, `requirements.txt` y `.env.example`
- [ ] Acordar con el docente cómo accederá a MLflow y a la API

### Fase 1 — Datos
- [ ] Descargar y descomprimir `jsb_chorales`
- [ ] Verificar columnas, número de corales, longitudes y rangos por voz
- [ ] Implementar las ventanas 32→16 con stride 8/16 y los DataLoaders
- [ ] Comprobar que no haya fugas entre corales ni entre particiones

### Fase 2 — Modelos y entrenamiento
- [ ] Encoder/decoder LSTM y GRU con embeddings concatenados
- [ ] Decoder GRU con atención aditiva
- [ ] Loop de entrenamiento con teacher forcing, early stopping y semilla 42
- [ ] Métricas: CE, exactitud por nota y exactitud por paso
- [ ] Integración con MLflow (parámetros, métricas por época, artefactos)

### Fase 3 — Experimentos
- [ ] Entrenar E1 (LSTM)
- [ ] Entrenar E2 (GRU)
- [ ] Entrenar E3 (GRU + atención)
- [ ] Elegir el hiperparámetro de E4 y entrenarlo
- [ ] Recargar los mejores checkpoints y armar la tabla comparativa
- [ ] Elegir el modelo por pérdida de validación y evaluarlo en `test`

### Fase 4 — Generación y análisis
- [ ] Generación por bloques con temperatura
- [ ] Exportar 2 MIDI (contexto + continuación)
- [ ] Hacer el mapa de atención de E3
- [ ] Analizar repeticiones, variedad y silencios

### Fase 5 — Registro y API
- [ ] `log_model` + Model Registry + alias `champion`
- [ ] Verificar la carga desde un proceso externo
- [ ] FastAPI: `/generate` y `/health` con validación Pydantic
- [ ] Pruebas: petición válida, `steps=33` y entrada inválida
- [ ] Despliegue en AWS

### Fase 6 — Informe y entrega
- [ ] Completar este README como informe:
  - tabla E1–E4;
  - justificaciones y resultados de test;
  - análisis de generación y de atención;
  - pruebas de la API;
  - herramientas de IA usadas
- [ ] Checklist final de verificación
- [ ] Entregar los enlaces al repo, al experimento de MLflow y a la API

## Estructura prevista

```
Bach/
├── src/bach/
│   ├── data.py        # carga CSV, ventanas, DataLoaders
│   ├── models.py      # LSTM, GRU, GRU + atención
│   ├── train.py       # entrenamiento + MLflow
│   ├── metrics.py     # pérdida y exactitudes
│   ├── generate.py    # generación por bloques, MIDI, mapa de atención
│   └── register.py    # registro en Model Registry
├── api/main.py        # FastAPI
├── notebooks/
├── outputs/           # MIDI, mapa de atención, curvas
├── requirements.txt
└── README.md
```

## Entregables

| Entregable | Contenido |
|------------|-----------|
| Repositorio | Código de datos, modelos, entrenamiento, generación y API; dependencias con versiones; instrucciones de ejecución |
| README | Tabla E1–E4, justificación de E4 y del modelo elegido, test, análisis de generación y atención, pruebas de la API, uso de IA |
| Evidencias musicales | 2 MIDI, corales usados, semilla, temperatura, tempo y mapa de atención de E3 |
| MLflow y API | Runs con métricas y artefactos, modelo con alias `champion` y URL de la API |

## Evaluación

| Componente | Valor |
|------------|-------|
| Modelos secuenciales (E1–E3, 4 voces, 32→16) | 1.5 |
| Experimentación y MLflow | 1.5 |
| Registro, API y generación | 1.0 |
| Sustentación | 1.0 |
| **Total** | **5.0** |

## Referencias

- Géron, A. *JSB Chorales Dataset* — https://github.com/ageron/handson-ml2/tree/master/datasets/jsb_chorales
- Boulanger-Lewandowski et al. (2012) — https://arxiv.org/abs/1206.6392
- MLflow Model Registry — https://mlflow.org/docs/latest/ml/model-registry/workflow/
- FastAPI — https://fastapi.tiangolo.com/tutorial/body/
- PyTorch seq2seq con atención — https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- pretty_midi — https://craffel.github.io/pretty-midi/
