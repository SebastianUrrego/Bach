"""Conversión de la matriz (pasos, 4) a MIDI con pretty_midi.

Convención del laboratorio: dentro de cada voz se fusionan las notas consecutivas iguales (la matriz no
distingue una nota sostenida de una repetida) y `0` es silencio. Cada paso es una semicorchea.
"""

import numpy as np
import pretty_midi

VOICE_NAMES = ["Soprano", "Alto", "Tenor", "Bajo"]
PROGRAM = 52  # Choir Aahs (General MIDI)


def step_seconds(tempo: float) -> float:
    """Duración de un paso (semicorchea) con el tempo en negras por minuto."""
    return 60.0 / (4 * tempo)


def to_midi(steps: np.ndarray, path, tempo: float = 80.0, marker_step: int | None = None,
            marker_text: str = "inicio de la continuacion") -> int:  # ASCII: los eventos de texto MIDI no son UTF-8
    """Escribe el MIDI y devuelve el número de notas (eventos) tras fusionar repeticiones."""
    dt = step_seconds(tempo)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    n_events = 0
    for v, name in enumerate(VOICE_NAMES):
        inst = pretty_midi.Instrument(program=PROGRAM, name=name)
        voice = steps[:, v]
        start = 0
        for t in range(1, len(voice) + 1):
            if t == len(voice) or voice[t] != voice[start]:
                if voice[start] != 0:
                    inst.notes.append(pretty_midi.Note(velocity=80, pitch=int(voice[start]),
                                                       start=start * dt, end=t * dt))
                    n_events += 1
                start = t
        pm.instruments.append(inst)
    if marker_step is not None:
        pm.text_events.append(pretty_midi.Text(marker_text, marker_step * dt))
    pm.write(str(path))
    return n_events
