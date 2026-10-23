"""Automatische Prüfung jedes fertigen Songs: Länge, Übersteuerung, Löcher, Musikanteil, Lautheit.

Gemessen wird am fertigen WAV, so wie es in der Bibliothek liegt. Die Schwellen stammen aus einer
Messreihe mit ACE-Step 1.5 (MobileMomente-Camper-Radio, Oktober 2026):
- Die Länge weicht bei einem gesunden Song höchstens Bruchteile einer Sekunde ab. Mehr als 1,5 s heißt,
  das Modell hat abgebrochen oder die Vorgabe nicht bekommen.
- Ein Loch ist eine Strecke unter −45 dBFS zwischen dem ersten und dem letzten lauten Fenster. Ein
  Break in Techno oder ein Atemholen vor dem Refrain dauert selten über 1,5 s; über 4 s ist es ein Aussetzer.
  Die Stille hinter dem letzten Ton zählt nicht als Loch.
- Musikanteil unter 60 % heißt: Der Song ist großteils leer oder fast unhörbar.
- Abgeschnittene Samples (|x| ≥ 0,999) hört man als Knacken. Ein paar einzelne sind ein Hinweis, viele ein Fehler.
Die Lautheit (LUFS nach ITU-R BS.1770, mit Gating) ist nur eine Angabe, kein Urteil.

Ergebnis: {"urteil": "ok" | "hinweis" | "fehler", "befunde": [{"code", "stufe", …Werte}], "werte": {…}}
Die Befunde tragen Codes statt Sätzen, damit die Oberfläche sie in jeder Sprache zeigen kann.
"""
import io
import math
import wave

import numpy as np

from .klang import _convolve

FENSTER = 0.5            # Sekunden je Messfenster
LAUT_DB = -40.0          # ab hier zählt ein Fenster als Musik
LOCH_DB = -45.0          # darunter beginnt ein Loch
LOCH_HINWEIS = 1.5       # Sekunden
LOCH_FEHLER = 4.0
LAENGE_TOLERANZ = 1.5    # Sekunden
MUSIK_MIN = 0.60
CLIP_GRENZE = 0.999
CLIP_FEHLER = 50         # abgeschnittene Samples, ab denen es ein Fehler ist


def _lesen(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        ch, width, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        roh = w.readframes(n)
    if width == 2:
        x = np.frombuffer(roh, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        x = np.frombuffer(roh, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"WAV mit {8 * width} Bit wird nicht geprüft")
    return x.reshape(-1, ch), fs


def _impulsantwort(stufen: list[tuple[list[float], list[float]]], laenge: int = 8192) -> np.ndarray:
    """Impulsantwort einer Biquad-Kette. Die K-Gewichtung klingt nach wenigen Millisekunden aus;
    8192 Samples sind bei 48 kHz das Hundertfache der längsten Zeitkonstante (Hochpass 38 Hz)."""
    h = np.zeros(laenge)
    h[0] = 1.0
    for b, a in stufen:
        y = np.zeros(laenge)
        x1 = x2 = y1 = y2 = 0.0
        for i in range(laenge):
            x0 = h[i]
            y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            y[i] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
        h = y
    return h


def _k_filter(fs: int) -> tuple[tuple[list, list], tuple[list, list]]:
    """K-Gewichtung nach BS.1770 (Hochregal + RLB-Hochpass), Koeffizienten für beliebige Abtastrate."""
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    a0 = 1 + k / q + k * k
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return (b1, a1), (b2, a2)


def lautheit(x: np.ndarray, fs: int) -> float | None:
    """Integrierte Lautheit in LUFS (BS.1770-4: 400-ms-Blöcke, 75 % Überlappung, absolutes und relatives Gating)."""
    h = _impulsantwort(list(_k_filter(fs)))
    y = np.stack([_convolve(x[:, c], h) for c in range(x.shape[1])], axis=1)   # FFT-Faltung wie beim Höhenausgleich
    block, schritt = int(0.4 * fs), int(0.1 * fs)
    if len(y) < block:
        return None
    quad = np.cumsum(np.concatenate([np.zeros((1, y.shape[1])), y * y]), axis=0)
    starts = np.arange(0, len(y) - block + 1, schritt)
    z = (quad[starts + block] - quad[starts]) / block           # mittleres Quadrat je Block und Kanal
    l = -0.691 + 10 * np.log10(np.maximum(z.sum(axis=1), 1e-12))
    l_abs = l[l > -70]
    if not len(l_abs):
        return None
    schwelle = -0.691 + 10 * np.log10(np.mean(10 ** ((l_abs + 0.691) / 10))) - 10
    l_rel = l_abs[l_abs > schwelle]
    return round(float(-0.691 + 10 * np.log10(np.mean(10 ** ((l_rel + 0.691) / 10)))), 1)


def pruefen(wav_bytes: bytes, soll_dauer: float = 0) -> dict:
    x, fs = _lesen(wav_bytes)
    dauer = len(x) / fs
    befunde = []

    if soll_dauer and soll_dauer > 0 and abs(dauer - soll_dauer) > LAENGE_TOLERANZ:
        befunde.append({"code": "length", "stufe": "fehler", "soll": round(soll_dauer, 1), "ist": round(dauer, 1)})

    clips = int(np.count_nonzero(np.abs(x) >= CLIP_GRENZE))
    if clips:
        befunde.append({"code": "clip", "stufe": "fehler" if clips >= CLIP_FEHLER else "hinweis", "anzahl": clips})

    mono = x.mean(axis=1)
    n = int(FENSTER * fs)
    k = len(mono) // n
    if k == 0:
        return {"urteil": "fehler", "befunde": [{"code": "empty", "stufe": "fehler"}], "werte": {"dauer": round(dauer, 2)}}
    rms = np.sqrt(np.mean(mono[: k * n].reshape(k, n) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    laut = np.flatnonzero(db > LAUT_DB)
    musik = len(laut) / k
    if musik < MUSIK_MIN:
        befunde.append({"code": "music", "stufe": "fehler", "anteil": round(musik * 100)})

    if len(laut):
        erstes, letztes = laut[0], laut[-1]
        lauf = 0
        for i in range(erstes, letztes + 2):
            if i <= letztes and db[i] < LOCH_DB:
                lauf += 1
                continue
            sek = lauf * FENSTER
            if sek >= LOCH_HINWEIS:
                befunde.append({"code": "hole", "stufe": "fehler" if sek >= LOCH_FEHLER else "hinweis",
                                "bei": round((i - lauf) * FENSTER, 1), "dauer": sek})
            lauf = 0
    else:
        befunde.append({"code": "empty", "stufe": "fehler"})

    werte = {"dauer": round(dauer, 2), "musik": round(musik * 100), "clips": clips,
             "spitze_db": round(float(20 * np.log10(max(np.abs(x).max(), 1e-9))), 1), "lufs": lautheit(x, fs)}
    stufen = {b["stufe"] for b in befunde}
    urteil = "fehler" if "fehler" in stufen else "hinweis" if stufen else "ok"
    return {"urteil": urteil, "befunde": befunde, "werte": werte}
