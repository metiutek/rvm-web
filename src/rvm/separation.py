"""Vocal / instrument separation with Demucs (Meta, Hybrid Transformer Demucs).

Audio is decoded with the bundled ffmpeg, so any format ffmpeg reads works —
mp3, wav, flac, m4a, ogg, and even the audio track of a video file.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .editing import _find_ffmpeg

SAMPLE_RATE = 44100

# Italian file-name suffix for each Demucs stem.
STEM_NAMES = {
    "vocals": "voce",
    "no_vocals": "base",
    "drums": "batteria",
    "bass": "basso",
    "other": "altro",
}


def _decode(path: Path, ffmpeg: str):
    """Decode any audio/video file to a float32 stereo tensor [2, T] at 44.1 kHz."""
    import numpy as np
    import torch

    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-f", "f32le", "-ac", "2", "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError("Impossibile leggere l'audio: " + (err[-1] if err else "nessuna traccia audio"))
    wav = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
    return torch.from_numpy(wav)


def separate_audio(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> list[str]:
    """Split a song into stems.

    params keys:
      input_path : str  (required)
      output_dir : str  folder for the stems (default: next to the input)
      mode       : "2" → voce + base, "4" → voce, batteria, basso, altro
      model      : "htdemucs" (fast) | "htdemucs_ft" (fine-tuned, ~4× slower, cleaner)
      fmt        : "wav" | "mp3"
      device     : "auto" | "cuda" | "cpu"
    Returns the list of written files.
    """
    import torch
    from demucs.api import Separator
    from demucs.audio import save_audio

    log = progress_cb or (lambda _m: None)

    inp = Path(params["input_path"])
    if not inp.exists():
        raise FileNotFoundError(f"File non trovato: {inp}")
    out_dir = Path(params.get("output_dir") or inp.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(params.get("mode", "2"))
    model = params.get("model", "htdemucs")
    fmt = params.get("fmt", "wav")
    if fmt not in ("wav", "mp3"):
        raise ValueError(f"Formato non supportato: {fmt}")

    device = params.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile su questo PC: scegli CPU o Automatico.")

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg non trovato.")

    log(f"Lettura audio: {inp.name}")
    wav = _decode(inp, ffmpeg)
    log(f"Durata: {wav.shape[1] / SAMPLE_RATE:.0f} s · dispositivo: {device.upper()}")

    log(f"Caricamento modello {model} (al primo uso viene scaricato)…")
    last_pct = [-1]

    def on_chunk(info: dict) -> None:
        if info["state"] != "end":
            return
        # A "bag" model runs several sub-models in sequence; fold them into one 0-100 scale.
        done = (info["model_idx_in_bag"] + (info["segment_offset"] + 1) / info["audio_length"]) / info["models"]
        pct = min(99, int(done * 100))
        if pct > last_pct[0]:
            last_pct[0] = pct
            log(f"PROGRESS {pct}")

    sep = Separator(model=model, device=device, callback=on_chunk)
    log("Separazione in corso…")
    _, stems = sep.separate_tensor(wav, SAMPLE_RATE)

    if mode == "2":
        vocals = stems["vocals"]
        backing = sum(t for name, t in stems.items() if name != "vocals")
        stems = {"vocals": vocals, "no_vocals": backing}

    written = []
    for name, tensor in stems.items():
        label = STEM_NAMES.get(name, name)
        dest = out_dir / f"{inp.stem}_{label}.{fmt}"
        save_audio(tensor.cpu(), dest, samplerate=sep.samplerate)
        written.append({"name": label, "path": str(dest)})
        log(f"Salvato: {dest.name}")

    # The UI picks this line up to show a player for each stem.
    log("RESULT " + json.dumps(written))
    return [w["path"] for w in written]
