"""FastAPI server + CLI entry point for rvm."""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

from .processing import process_video, upscale_image, upscale_video
from .editing import edit_image, edit_video

app = FastAPI(title="rvm")

_mp_queue: Optional[mp.Queue] = None
_current_proc: Optional[mp.Process] = None
_job_running = threading.Event()


class ProcessRequest(BaseModel):
    model_name: str = "mobilenetv3"
    input_source: str = ""
    output_composition: str = ""
    output_alpha: str = ""
    output_foreground: str = ""
    output_type: str = "video"
    downsample_ratio: str = "none"
    output_video_mbps: float = 4.0
    seq_chunk: int = 1
    num_workers: int = 0
    device: str = "auto"


class ImageRequest(BaseModel):
    input_path: str
    output_path: str


class UpscaleImageRequest(BaseModel):
    input_path: str
    output_path: str
    method: str = "lanczos"
    scale: int = 2


class UpscaleVideoRequest(BaseModel):
    input_path: str
    output_path: str
    method: str = "lanczos"
    scale: int = 2
    crf: int = 16


class EditImageRequest(BaseModel):
    input_path: str
    output_path: str
    rotate: int = 0
    flip_h: bool = False
    flip_v: bool = False
    crop: Optional[list[int]] = None
    resize_w: int = 0
    resize_h: int = 0
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    filter: str = "none"
    fmt: str = ""
    quality: int = 92


class SeparateRequest(BaseModel):
    input_path: str
    output_dir: str = ""
    mode: str = "2"
    model: str = "htdemucs"
    fmt: str = "wav"
    device: str = "auto"


class EditVideoRequest(BaseModel):
    input_path: str
    output_path: str
    operation: str = "convert"
    start: float = 0.0
    end: Optional[float] = None
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    factor: float = 1.0
    fps: int = 12
    crf: int = 26


def _worker(params: dict, q: "mp.Queue[str]") -> None:
    from rvm.processing import process_video
    try:
        process_video(params, progress_cb=lambda msg: q.put(msg))
    except Exception as exc:
        q.put(f"ERROR: {exc}")
    finally:
        q.put("__done__")


def _upscale_video_worker(params: dict, q: "mp.Queue[str]") -> None:
    from rvm.processing import upscale_video
    try:
        upscale_video(params, progress_cb=lambda msg: q.put(msg))
    except Exception as exc:
        q.put(f"ERROR: {exc}")
    finally:
        q.put("__done__")


def _edit_video_worker(params: dict, q: "mp.Queue[str]") -> None:
    from rvm.editing import edit_video
    try:
        edit_video(params, progress_cb=lambda msg: q.put(msg))
    except Exception as exc:
        q.put(f"ERROR: {exc}")
    finally:
        q.put("__done__")


def _separate_worker(params: dict, q: "mp.Queue[str]") -> None:
    from rvm.separation import separate_audio
    try:
        separate_audio(params, progress_cb=lambda msg: q.put(msg))
    except Exception as exc:
        q.put(f"ERROR: {exc}")
    finally:
        q.put("__done__")


def _monitor_proc(proc: mp.Process) -> None:
    def _run() -> None:
        proc.join()
        _job_running.clear()
    threading.Thread(target=_run, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    html = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename).name if file.filename else "input.mp4"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".mp4"

    upload_dir = Path.home() / "Downloads" / "rvm-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / original_name
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    dest.write_bytes(content)

    output_path = str(upload_dir / f"{dest.stem}_output{suffix}")
    return {"ok": True, "path": str(dest), "output_path": output_path}


@app.post("/api/process")
async def api_process(req: ProcessRequest) -> dict:
    global _current_proc, _mp_queue
    if _job_running.is_set():
        return {"ok": False, "error": "A job is already running."}

    params = req.model_dump()
    params["downsample_ratio"] = None if params["downsample_ratio"] == "none" else float(params["downsample_ratio"])

    _mp_queue = mp.Queue()
    _job_running.set()
    proc = mp.Process(target=_worker, args=(params, _mp_queue), daemon=True)
    proc.start()
    _current_proc = proc
    _monitor_proc(proc)
    return {"ok": True}


@app.post("/api/cancel")
async def api_cancel() -> dict:
    global _current_proc
    if not _job_running.is_set():
        return {"ok": False, "error": "No job running."}
    proc = _current_proc
    if proc and proc.is_alive():
        proc.terminate()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: proc.join(timeout=5))
    _job_running.clear()
    if _mp_queue is not None:
        _mp_queue.put("__cancelled__")
    return {"ok": True}


@app.get("/api/events")
async def api_events() -> StreamingResponse:
    q = _mp_queue

    async def generator():
        if q is None:
            yield f"data: {json.dumps({'msg': '__done__'})}\n\n"
            return
        loop = asyncio.get_running_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=0.3))
                yield f"data: {json.dumps({'msg': msg})}\n\n"
                if msg in ("__done__", "__cancelled__"):
                    break
            except Exception:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/status")
async def api_status() -> dict:
    return {"running": _job_running.is_set()}


@app.post("/api/upload-image")
async def api_upload_image(file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename).name if file.filename else "input.png"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".png"

    upload_dir = Path.home() / "Downloads" / "rvm-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / original_name
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    dest.write_bytes(content)

    output_path = str(upload_dir / f"{dest.stem}_nobg.png")
    return {"ok": True, "path": str(dest), "output_path": output_path}


def _remove_bg(input_path: Path, output_path: Path) -> None:
    from rembg import remove

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = remove(input_path.read_bytes())
    output_path.write_bytes(result)


@app.post("/api/process-image")
async def api_process_image(req: ImageRequest) -> dict:
    input_path = Path(req.input_path)
    output_path = Path(req.output_path)

    if not input_path.exists():
        return {"ok": False, "error": f"File non trovato: {input_path}"}

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _remove_bg, input_path, output_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "output_path": str(output_path)}


@app.get("/api/preview-image")
async def api_preview_image(path: str) -> FileResponse:
    from fastapi import HTTPException

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "File not found")
    suffix = p.suffix.lower()
    media = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
    return FileResponse(str(p), media_type=media)


@app.post("/api/upload-upscale")
async def api_upload_upscale(file: UploadFile = File(...)) -> dict:
    original_name = Path(file.filename).name if file.filename else "input.mp4"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".mp4"

    upload_dir = Path.home() / "Downloads" / "rvm-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / original_name
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    dest.write_bytes(content)

    output_path = str(upload_dir / f"{dest.stem}_upscaled{suffix}")
    return {"ok": True, "path": str(dest), "output_path": output_path}


@app.post("/api/upscale-image")
async def api_upscale_image(req: UpscaleImageRequest) -> dict:
    input_path = Path(req.input_path)
    output_path = Path(req.output_path)

    if not input_path.exists():
        return {"ok": False, "error": f"File non trovato: {input_path}"}
    if req.scale not in (2, 4):
        return {"ok": False, "error": "Scale deve essere 2 o 4"}

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: upscale_image(input_path, output_path, req.method, req.scale),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "output_path": str(output_path)}


@app.get("/api/check-realesrgan")
async def api_check_realesrgan() -> dict:
    return {"ok": True}


@app.post("/api/upscale-video")
async def api_upscale_video(req: UpscaleVideoRequest) -> dict:
    global _current_proc, _mp_queue
    if _job_running.is_set():
        return {"ok": False, "error": "Un job è già in esecuzione."}

    params = {
        "input_path": req.input_path,
        "output_path": req.output_path,
        "method": req.method,
        "scale": req.scale,
        "crf": req.crf,
    }

    _mp_queue = mp.Queue()
    _job_running.set()
    proc = mp.Process(target=_upscale_video_worker, args=(params, _mp_queue), daemon=True)
    proc.start()
    _current_proc = proc
    _monitor_proc(proc)
    return {"ok": True}


@app.post("/api/edit-image")
async def api_edit_image(req: EditImageRequest) -> dict:
    params = req.model_dump()
    if not Path(req.input_path).exists():
        return {"ok": False, "error": f"File non trovato: {req.input_path}"}
    try:
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: edit_image(params))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "output_path": out}


@app.post("/api/edit-video")
async def api_edit_video(req: EditVideoRequest) -> dict:
    global _current_proc, _mp_queue
    if _job_running.is_set():
        return {"ok": False, "error": "Un job è già in esecuzione."}
    if not Path(req.input_path).exists():
        return {"ok": False, "error": f"File non trovato: {req.input_path}"}

    params = req.model_dump()
    _mp_queue = mp.Queue()
    _job_running.set()
    proc = mp.Process(target=_edit_video_worker, args=(params, _mp_queue), daemon=True)
    proc.start()
    _current_proc = proc
    _monitor_proc(proc)
    return {"ok": True}


@app.post("/api/separate")
async def api_separate(req: SeparateRequest) -> dict:
    global _current_proc, _mp_queue
    if _job_running.is_set():
        return {"ok": False, "error": "Un job è già in esecuzione."}
    if not Path(req.input_path).exists():
        return {"ok": False, "error": f"File non trovato: {req.input_path}"}

    params = req.model_dump()
    _mp_queue = mp.Queue()
    _job_running.set()
    proc = mp.Process(target=_separate_worker, args=(params, _mp_queue), daemon=True)
    proc.start()
    _current_proc = proc
    _monitor_proc(proc)
    return {"ok": True}


@app.get("/api/media")
async def api_media(path: str) -> FileResponse:
    """Serve a local media file (used by the audio players)."""
    import mimetypes
    from fastapi import HTTPException

    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "File not found")
    media = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return FileResponse(str(p), media_type=media)


def main() -> None:
    mp.freeze_support()

    def open_browser() -> None:
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
