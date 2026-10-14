"""RobustVideoMatting inference wrapper + upscaling utilities."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import av
import torch

_model_cache: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    key = f"{model_name}_{device}"
    with _lock:
        if key not in _model_cache:
            model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                model_name,
                pretrained=True,
                trust_repo=True,
            )
            _model_cache[key] = model.to(device).eval()
        return _model_cache[key]


# ---------------------------------------------------------------------------
# Video background removal — own reader/writer loop around the RVM model.
#
# RVM's bundled ``convert_video`` sets the stream width/height on every
# write, which PyAV >= 14 rejects ("Cannot change width after codec is
# open"), ignores the rotation flag of phone videos, drops the audio track
# and can only composite on green. This loop fixes all of that.
# ---------------------------------------------------------------------------

# container -> (codec, pix_fmt, supports alpha)
_VIDEO_FORMATS = {
    "mp4":  ("libx264", "yuv420p", False),
    "mov":  ("prores_ks", "yuva444p10le", True),
    "webm": ("libvpx-vp9", "yuva420p", True),
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = (value or "#00ff00").lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _frame_rotation(frame) -> int:
    """Number of 90° counter-clockwise turns needed to display the frame upright."""
    rot = getattr(frame, "rotation", 0) or 0
    return int(round(rot / 90.0)) % 4


class _VideoWriter:
    """PyAV writer that sets the encoder size once, from the first frame."""

    def __init__(self, path: str, fps, fmt: str, bit_rate: int):
        codec, pix_fmt, _ = _VIDEO_FORMATS[fmt]
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.pix_fmt = pix_fmt
        if codec == "prores_ks":
            self.stream.options = {"profile": "4"}  # ProRes 4444
        else:
            self.stream.bit_rate = bit_rate
        if codec == "libvpx-vp9":
            self.stream.options = {"auto-alt-ref": "0"}  # required for VP9 alpha
        # 4:2:0 subsampling needs even dimensions
        self.even = "420" in pix_fmt
        self.size: Optional[tuple[int, int]] = None

    def write(self, rgb: "np.ndarray") -> None:
        """rgb: HxWx3 or HxWx4 uint8."""
        import numpy as np

        h, w = rgb.shape[:2]
        if self.size is None:
            ew, eh = (w + (w & 1), h + (h & 1)) if self.even else (w, h)
            self.stream.width, self.stream.height = ew, eh
            self.size = (ew, eh)
        ew, eh = self.size
        if (w, h) != (ew, eh):
            rgb = np.pad(rgb, ((0, eh - h), (0, ew - w), (0, 0)), mode="edge")
        fmt = "rgba" if rgb.shape[2] == 4 else "rgb24"
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format=fmt)
        for pkt in self.stream.encode(frame):
            self.container.mux(pkt)

    def close(self) -> None:
        try:
            if self.size is not None:
                for pkt in self.stream.encode():
                    self.container.mux(pkt)
        finally:
            self.container.close()


class _PngWriter:
    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def write(self, rgb: "np.ndarray") -> None:
        from PIL import Image

        Image.fromarray(rgb).save(self.dir / f"{self.n:05d}.png")
        self.n += 1

    def close(self) -> None:
        pass


def _make_background(params: dict, h: int, w: int, device, dtype):
    """Return a callable(src[T,3,H,W]) -> background tensor, or None for transparent."""
    import numpy as np

    mode = params.get("background", "green")
    if mode == "transparent":
        return None
    if mode == "blur":
        import torchvision.transforms.functional as TF

        k = max(3, (min(h, w) // 25) | 1)  # odd kernel, scales with resolution
        return lambda src: TF.gaussian_blur(src, [k, k], [k / 3.0, k / 3.0])
    if mode == "image" and params.get("background_image"):
        from PIL import Image

        img = Image.open(params["background_image"]).convert("RGB")
        # cover-fit: scale to fill, then centre-crop
        s = max(w / img.width, h / img.height)
        img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
        left, top = (img.width - w) // 2, (img.height - h) // 2
        img = img.crop((left, top, left + w, top + h))
        t = torch.from_numpy(np.asarray(img).copy()).to(device, dtype).div(255).permute(2, 0, 1).unsqueeze(0)
        return lambda src: t
    color = (120, 255, 155) if mode == "green" else _hex_to_rgb(params.get("background_color", "#00ff00"))
    t = torch.tensor(color, device=device, dtype=dtype).div(255).view(1, 3, 1, 1)
    return lambda src: t


def _to_uint8(t: "torch.Tensor") -> "np.ndarray":
    """[T,C,H,W] float 0-1 -> [T,H,W,C] uint8."""
    return t.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()


def process_video(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Run RVM background removal.

    params keys:
      model_name        : "mobilenetv3" | "resnet50"
      input_source      : str path
      output_composition: str path or None
      output_alpha      : str path or None
      output_foreground : str path or None
      output_type       : "video" | "png_sequence"
      output_format     : "mp4" | "mov" | "webm"   (video only)
      background        : "green" | "color" | "blur" | "image" | "transparent"
      background_color  : "#rrggbb"
      background_image  : str path
      keep_audio        : bool
      downsample_ratio  : float or None
      output_video_mbps : float
      seq_chunk         : int
      device            : "auto" | "cpu" | "cuda"
    """
    import os
    import numpy as np

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    device_str = params.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    output_composition = params.get("output_composition") or None
    output_alpha       = params.get("output_alpha") or None
    output_foreground  = params.get("output_foreground") or None
    if not any([output_composition, output_alpha, output_foreground]):
        raise ValueError("At least one output path must be specified.")

    output_type = params.get("output_type", "video")
    fmt = params.get("output_format", "mp4")
    if fmt not in _VIDEO_FORMATS:
        raise ValueError(f"Unknown output format: {fmt}")
    background = params.get("background", "green")
    if background == "transparent" and output_type == "video" and not _VIDEO_FORMATS[fmt][2]:
        raise ValueError("MP4 has no alpha channel — pick MOV (ProRes 4444), WebM (VP9) "
                         "or PNG sequence for a transparent background.")

    log(f"Loading model '{params['model_name']}' on {device}…")
    model = load_model(params["model_name"], device)
    dtype = next(model.parameters()).dtype

    downsample = params.get("downsample_ratio")
    downsample = float(downsample) if downsample is not None else None
    seq_chunk = max(1, int(params.get("seq_chunk", 1)))
    bit_rate = int(float(params.get("output_video_mbps", 4)) * 1_000_000)
    input_source = params["input_source"]

    in_c = av.open(input_source)
    in_v = in_c.streams.video[0]
    in_v.thread_type = "AUTO"
    fps = in_v.average_rate or 30
    total = in_v.frames or 0
    if not total and in_v.duration and in_v.time_base:
        total = int(float(in_v.duration * in_v.time_base) * float(fps))
    has_audio = len(in_c.streams.audio) > 0
    log(f"Input: {in_v.width}x{in_v.height}, {float(fps):.2f} fps, {total or '?'} frames")

    def open_writer(path: str):
        if output_type == "png_sequence":
            return _PngWriter(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return _VideoWriter(path, fps, fmt, bit_rate)

    # The composition goes to a temp file first when the audio still has to be muxed on.
    mux_audio = bool(params.get("keep_audio", True)) and has_audio and output_type == "video"
    com_target = output_composition
    if output_composition and mux_audio:
        p = Path(output_composition)
        com_target = str(p.with_name(p.stem + "__noaudio" + p.suffix))

    writers: dict = {}
    done = 0
    try:
        if output_composition:
            writers["com"] = open_writer(com_target)
        if output_alpha:
            writers["pha"] = open_writer(output_alpha)
        if output_foreground:
            writers["fgr"] = open_writer(output_foreground)

        rec = [None] * 4
        bg_fn = None
        bg_ready = False
        last_pct = -1
        batch: list = []

        def run_batch(frames: list) -> None:
            nonlocal rec, bg_fn, bg_ready, downsample
            src = torch.from_numpy(np.stack(frames)).to(device, non_blocking=True)
            src = src.permute(0, 3, 1, 2).to(dtype).div(255)  # [T,3,H,W]
            h, w = src.shape[2:]
            if downsample is None:
                downsample = min(512 / max(h, w), 1)
            if not bg_ready and "com" in writers:
                bg_fn = _make_background(params, h, w, device, dtype)
                bg_ready = True
            fgr, pha, *rec = model(src.unsqueeze(0), *rec, downsample)
            fgr, pha = fgr[0], pha[0]
            if "fgr" in writers:
                for f in _to_uint8(fgr):
                    writers["fgr"].write(f)
            if "pha" in writers:
                for f in _to_uint8(pha.repeat(1, 3, 1, 1)):
                    writers["pha"].write(f)
            if "com" in writers:
                rgba = bg_fn is None or (output_type == "png_sequence" and background == "green")
                if rgba:  # transparent (and the legacy PNG-sequence default) → RGBA
                    com = torch.cat([fgr * pha.gt(0), pha], dim=1)
                else:
                    com = fgr * pha + bg_fn(src) * (1 - pha)
                for f in _to_uint8(com):
                    writers["com"].write(f)

        with torch.no_grad():
            log("Processing video…")
            for frame in in_c.decode(in_v):
                arr = frame.to_ndarray(format="rgb24")
                k = _frame_rotation(frame)
                if k:
                    arr = np.ascontiguousarray(np.rot90(arr, k))
                batch.append(arr)
                if len(batch) < seq_chunk:
                    continue
                run_batch(batch)
                done += len(batch)
                batch = []
                if total:
                    pct = min(99, done * 100 // total)
                    if pct != last_pct:
                        last_pct = pct
                        log(f"PROGRESS {pct} {done}/{total}")
                elif done % 50 == 0:
                    log(f"Frame {done}…")
            if batch:
                run_batch(batch)
                done += len(batch)
    finally:
        in_c.close()
        for wr in writers.values():
            wr.close()

    log(f"{done} frames processed.")
    if output_composition and mux_audio:
        log("Adding the original audio…")
        if _mux_audio(com_target, input_source, output_composition, log):
            try:
                os.remove(com_target)
            except OSError:
                pass
        else:
            os.replace(com_target, output_composition)
    log("Done.")


# ---------------------------------------------------------------------------
# Upscaling — built-in Real-ESRGAN (RRDBNet), no external packages needed
# ---------------------------------------------------------------------------

import math
import torch.nn as nn
import torch.nn.functional as F


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.rdb3(self.rdb2(self.rdb1(x)))) * 0.2 + x


class _RRDBNet(nn.Module):
    """RRDBNet — same architecture as Real-ESRGAN x2plus / x4plus models."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23,
                 num_grow_ch: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch * 4 if scale == 2 else num_in_ch
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class _RealESRGANUpsampler:
    """Thin wrapper: downloads weights, runs tiled inference."""

    _MODELS = {
        2: ("RealESRGAN_x2plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
        4: ("RealESRGAN_x4plus.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    }

    def __init__(self, scale: int, tile: int = 512, tile_pad: int = 10):
        import urllib.request
        net_scale = 2 if scale <= 2 else 4
        fname, url = self._MODELS[net_scale]
        cache = Path.home() / ".cache" / "rvm_web" / "realesrgan"
        cache.mkdir(parents=True, exist_ok=True)
        weights = cache / fname
        if not weights.exists():
            urllib.request.urlretrieve(url, weights)

        model = _RRDBNet(scale=net_scale)
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("params_ema") or state.get("params") or state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.net_scale = net_scale
        self.tile = tile
        self.tile_pad = tile_pad

    def enhance(self, img_rgb: "numpy.ndarray") -> "numpy.ndarray":
        """HxWx3 uint8 RGB in → HxWx3 uint8 RGB out."""
        import numpy as np
        t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, h, w = t.shape
        with torch.no_grad():
            if self.tile == 0 or (h <= self.tile and w <= self.tile):
                out = self.model(t)
            else:
                out = self._tiled(t)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        return (out.cpu().numpy() * 255.0).round().astype(np.uint8)

    def _tiled(self, img: torch.Tensor) -> torch.Tensor:
        s = self.net_scale
        _, c, h, w = img.shape
        tile, pad = self.tile, self.tile_pad
        out = torch.zeros(1, c, h * s, w * s, dtype=img.dtype, device=img.device)
        for yi in range(math.ceil(h / tile)):
            for xi in range(math.ceil(w / tile)):
                x1 = max(xi * tile - pad, 0)
                x2 = min((xi + 1) * tile + pad, w)
                y1 = max(yi * tile - pad, 0)
                y2 = min((yi + 1) * tile + pad, h)
                patch = self.model(img[:, :, y1:y2, x1:x2])
                ox1, ox2 = xi * tile * s, min((xi + 1) * tile, w) * s
                oy1, oy2 = yi * tile * s, min((yi + 1) * tile, h) * s
                px1 = (xi * tile - x1) * s
                py1 = (yi * tile - y1) * s
                out[:, :, oy1:oy2, ox1:ox2] = patch[:, :, py1:py1 + (oy2 - oy1), px1:px1 + (ox2 - ox1)]
        return out


_upsampler_cache: dict[int, "_RealESRGANUpsampler"] = {}
_upsampler_lock = threading.Lock()


def _get_realesrgan_upsampler(scale: int) -> "_RealESRGANUpsampler":
    with _upsampler_lock:
        if scale not in _upsampler_cache:
            _upsampler_cache[scale] = _RealESRGANUpsampler(scale)
        return _upsampler_cache[scale]


def upscale_image(
    input_path: Path,
    output_path: Path,
    method: str = "lanczos",
    scale: int = 2,
) -> None:
    import numpy as np
    from PIL import Image  # type: ignore

    img = Image.open(str(input_path))
    w, h = img.size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "realesrgan":
        upsampler = _get_realesrgan_upsampler(scale)
        rgb = np.array(img.convert("RGB"))
        result = Image.fromarray(upsampler.enhance(rgb))
        result.save(str(output_path))
    else:
        img.resize((w * scale, h * scale), Image.LANCZOS).save(str(output_path))


def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable: system PATH first, then imageio-ffmpeg bundle."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        return get_ffmpeg_exe()
    except Exception:
        return None


def _mux_audio(video_path: str, audio_src: str, output_path: str,
               log: Callable[[str], None]) -> bool:
    """Mux audio from audio_src onto video_path → output_path. Returns True on success."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        log("WARN: ffmpeg non trovato (pip install imageio-ffmpeg) — output senza audio")
        return False

    import subprocess

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,        # 0: upscaled video (no audio)
        "-i", audio_src,         # 1: original (for audio)
        "-map", "0:v:0",
        "-map", "1:a:0?",        # optional audio — won't fail if source has none
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("Audio unito al video.")
        return True
    except Exception as exc:
        log(f"WARN: unione audio fallita ({exc}) — output senza audio")
        return False


def upscale_video(params: dict, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    import os
    import av  # type: ignore
    from PIL import Image  # type: ignore

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    input_path = str(params["input_path"])
    output_path = str(params["output_path"])
    method = params.get("method", "lanczos")
    scale = int(params.get("scale", 2))
    crf = int(params.get("crf", 16))

    upsampler = None
    if method == "realesrgan":
        log("Caricamento modello Real-ESRGAN…")
        upsampler = _get_realesrgan_upsampler(scale)
        log(f"Modello caricato su {upsampler.device}.")

    def process_frame(pil_frame: "Image.Image") -> "Image.Image":
        if upsampler is not None:
            import numpy as np
            return Image.fromarray(upsampler.enhance(np.array(pil_frame.convert("RGB"))))
        w, h = pil_frame.size
        return pil_frame.resize((w * scale, h * scale), Image.LANCZOS)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Upscale into a video-only temp file, then mux original audio back with ffmpeg.
    # This avoids the brittle, version-dependent PyAV audio stream-copy API.
    temp_video = str(out_dir / f"{Path(output_path).stem}__videotmp.mp4")

    has_audio = False
    with av.open(input_path) as in_c:
        in_v = in_c.streams.video[0]
        fps = in_v.average_rate
        total = in_v.frames or 0
        orig_w, orig_h = in_v.width, in_v.height
        has_audio = len(in_c.streams.audio) > 0
        log(f"Risoluzione: {orig_w}x{orig_h} → {orig_w * scale}x{orig_h * scale}, {float(fps):.2f} fps")

        with av.open(temp_video, "w") as out_c:
            # Create the encoder lazily from the first processed frame so the
            # stream dimensions always match the real output (robust against any
            # Real-ESRGAN rounding). Let the encoder generate timestamps from the
            # stream rate — copying the input pts/time_base produces invalid,
            # non-monotonic DTS and the muxer rejects packets (errno 22 EINVAL).
            out_v = None
            frame_idx = 0
            for frame in in_c.decode(video=0):
                upscaled = process_frame(frame.to_image())
                if out_v is None:
                    out_v = out_c.add_stream("libx264", rate=fps)
                    out_v.width, out_v.height = upscaled.size
                    out_v.pix_fmt = "yuv420p"
                    # Lower CRF = higher quality (default libx264 would
                    # re-compress heavily and throw away upscaled detail).
                    out_v.options = {"crf": str(crf), "preset": "slow"}
                out_frame = av.VideoFrame.from_image(upscaled)
                for pkt in out_v.encode(out_frame):
                    out_c.mux(pkt)
                frame_idx += 1
                if frame_idx == 1 or frame_idx % 25 == 0:
                    progress = f"{frame_idx}/{total}" if total else str(frame_idx)
                    log(f"Frame {progress} elaborati…")

            if out_v is not None:
                for pkt in out_v.encode():
                    out_c.mux(pkt)

    # Attach audio
    if has_audio:
        log("Unione audio…")
        if _mux_audio(temp_video, input_path, output_path, log):
            try:
                os.remove(temp_video)
            except OSError:
                pass
        else:
            os.replace(temp_video, output_path)
    else:
        os.replace(temp_video, output_path)

    log("Done.")
