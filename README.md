<h1 align="center">RVM — cut the background out of a video, on your own PC</h1>

<p align="center">
  <strong>Give it a video, get back the person with the background gone.</strong><br/>
  No green screen. No upload. No subscription. No watermark. It runs on your computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/install-one%20command-purple?style=flat-square" alt="One command install"/>
  <img src="https://img.shields.io/badge/GPU-CUDA%20%2B%20CPU%20fallback-brightgreen?style=flat-square" alt="GPU"/>
  <img src="https://img.shields.io/badge/cloud-none-informational?style=flat-square" alt="No cloud"/>
</p>

```bash
uv tool install git+https://github.com/metiu1/rvm-web
rvm
```

<p align="center">
  <img src="docs/rvm-spot.gif" alt="RVM demo — background removed from a video, on your own PC" width="900"/>
</p>

<p align="center">
  <img src="docs/screenshot.png" alt="RVM local web UI — video background removal panel with model, output type and device selection" width="900"/>
</p>

---

## What it is

RVM is a small app you install with one command. It opens in your browser like a
website, but there is no website: everything runs on your own machine, and your
footage never leaves it.

You pick a video, press start, and it removes the background — the same trick a green
screen gives you, except you do not need the green screen. What comes out is **real
transparency** (an actual alpha channel), so you can drop the clip straight into
Premiere, DaVinci Resolve, After Effects or OBS and put whatever you like behind it.

You can also just cut out a photo, upscale a blurry video to 4×, or trim, crop, speed
up and compress a clip — the tools you keep opening a different website for, all in
one place, offline.

## Who it is for

- **Creators, streamers and editors** who want the "no green screen" cutout without
  paying a monthly fee per export.
- **Anyone whose footage cannot be uploaded** — client work, NDAs, faces of people who
  did not consent to a stranger's server, anything you would rather keep on your disk.
- **People with long or high-resolution video**, which is exactly what free online
  tools refuse to process.

## Why not just use an online tool

Web background removers cap the resolution, stamp a watermark on the result, put you
in a queue behind everyone else, charge by the minute, and keep a copy of your video
on their servers.

This one has no cap, no watermark, no queue, no account and no server. Your graphics
card does the work, your disk keeps the file. If you do not have a GPU it still runs,
just slower.

---

## Features

Sidebar navigation with seven tools:

- **Video background removal** ([RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)) — MobileNetV3 (fast) or ResNet50 (accurate); export as MP4 composition, alpha mask, or PNG sequence with real RGBA transparency
- **Image background removal** ([rembg](https://github.com/danielgatis/rembg) / U2Net) — upload a photo, get a transparent PNG, with before/after preview
- **Image upscaling** — 2× / 4×, Lanczos (no extra deps) or Real-ESRGAN (AI)
- **Video upscaling** — 2× / 4×, Lanczos or Real-ESRGAN, **original audio preserved** (muxed back via bundled ffmpeg)
- **Image editing** (Pillow) — crop, resize, rotate/flip, brightness/contrast/saturation/sharpness, filters (B&W, sepia, invert, blur, sharpen, auto-enhance), format convert + quality (PNG/JPG/WEBP), live before/after preview
- **Video editing** (bundled ffmpeg) — trim, crop, resize, change speed, compress, format convert, export GIF, extract audio, mute, reverse — with live progress
- **Vocal separation** ([Demucs](https://github.com/adefossez/demucs) / htdemucs) — split a song into vocals + instrumental, or 4 stems (vocals, drums, bass, other); reads any audio format or the audio track of a video, exports WAV or MP3, with in-page players

Plus:

- GPU accelerated (CUDA) with automatic CPU fallback
- Upload files directly from the browser (saved to `~/Downloads/rvm-uploads/`)
- `ffmpeg` ships bundled via `imageio-ffmpeg` — no system install required
- Installable in one command via `uv`

---

## Quick Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install git+https://github.com/metiu1/rvm-web
rvm
```

A browser window opens automatically at `http://localhost:7860`.
(The old `rvm-web` command still works as an alias.)

### CUDA (NVIDIA GPU) — recommended

On Windows, PyPI serves **CPU-only** torch by default. To get GPU
acceleration, tell uv to auto-detect your CUDA driver and pull matching
wheels:

```bash
uv tool install --reinstall --torch-backend=auto git+https://github.com/metiu1/rvm-web
```

Or set it once for every future install (persists across terminals):

```powershell
# PowerShell (Windows) — set once
[Environment]::SetEnvironmentVariable('UV_TORCH_BACKEND','auto','User')
```

```bash
# bash/zsh (Linux/macOS) — add to your shell profile
export UV_TORCH_BACKEND=auto
```

Verify CUDA is active — the upscaling log shows `Modello caricato su cuda:0`,
or check directly:

```bash
uv tool run --from rvm python -c "import torch; print(torch.cuda.is_available())"
```

> **Real-ESRGAN** AI upscaling is **built-in** — no extra packages needed.
> Model weights download automatically on first use to `~/.cache/rvm_web/`.
> `imageio-ffmpeg` (bundled ffmpeg, used to keep audio when upscaling video)
> is also installed automatically.

---

## Parameters

| Parameter | Options | Default |
|---|---|---|
| Model | mobilenetv3, resnet50 | mobilenetv3 |
| Output type | video, png_sequence | video |
| Downsample ratio | auto, 0.25, 0.5, 0.75, 1.0 | auto |
| Bitrate (Mbps) | 1, 2, 4, 8, 16 | 4 |
| Device | auto, cpu, cuda | auto |
| Seq chunk | 1, 2, 4, 8 | 1 |
| Workers | 0, 1, 2, 4 | 0 |

---

## Image Background Removal

The image section uses [rembg](https://github.com/danielgatis/rembg) with the U2Net model.  
On **first use**, the model (~170 MB) is downloaded automatically to `~/.u2net/`.  
Output is a PNG with full alpha transparency.

---

## Upscaling

Both image and video upscaling support 2× and 4× with two methods:

- **Lanczos** — classic high-quality resampling, no extra dependencies, fast.
- **Real-ESRGAN** — AI super-resolution, built-in (no extra packages). Model
  weights download automatically on first use to `~/.cache/rvm_web/`.

For **video** upscaling the original audio track is preserved: the video is
upscaled frame-by-frame, then the audio is muxed back with ffmpeg
(`imageio-ffmpeg`, bundled). If no audio track exists the video is written
directly.

---

## Known Fix — `av` Compatibility

RVM's `inference_utils.py` passes frame rate as a string to `av.add_stream()`.  
Newer `av` versions require a `Fraction`. The fix is applied automatically on first run.

---

## Credits

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) by Peter Lin et al.
- Web UI built with [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT

---

made by https://metiu.xyz/
