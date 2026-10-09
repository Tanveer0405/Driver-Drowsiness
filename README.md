# Edge AI Driver Drowsiness Monitor

A real-time driver drowsiness detector that runs **on the edge**. A laptop webcam streams video to a **Raspberry Pi 4**, which finds the face and eyes, classifies each eye as open or closed with a lightweight **TensorFlow Lite** model, and analyses behaviour over time. When it detects drowsiness, the laptop plays an alarm.

> **Prototype for learning and research. It is not a certified safety system. Never test drowsiness while actually driving.**

## How it works

```
 Laptop                                   Raspberry Pi 4
 ┌──────────────┐   JPEG frames (ZeroMQ)  ┌──────────────────────────────────┐
 │ Webcam       │ ──────────────────────▶ │ 1. MediaPipe face landmarks      │
 │ Preview      │                         │ 2. Eye crops (96x96 grayscale)   │
 │ Speaker alarm│ ◀────────────────────── │ 3. TFLite eye-state CNN          │
 └──────────────┘   JSON {level, reasons} │ 4. Fuse with EAR                 │
                                          │ 5. Temporal drowsiness analysis  │
                                          └──────────────────────────────────┘
```

1. **Face and eye detection:** MediaPipe Face Landmarker returns 478 landmarks (including the irises).
2. **Eye-state model:** both eyes are cropped to 96x96 grayscale and fed to a MobileNetV2 (alpha 0.5) classifier that outputs P(closed). It is combined with the Eye Aspect Ratio (EAR) for robustness.
3. **Temporal analysis:** a per-driver calibration (first 45 s) sets a personal baseline, then rolling time windows look for drowsy behaviour.

### Detection rules

| Signal | Trigger | Level |
|---|---|---|
| Eyes closed | 1.5 s or more continuously | ALARM |
| PERCLOS | 30% or more of frames closed over 30 s | ALARM |
| No face visible | more than 3 s | ALARM |
| Fixed stare | gaze spread under 30% of baseline and no blink for 10 s | WARNING |
| Low blink rate + drooping lids | blink rate below baseline and EAR below 80% of baseline | WARNING |
| Head nod | head pitch drifts from baseline | WARNING |
| Two warnings together | | ALARM |

All thresholds live in [`common/config.py`](common/config.py).

## Repository structure

```
common/         config.py (all thresholds), types.py (shared dataclasses)
training/       dataset, model builder, trainer, TFLite exporter (local training scripts)
notebooks/      drowsiness_eye_state_colab.ipynb  <- recommended way to train (Google Colab GPU)
pi_node/        face_eye_extractor, eye_classifier, drowsiness_analyzer, rolling, server, main
laptop_node/    webcam, pi_link, alarm, overlay, main
local_demo.py   run the whole pipeline on one machine for testing
models/         eye_state.tflite goes here
```

## Quick start

### 1. Train the model (Google Colab)

1. Open `notebooks/drowsiness_eye_state_colab.ipynb` in Google Colab.
2. `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all` (about 15-25 minutes).
3. Download `eye_state.tflite` and `metrics.json`, and put the model in `models/`.

The notebook uses a **subject-wise split** (train, validation and test contain different people), heavy augmentation, two-stage transfer learning and a TFLite export that is re-tested on the test set.

### 2. Test on your computer

```bash
pip install -r requirements_pi.txt
python local_demo.py --model models/eye_state.tflite
```

The first run downloads the MediaPipe face landmark model to `models/face_landmarker.task`.

### 3. Run on the Raspberry Pi 4

Use **64-bit Raspberry Pi OS** and **Python 3.11 or 3.12** (see the Pi 4 notes below).

```bash
# on the Pi, inside the project folder
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_pi.txt
python -m pi_node.main --model models/eye_state.tflite
# prints: Listening on port 5555
```

Find the Pi's address with `hostname -I`.

### 4. Run the laptop client

```bash
pip install -r requirements_laptop.txt
python -m web_node.app --host sct2026.local
```

Keys: `q` quit, `r` recalibrate. For the first 45 s, look ahead and blink normally while it calibrates. Both devices must be on the **same network**. A raw IPv6 address also works as `--host`.

## Pi 4 notes and troubleshooting

| Problem | Cause and fix |
|---|---|
| `Illegal instruction` / "compiled with aes enabled" | Recent MediaPipe builds need CPU features the Pi 4 lacks. Use `mediapipe==0.10.18` on Python 3.11/3.12. If the OS ships Python 3.13, install 3.12 with [uv](https://github.com/astral-sh/uv): `uv venv --python 3.12 ~/venv312`. |
| `module 'mediapipe' has no attribute 'solutions'` | Newer MediaPipe removed the old API. This project already uses the Tasks `FaceLandmarker` API. |
| `Pi not responding` in the preview | Wrong IP, server not running, or the devices are on different networks. Check `hostname -I` on the Pi and test with `nc -vz <ip> 5555`. |
| Mac shows `192.0.0.2` and cannot reach the Pi | Phone hotspots can give a device an IPv6-only setup. The code supports IPv6, so use the Pi's `2409:...` address with `--host`, or use a normal Wi-Fi router. |
| `PortAudio library not found` | `sudo apt install libportaudio2` |
| Slow frame rate | Lower `frame_width` / `frame_height` in `common/config.py`, or train with `alpha=0.35`. |

## Results

Fill in from the `metrics.json` produced by the Colab notebook (test set = people never seen in training):

| Metric | Value |
|---|---|
| Test accuracy | 0.956247500034634 |
| Recall (closed eyes) | 0.9951187042378522 |
| TFLite test accuracy | 0.646 |

## Limitations and future work

- The MRL training data is infrared, while a normal webcam is RGB. Adding your own webcam eye crops improves accuracy.
- The stare and head-nod rules are heuristics validated on simulated data, not real driving.
- Planned: near-infrared camera for night driving, yawning detection, an on-device camera and buzzer on the Pi, and validation on public drowsiness datasets (UTA-RLDD, NTHU-DDD).

## Tech stack

Python, TensorFlow / TensorFlow Lite (LiteRT), MediaPipe, OpenCV, ZeroMQ, pygame.
