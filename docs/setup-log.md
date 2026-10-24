# Step 1 — Setup log: problems found and how they were resolved

Environment used: macOS 15 (Apple Silicon, arm64), Python 3.11.17, virtualenv `.venv`.

## Problem 1 — The pinned dependencies cannot be installed

The legacy `requirements.txt` pinned:

```
Flask==2.2.1
keras==2.9.0
numpy==1.23.2
Pillow==9.3.0
tensorflow==2.10.0
```

`pip install -r requirements.txt` failed:

```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.10.0
       (from versions: 2.13.0rc0, ... 2.22.0rc0)
ERROR: No matching distribution found for tensorflow==2.10.0
```

**Cause.** TensorFlow 2.10 was released before TensorFlow published arm64 macOS wheels
under the `tensorflow` package name, so no distribution exists for this platform. The
oldest version available for Apple Silicon is 2.13. The project was originally developed
on Windows (see Problem 2), where the pinned versions do install.

**Resolution.** Moved to TensorFlow 2.15.1, the newest release that still bundles Keras 2.
This matters because the saved model is a Keras 2 `.h5` file: Keras 3, which ships with
TensorFlow 2.16 and later, no longer loads that format without the `tf-keras`
compatibility shim. NumPy was raised to 1.26.4, the last 1.x release, because TensorFlow
2.15 is not built against NumPy 2.x. `Werkzeug` was pinned below 3.0 because Flask 2.2
imports symbols that Werkzeug 3 removed.

Full failure output: [`evidence/01-original-requirements-failure.txt`](evidence/01-original-requirements-failure.txt)

## Problem 2 — The model path was hardcoded to the original author's machine

`model.py` loaded the model from an absolute Windows path:

```python
model = load_model("D:/SACHIN/Models/Hand-Sign-Digit-Language/digit_model.h5")
```

Importing `model` therefore failed on any other machine:

```
OSError: No file or directory found at D:/SACHIN/Models/Hand-Sign-Digit-Language/digit_model.h5
```

**Resolution.** The model file `digit_model.h5` is committed at the repository root, so the
path is now resolved relative to the module's own location. This works regardless of
operating system or the directory the app is started from:

```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")
model = load_model(MODEL_PATH)
```

## Problem 3 — Editor and cache files were committed

`.idea/` (PyCharm settings) and `__pycache__/` were tracked in the repository. They were
removed and a `.gitignore` added.

## Verification

With the environment above, the model loads and predicts correctly:

```
test_images/4/Sign 4 (92).jpeg -> shape (1, 224, 224, 3) pred 4
test_images/7/Sign 7 (54).jpeg -> shape (1, 224, 224, 3) pred 7
```

The application was then started with `python app.py` (port 9000) and exercised through a
browser. Screenshots are in [`screenshots/`](screenshots):

| File | Shows |
| --- | --- |
| `01-home-page.jpg` | The home page served at `http://127.0.0.1:9000/` |
| `02-image-selected.jpg` | A hand-sign image for the digit 4 selected and previewed |
| `03-prediction-result.jpg` | The result page returning the prediction `4` |

## How to run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py      # serves on http://127.0.0.1:9000
```
