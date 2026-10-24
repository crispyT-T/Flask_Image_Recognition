# Hand Sign Digit Recognition — with CI

A Flask web application that predicts a digit from a photograph of a hand sign, using a
Keras model trained by transfer learning.

Forked from [Sachin-crypto/Flask_Image_Recognition](https://github.com/Sachin-crypto/Flask_Image_Recognition)
and brought under continuous integration: Pylint, pre-commit, pytest and GitHub Actions.

[![CI](https://github.com/crispyT-T/Flask_Image_Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/crispyT-T/Flask_Image_Recognition/actions/workflows/ci.yml)

## Running it

Requires **Python 3.11**. The pinned dependencies do not support newer versions, and
TensorFlow 2.15 is the last release that bundles Keras 2, which the saved `.h5` model
needs.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:9000>, choose an image from `test_images/` and submit.

## Developing

```bash
pip install -r requirements-dev.txt
pre-commit install          # run the checks before each commit
pytest                      # 19 tests, under a second
pytest -m "not slow"        # skip the one test that loads the real model
pylint app.py model.py conftest.py tests/
```

The same checks run in CI on every push and pull request to `main`.

## Layout

| Path | What it is |
| --- | --- |
| `app.py` | Flask application: the upload form and the prediction route |
| `model.py` | Model loading, image preprocessing, inference |
| `digit_model.h5` | The trained Keras model, 28 MB |
| `templates/`, `static/` | Jinja2 templates, CSS and the client-side image preview |
| `tests/`, `conftest.py` | The test suite and its fixtures |
| `test_images/` | 100 sample images, ten per digit |
| `docs/` | Setup log, architecture notes, quality notes, diagrams, screenshots |

## Documentation

- [Setup log](docs/setup-log.md) — the problems found getting the legacy project to run
- [Architecture](docs/architecture.md) — structure, request flow, diagram
- [Code quality and tests](docs/quality-and-tests.md) — Pylint, pre-commit, pytest
- [Assignment report](docs/REPORT.md) — the full write-up
