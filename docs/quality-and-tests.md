# Steps 3 and 4 — Code quality and automated tests

## Pylint configuration

The configuration is in [`.pylintrc`](../.pylintrc) and is deliberately short. The
approach was to keep Pylint's defaults and fix what it reported, rather than disable
checks to raise the score. Three settings carry the whole config:

| Setting | Value | Why |
| --- | --- | --- |
| `ignored-modules` | `keras`, `tensorflow` | Both build much of their public API lazily at import time. Static analysis cannot see those attributes and reports false `no-member` errors. Excluding them from introspection avoids the noise without hiding problems in our own code. |
| `fail-under` | `9.0` | Pylint exits non-zero below this, so a lint regression fails the build rather than being merely noted. |
| `max-line-length` | `100` | Pylint 3's own default, stated explicitly so it is not accidentally inherited. |

Only `duplicate-code` is disabled outright. Exactly one message is suppressed inline, in
`conftest.py`, with the reason written next to it:

```python
from app import app  # pylint: disable=import-outside-toplevel
```

Importing `app` at module scope would pull in `model.py` and load the model before the
patch that replaces the loader is active, which is the whole point of the fixture. The
suppression is local to that one line.

## What Pylint reported, and what was done about it

The baseline run scored **7.50/10** ([`evidence/02-pylint-baseline.txt`](evidence/02-pylint-baseline.txt)):

| Message | Where | Resolution |
| --- | --- | --- |
| `C0114` missing module docstring | both modules | Added. `model.py`'s docstring records that the model is loaded once per process, which is the module's most important property. |
| `C0116` missing function docstring | four functions | Added. `preprocess_img`'s notes that it accepts a path *or* a stream, since the parameter name `img_path` suggests otherwise. |
| `W0702` bare `except:` | `app.py` | Replaced with `except (KeyError, OSError, ValueError)`, one per failure a bad upload actually produces, plus `app.logger.exception` so the real cause reaches the log. The bare form also caught `KeyboardInterrupt` and `SystemExit`, so the server could not be interrupted cleanly during a request. |
| `R1710` inconsistent returns | `app.py` | The body was wrapped in `if request.method == 'POST'`, which the route already guarantees. If it were ever false the view returned `None`, which Flask rejects. Removing the redundant check removed the path. |

The final score is **10.00/10** ([`evidence/03-pylint-after-fixes.txt`](evidence/03-pylint-after-fixes.txt)),
reached by fixing eight findings and suppressing one with a stated reason.

### A note on the score

The assignment is explicit that a perfect score is not the objective. It is worth saying
which of these changes mattered and which were bookkeeping. The docstrings raised the
score most (+1.88) but changed no behaviour. The two that mattered were the error
handling and the inconsistent return: the first was hiding the cause of every failure,
and the second left a live path where the view returned `None`. The score moved least
for the fixes that mattered most.

## Pre-commit configuration

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) runs on every commit after
`pre-commit install`:

| Hook | Purpose |
| --- | --- |
| `trailing-whitespace`, `end-of-file-fixer` | Whitespace hygiene. On first run these fixed missing end-of-file newlines in four templates and the stylesheet. |
| `check-yaml` | Validates the GitHub Actions workflow before it reaches CI. |
| `check-added-large-files` (`--maxkb=30720`) | The limit sits just above `digit_model.h5` at 28 MB, so the model can be committed but nothing larger slips in. |
| `check-merge-conflict` | Blocks conflict markers. |
| `debug-statements` | Blocks a stray `breakpoint()` or `pdb` import. |
| `pylint` (local hook) | Runs Pylint from the project's own virtualenv, so it resolves Flask and Keras exactly as CI does. A `language: system` hook is used rather than a pre-commit managed environment, which would not have the dependencies installed. |

## Test suite

Nineteen tests in [`tests/`](../tests), covering the five areas the assignment lists:

| Requirement | Tests |
| --- | --- |
| Home page accessible | `test_home_page_is_served` |
| Valid request, expected response | `test_valid_upload_returns_the_predicted_digit` — also asserts the model was handed exactly one correctly shaped, normalised batch |
| Missing uploaded file | `test_missing_file_is_rejected` |
| Invalid or unexpected input | `test_non_image_upload_is_rejected`, `test_empty_filename_is_rejected`, `test_greyscale_image_is_rejected`, `test_get_is_not_allowed_on_the_prediction_route` |
| Preprocessing / prediction component | Nine tests in `test_model.py` covering shape, dtype, normalisation, path and stream inputs, and the argmax reduction |

### Edge cases

- **1×1 and 4000×10 images** — extreme aspect ratios still resize to 224×224.
- **An image already at 224×224** — the resize is a no-op and must not corrupt anything.
- **A greyscale image** — preprocesses to one channel, so the reshape to three raises
  `ValueError`. The application rejects it rather than predicting from wrong data.
- **An empty filename** — what the browser sends when the form is submitted with nothing
  chosen.
- **A prediction that is not 4** — `test_predict_result_follows_the_probabilities` points
  the stub at class 9, proving the result tracks the model rather than being hard-coded.
- **The stub itself** — `test_stub_probabilities_are_a_valid_distribution` checks the
  fixture still looks like real model output, so the suite cannot drift into testing a
  nonsense stand-in.

### Mocking, and why

`model.py` calls `load_model` at module scope, so importing it reads the 28 MB file and
builds the network. Left alone, every test run would pay that cost before collection.

`conftest.py` patches `keras.models.load_model` at module import — before any test module
imports `app` or `model` — so the application under test receives a stub whose `predict`
returns a fixed ten-class probability vector. The suite runs in **0.44 seconds**.

One test, marked `slow`, exercises the genuine model end to end:
`test_the_real_model_classifies_a_known_sample` loads `digit_model.h5` through a saved
reference to the unpatched loader and asserts a real sample of the digit 4 is classified
as 4 with more than 50% confidence. It also asserts the fixture handed back a real Keras
object, because the stub returns a vector peaking at 4 as well and the test would
otherwise pass whether or not the real model was loaded.

Run everything with `pytest`, or skip the model load with `pytest -m "not slow"`.

## The defect the tests found

Writing the error cases exposed a genuine problem. Both the success page and the error
page were returned with **HTTP 200**, so nothing reading the status code could tell a
rejected upload from a successful prediction. Four tests failed on the first run
([`evidence/04-pytest-failing.txt`](evidence/04-pytest-failing.txt)):

```
FAILED tests/test_routes.py::test_missing_file_is_rejected - assert 200 == 400
FAILED tests/test_routes.py::test_non_image_upload_is_rejected - assert 200 == 400
FAILED tests/test_routes.py::test_empty_filename_is_rejected - assert 200 == 400
FAILED tests/test_routes.py::test_greyscale_image_is_rejected - assert 200 == 400
4 failed, 15 passed in 1.43s
```

The failing tests were committed first (`a0f88aa`) to record the problem, and the fix
followed in `f091634`: the error branch now returns `400`. The page a person sees is
unchanged — same template, same message — so the original behaviour is preserved for
anyone using a browser, while a programmatic caller can now tell the difference.

All nineteen tests pass ([`evidence/05-pytest-passing.txt`](evidence/05-pytest-passing.txt)).

## Commit sequence

```
f091634  fix: return 400 when the upload cannot be processed      <- tests pass
a0f88aa  test: cover the routes, preprocessing and prediction     <- 4 tests fail
e293af1  ci: run formatting checks and Pylint before every commit
ace1a98  fix: catch the failures we expect and log the ones we do not   10.00/10
fbb425f  style: add the module and function docstrings Pylint asked for  9.38/10
cbf3610  ci: add a Pylint configuration                                  7.50/10
```
