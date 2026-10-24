"""Shared pytest fixtures.

`model.py` calls `load_model` at module scope, so importing it reads a 28 MB
HDF5 file and builds the network. That happens before any test runs and would
dominate the suite's runtime, so `load_model` is patched here - before any test
module imports `app` or `model` - to hand back a stub whose `predict` returns a
fixed probability vector.

Tests that need the genuine model ask for the `real_model` fixture, which uses
the reference to the unpatched loader kept below. Those tests are marked `slow`
so they can be deselected.
"""

import io
import os
from unittest.mock import MagicMock, patch

import keras.models
import numpy as np
import pytest
from PIL import Image

# Keep a handle on the real loader before patching it, for the slow tests.
_ORIGINAL_LOAD_MODEL = keras.models.load_model

# Ten class probabilities with the highest at index 4, so a stubbed prediction
# is the digit 4 and a test can tell a real answer from an accidental zero.
STUB_PROBABILITIES = np.array(
    [[0.01, 0.01, 0.02, 0.05, 0.86, 0.02, 0.01, 0.01, 0.005, 0.005]],
    dtype="float32",
)

_stub_model = MagicMock(name="stub_keras_model")
_stub_model.predict.return_value = STUB_PROBABILITIES

# Started at import time, never stopped: the stub stands in for the whole
# session. model.py binds `load_model` when it is first imported, which is
# after this module has run.
patch.object(keras.models, "load_model", return_value=_stub_model).start()


@pytest.fixture(name="stub_model")
def stub_model_fixture():
    """The stubbed Keras model, so a test can change what predict returns."""
    _stub_model.predict.reset_mock()
    _stub_model.predict.return_value = STUB_PROBABILITIES
    return _stub_model


@pytest.fixture(name="client")
def client_fixture():
    """A Flask test client for the application."""
    # Deliberately imported inside the fixture, not at module scope: importing
    # app pulls in model.py, which loads the model at import time. It must not
    # happen until the patch above is in place.
    from app import app  # pylint: disable=import-outside-toplevel

    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


@pytest.fixture(name="real_model", scope="session")
def real_model_fixture():
    """The genuine trained model, loaded once for the tests marked slow."""
    path = os.path.join(os.path.dirname(__file__), "digit_model.h5")
    return _ORIGINAL_LOAD_MODEL(path)


def make_image_bytes(size=(300, 300), colour=(120, 90, 60), image_format="JPEG",
                     mode="RGB"):
    """Return an in-memory image file, so tests do not depend on fixtures on disk."""
    buffer = io.BytesIO()
    Image.new(mode, size, colour if mode == "RGB" else 128).save(buffer, image_format)
    buffer.seek(0)
    return buffer


@pytest.fixture(name="image_bytes")
def image_bytes_fixture():
    """A valid in-memory JPEG."""
    return make_image_bytes()
