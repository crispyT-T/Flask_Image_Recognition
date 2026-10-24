"""Tests for preprocessing and inference."""

import io

import numpy as np
import pytest
from PIL import UnidentifiedImageError

from conftest import STUB_PROBABILITIES, make_image_bytes
from model import predict_result, preprocess_img


def test_preprocess_returns_a_single_normalised_rgb_batch():
    """Any RGB image becomes one 224x224x3 array scaled to 0-1."""
    result = preprocess_img(make_image_bytes(size=(640, 480)))

    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_accepts_a_path_as_well_as_a_stream(tmp_path):
    """The parameter is named img_path, and a real path must still work."""
    path = tmp_path / "sign.jpeg"
    path.write_bytes(make_image_bytes().getvalue())

    assert preprocess_img(str(path)).shape == (1, 224, 224, 3)


@pytest.mark.parametrize("size", [(1, 1), (224, 224), (4000, 10)])
def test_preprocess_resizes_any_dimensions(size):
    """Edge case: extreme and already-correct sizes resize without error."""
    assert preprocess_img(make_image_bytes(size=size)).shape == (1, 224, 224, 3)


def test_preprocess_rejects_a_file_that_is_not_an_image():
    """Pillow raises rather than returning something the model would accept."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(io.BytesIO(b"plain text, not an image"))


def test_preprocess_rejects_a_missing_path():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("no/such/image.jpeg")


def test_predict_result_returns_the_most_likely_class(stub_model):
    """The ten probabilities are reduced to the index of the largest."""
    batch = preprocess_img(make_image_bytes())

    assert predict_result(batch) == 4
    stub_model.predict.assert_called_once()


def test_predict_result_follows_the_probabilities(stub_model):
    """Edge case: the answer tracks the model, it is not hard-coded."""
    probabilities = np.zeros((1, 10), dtype="float32")
    probabilities[0][9] = 1.0
    stub_model.predict.return_value = probabilities

    assert predict_result(preprocess_img(make_image_bytes())) == 9


def test_predict_result_returns_an_integer(stub_model):
    """The template stringifies the result, so it must be a scalar integer."""
    result = predict_result(preprocess_img(make_image_bytes()))

    assert isinstance(result, (int, np.integer))
    assert 0 <= int(result) <= 9
    assert str(result) == "4", "result.html renders str(pred), so it must read cleanly"
    stub_model.predict.assert_called_once()


@pytest.mark.slow
def test_the_real_model_classifies_a_known_sample(real_model):
    """The genuine model, on a genuine sample, still predicts the right digit.

    Mocked tests prove the plumbing. This one proves the model file in the
    repository is the trained model and still loads.
    """
    # Guard against the test passing vacuously: the stub also returns a
    # (1, 10) vector peaking at 4, so prove this is a genuine Keras model.
    assert type(real_model).__module__.startswith("keras")

    batch = preprocess_img("test_images/4/Sign 4 (92).jpeg")
    probabilities = real_model.predict(batch, verbose=0)

    assert probabilities.shape == (1, 10)
    assert np.argmax(probabilities[0]) == 4
    assert probabilities[0][4] > 0.5, "the correct class should dominate"


def test_stub_probabilities_are_a_valid_distribution():
    """Guards the fixture itself: the stub must look like real model output."""
    assert STUB_PROBABILITIES.shape == (1, 10)
    assert STUB_PROBABILITIES.sum() == pytest.approx(1.0, abs=1e-6)
