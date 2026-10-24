"""Tests for the Flask routes."""

import io

from conftest import make_image_bytes


def test_home_page_is_served(client):
    """The home page responds and renders the upload form."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Hand Sign Digit Language Detection" in body
    assert 'type="file"' in body


def test_valid_upload_returns_the_predicted_digit(client, image_bytes, stub_model):
    """A good image produces the digit the model predicts."""
    response = client.post(
        "/prediction",
        data={"file": (image_bytes, "sign.jpeg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert "4" in response.get_data(as_text=True)
    stub_model.predict.assert_called_once()

    # The model must be given one correctly shaped, normalised image.
    batch = stub_model.predict.call_args.args[0]
    assert batch.shape == (1, 224, 224, 3)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_missing_file_is_rejected(client, stub_model):
    """A POST with no file field is a client error, not a prediction."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "File cannot be processed." in response.get_data(as_text=True)
    stub_model.predict.assert_not_called()


def test_non_image_upload_is_rejected(client, stub_model):
    """A file that is not an image is a client error, not a prediction."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "File cannot be processed." in response.get_data(as_text=True)
    stub_model.predict.assert_not_called()


def test_empty_filename_is_rejected(client, stub_model):
    """Submitting the form with no file chosen sends an empty part."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    stub_model.predict.assert_not_called()


def test_greyscale_image_is_rejected(client, stub_model):
    """A single-channel image cannot be reshaped to three channels.

    The model expects (1, 224, 224, 3). A greyscale image preprocesses to one
    channel, so the reshape raises ValueError rather than the application
    silently predicting from the wrong data.
    """
    response = client.post(
        "/prediction",
        data={"file": (make_image_bytes(mode="L"), "grey.jpeg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    stub_model.predict.assert_not_called()


def test_get_is_not_allowed_on_the_prediction_route(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405
