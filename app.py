"""Flask web application for hand-sign digit recognition.

Serves an upload form and returns the digit predicted from the uploaded image.
"""

from flask import Flask, render_template, request

from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """Render the upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Predict the digit in the uploaded image and render the result page."""
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except (KeyError, OSError, ValueError):
        # KeyError: no file field in the request.
        # OSError: Pillow could not identify or read the image.
        # ValueError: the array did not fit the shape the model expects.
        app.logger.exception("Could not process the uploaded file")
        error = "File cannot be processed."
        # 400, not 200: the upload was the problem, and a caller should be able
        # to tell a rejected file from a successful prediction by status alone.
        return render_template("result.html", err=error), 400


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
