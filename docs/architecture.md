# Step 2 — Legacy code and architecture

## Main files

| File | Purpose |
| --- | --- |
| `app.py` | The Flask entry point. Creates the application object and defines both routes: `/` serves the upload form, `/prediction` accepts the uploaded file and returns the predicted digit. |
| `model.py` | The machine-learning layer. Loads the trained Keras model at import time and exposes two functions: `preprocess_img` turns an uploaded file into a model-ready array, and `predict_result` runs inference and returns a single digit. |
| `digit_model.h5` | The trained model itself: a 28 MB Keras 2 HDF5 file, a transfer-learning classifier over 224x224 RGB images with ten output classes (digits 0-9). |
| `templates/layout.html` | The Jinja2 base template. Pulls in Bootstrap 4.3 and jQuery from a CDN, the project's own `custom.css`, and `image_upload.js`. Defines the `title` and `content` blocks. |
| `templates/index.html` | The upload page. A `multipart/form-data` form posting to `predict_image_file`, with a file input named `file` and a client-side preview area. |
| `templates/result.html` | The result page. Shows `predictions` when inference succeeded, and `err` when it did not. |
| `static/css/custom.css` | The gradient background and upload-area styling. |
| `static/js/image_upload.js` | Client-side only. Reads the chosen file with a `FileReader` to show a thumbnail before submission, and writes the file name into the label. It does not talk to the server. |
| `test_images/` | 100 sample images in ten folders, `0` through `9`, for manual testing. |

## Where the Flask application is configured

In `app.py`, lines 6 and 29-30:

```python
app = Flask(__name__)
...
if __name__ == "__main__":
    app.run(port=9000, debug=True)
```

There is no config object, no environment-variable handling and no separate settings
module. `Flask(__name__)` takes every default, which is how `templates/` and `static/`
are discovered. The only explicit configuration is the port and the debug flag, and
both are set in the `__main__` guard, so they apply only when the file is run directly
with `python app.py`. Started through the Flask CLI instead, the app listens on the
CLI's default port 5000.

## Where the model is loaded

In `model.py`, at module level rather than inside a function:

```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")
model = load_model(MODEL_PATH)
```

This is a significant design detail. Because the call sits at module scope, the full
28 MB model is read and built the moment anything imports `model` — including
`app.py` at startup, and including any test file. It is loaded once per process and
held in the module-level global `model`, which `predict_result` closes over.

The benefit is that inference does not pay the loading cost per request. The cost is
that the module cannot be imported at all without the model file present, which makes
importing it in a test suite slow and forces the tests in Step 4 to mock it.

(Before the fix recorded in the setup log, this line pointed at an absolute Windows
path on the original author's machine.)

## Where uploaded images are preprocessed

In `model.py`, `preprocess_img`:

```python
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape
```

Four steps: open with Pillow, resize to the 224x224 the model expects, convert to a
float array scaled from 0-255 down to 0-1, then reshape into a batch of one.

Two things are worth noting for later steps. The parameter is named `img_path`, but
`app.py` passes it a file *stream*, not a path; Pillow accepts either, so the name is
misleading rather than wrong. And the resize does not preserve aspect ratio, so a
non-square image is distorted rather than cropped or padded.

## Where templates and static resources are stored

In the Flask defaults: `templates/` for the three Jinja2 templates and `static/` for CSS
and JavaScript, both resolved relative to `app.py`. Bootstrap and jQuery are not
vendored; they load from a CDN in `layout.html`, so the styled pages need an internet
connection.

## How an uploaded image becomes a prediction

1. The browser requests `GET /`. `main()` renders `index.html`, which extends
   `layout.html`.
2. The user picks a file. `image_upload.js` previews it in the browser and updates the
   label. Nothing has reached the server yet.
3. Submitting posts `multipart/form-data` to `POST /prediction`, with the file under the
   form field `file`.
4. `predict_image_file()` reads `request.files['file'].stream` and hands that stream to
   `preprocess_img`, which returns a `(1, 224, 224, 3)` float array.
5. `predict_result` calls `model.predict` on the array. The model returns ten class
   probabilities; `np.argmax` reduces them to the single most likely digit.
6. The digit is stringified and passed to `result.html` as `predictions`, and the
   rendered page goes back to the browser.
7. If anything in steps 4-6 raises, a bare `except:` catches it and the same template is
   rendered with `err` set to "File cannot be processed."

## Observations carried into Steps 3 and 4

These are recorded here because they shape the Pylint fixes and the test suite:

- **The bare `except:` flattens every failure into one message.** A missing form field
  (`KeyError`), a file that is not an image (`UnidentifiedImageError`) and a genuine
  model error all produce identical output, and the real exception is never logged.
  It also catches `KeyboardInterrupt` and `SystemExit`, which is why Pylint flags it
  as `W0702`.
- **`predict_image_file` can return `None`.** The body is wrapped in
  `if request.method == 'POST'`, which is redundant given the route already restricts
  methods, and nothing returns if that check is somehow false. Flask raises an error
  when a view returns `None`. Pylint flags this as `R1710`.
- **No docstrings anywhere**, on either module or any function (`C0114`, `C0116`).
- **The model is a module-level import side effect**, so tests must mock it to run at
  any reasonable speed.

## Architecture diagram

![Architecture](diagrams/architecture.svg)

The deployment picture is deliberately simple: one Flask process holding the model in
memory, serving a browser over HTTP. There is no database, no queue, no separate model
server and no persistence. The uploaded file is read from the request stream and never
written to disk.
