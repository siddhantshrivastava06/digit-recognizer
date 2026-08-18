# Digit Recognizer

A Flask web app where you draw a digit (0-9) on a canvas and a trained model predicts it.

## What this project is for

The point was learning to take a model out of a notebook and serve it through a real app: Flask backend, HTML/JS frontend,
request/response flow between the two.

## Stack

- **Model**: RandomForestClassifier (scikit-learn)
- **Backend**: Flask. Serves the page, /predict accepts pixel data as JSON and
  returns a prediction.
- **Frontend**: plain HTML5 canvas + vanilla JS. No frameworks.

## How it works

1. User draws a digit on a 140x140 canvas.
2. On Predict, JS finds the bounding box of the drawn (non-black) pixels, crops to that
   box with padding, and resizes just the crop to 8x8 — matching the format the model was
   trained on.
3. The 8x8 grayscale values are sent to /predict as a flat array of 64 numbers.
4. Flask loads the pickled model, predicts, returns the digit as JSON.
5. JS displays the result.

<img width="333" height="451" alt="image" src="https://github.com/user-attachments/assets/044ef6e9-7ac1-4aff-956d-321be03f06e5" />

<img width="316" height="454" alt="image" src="https://github.com/user-attachments/assets/5b69ed19-4864-450e-bd8f-d6ecd0dde57d" />


## Known limitations

- Trained on `load_digits` (8x8, 1,797 images), not full MNIST (28x28, 70,000 images).
  Chosen deliberately to keep the modeling side simple so effort went into the Flask/JS
  integration instead. Swapping in full MNIST later would be a natural next step.
- Hand-drawn digits still won't match training data perfectly (stroke width, exact
  centering), so occasional misreads are expected even after the centering fix.

## Running it locally

```
pip install flask scikit-learn numpy
python train_model.py   # trains and saves digit_model.pkl
python app.py            # starts the Flask server
```

Then open `http://127.0.0.1:5000` in a browser.

## Possible next steps

- Swap in full MNIST for a more resume-recognizable dataset.
- Add a confidence score alongside the prediction (model.predict_proba).
- Deploy it publicly (Render, Railway, or similar) instead of localhost-only.
