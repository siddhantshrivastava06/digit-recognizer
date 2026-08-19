# Digit Recognizer

A Flask web app where you draw a digit (0-9) on a canvas and a trained model predicts it.

## Description

The point was learning to use a model and understand a model and build a Flask backend and a HTML/JS frontend.

## Stack

- **Model**: RandomForestClassifier (impored from scikit-learn)
- **Backend**: Flask. Serves the page, /predict accepts pixel data and
  returns a prediction.
- **Frontend**: HTML and JS. 

## Approach & Results

1. User draws a digit on a 140x140 canvas.
2. On Predict, JS finds the bounding box of the drawn (non-black) pixels, crops to that
   box with padding, and resizes just the crop to 8x8 — matching the format the model was
   trained on.
3. The 8x8 grayscale values are sent to /predict. 
4. The result is displayed.

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
