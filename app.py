from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = np.array(data['pixels']).reshape(1, -1)  # shape (1, 64)
    prediction = model.predict(pixels)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)