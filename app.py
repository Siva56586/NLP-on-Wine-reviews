from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
MODEL_PATH = "C:/Users/gnane/Desktop/NLP-Wine-Reviews/sentiment_model.h5"
TOKENIZER_PATH = "C:/Users/gnane/Desktop/NLP-Wine-Reviews/tokenizer.pkl"
MAX_LEN = 40

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form['review']
        sequence = tokenizer.texts_to_sequences([review_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction_prob = model.predict(padded_sequence)[0][0]
        sentiment = "Positive" if prediction_prob >= 0.5 else "Negative"
        confidence = round(float(prediction_prob if sentiment == "Positive" else 1 - prediction_prob), 4)
        return jsonify({"prediction": sentiment, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
