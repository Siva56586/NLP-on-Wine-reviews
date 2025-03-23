import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime

# ===== Paths =====
DATASET_PATH = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\winemag-data_first150k.csv"
MODEL_PATH = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\sentiment_model.h5"
TOKENIZER_PATH = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\tokenizer.pkl"
PLOTS_DIR = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\plots"
POSITIVE_PATH = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\positive_keywords.txt"
NEGATIVE_PATH = r"C:\Users\gnane\Desktop\NLP-Wine-Reviews\negative_keywords.txt"

os.makedirs(PLOTS_DIR, exist_ok=True)  # Create plots directory if not exists

# ===== Load Keywords from Files =====
def load_keywords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file.readlines()]

positive_keywords = load_keywords(POSITIVE_PATH)
negative_keywords = load_keywords(NEGATIVE_PATH)

# ===== Set CPU Optimization =====
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# ===== Parameters =====
MAX_LEN = 40
VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
BATCH_SIZE = 256  # Reduced batch size for better stability
EPOCHS = 15  # Increased for better accuracy

# ===== Keyword-based Classification with Priority =====
def classify_review_keyword_based(review):
    review_lower = review.lower()
    has_positive = any(word in review_lower for word in positive_keywords)
    has_negative = any(word in review_lower for word in negative_keywords)

    if has_negative:
        return 0  # Prioritize negative sentiment
    elif has_positive:
        return 1  # Only positive if no negative sentiment is detected
    else:
        return None

# ===== Load and Process Dataset =====
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["description", "points"])

# Assign labels using keyword-based classification
df["label"] = df["description"].apply(classify_review_keyword_based)
df["label"].fillna(df["points"].apply(lambda x: 1 if x >= 90 else 0), inplace=True)

# Tokenization Handling: Load existing tokenizer or create a new one
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["description"].astype(str).str.lower().str.strip().tolist())
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# Tokenization & Padding
texts = df["description"].astype(str).str.lower().str.strip().tolist()
labels = df["label"].values
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

# ===== Handle Class Imbalance (Using RandomOverSampler for Simplicity) =====
ros = RandomOverSampler(sampling_strategy=1.0, random_state=42)  # Fully balance dataset
X_resampled, y_resampled = ros.fit_resample(padded_sequences, labels)

# ===== Train/Test Split (After Resampling) =====
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# ===== Calculate Class Weights =====
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ===== Optimized Model =====
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(128, return_sequences=False),  # Increased LSTM units for better learning
    Dropout(0.2),  # Reduced dropout to avoid over-regularization
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# ===== Train Model with Class Weights =====
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, class_weight=class_weights_dict)

# ===== Save Model =====
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")

# ===== Evaluate Model =====
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.5).astype("int32").flatten()

# Final Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"✅ Final Accuracy: {accuracy:.4f}")

# ===== Classification Report =====
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ===== Save Accuracy Plot =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_filename = os.path.join(PLOTS_DIR, f"accuracy_plot_{timestamp}.png")
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.grid(True)
plt.savefig(plot_filename)
plt.show()
print(f"📊 Accuracy plot saved at: {plot_filename}")
