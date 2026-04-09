import sounddevice as sd
import numpy as np
import librosa
import pandas as pd
import os
import pickle
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from datetime import datetime

# LOAD MODEL
model = load_model("emotion_model.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

SR = 22050
DURATION = 2

# 👉 IMPORTANT: Ask actual emotion (for confusion matrix)
true_label = input("Enter actual emotion (happy/neutral/sad/angry): ")

# RECORD
print("Recording...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
audio = audio.flatten()

# SAVE AUDIO
os.makedirs("recordings", exist_ok=True)
filename = f"recordings/audio_{datetime.now().strftime('%H%M%S')}.wav"
write(filename, SR, audio)

# MFCC
# -----------------------------
# EXTRACT MFCC (FIXED)
# -----------------------------
mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)

# Fix time dimension
if mfcc.shape[1] < 100:
    mfcc = np.pad(mfcc, ((0,0),(0,100-mfcc.shape[1])), mode='constant')
else:
    mfcc = mfcc[:, :100]

# Normalize
mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

# CNN reshape
mfcc = mfcc.reshape(1, 40, 100, 1)

# -----------------------------
# PREDICT
# -----------------------------
prediction = model.predict(mfcc)
pred_label = le.inverse_transform([np.argmax(prediction)])[0]
confidence = np.max(prediction)

print("Predicted:", pred_label, "Confidence:", confidence)

# SAVE TO CSV
data = {
    "time": datetime.now(),
    "file": filename,
    "actual": true_label,
    "predicted": pred_label,
    "confidence": float(confidence)
}

df = pd.DataFrame([data])

if os.path.exists("results.csv"):
    df.to_csv("results.csv", mode='a', header=False, index=False)
else:
    df.to_csv("results.csv", index=False)

print("✅ Saved to results.csv")
