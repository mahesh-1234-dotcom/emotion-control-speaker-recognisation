import os
import numpy as np
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
EMOTIONS = ['happy', 'neutral', 'sad', 'angry']
SR = 22050

# -----------------------------
# FEATURE EXTRACTION (CNN)
# -----------------------------
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)

# Fix shape to 100 time frames
    if mfcc.shape[1] < 100:
      mfcc = np.pad(mfcc, ((0,0),(0,100-mfcc.shape[1])), mode='constant')
    else:
      mfcc = mfcc[:, :100]

# Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

# Reshape for CNN
    mfcc = mfcc.reshape(1, 40, 100, 1)
    return mfcc 

    

    


# -----------------------------
# LOAD DATASET
# -----------------------------
X = []
y = []

for emotion in EMOTIONS:
    folder = os.path.join(DATASET_PATH, emotion)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            mfcc = extract_mfcc(path)

            X.append(mfcc)
            y.append(emotion)

# -----------------------------
# CONVERT TO NUMPY
# -----------------------------
X = np.array(X)
y = np.array(y)

X = (X- np.mean(X)) / (np.std(X) + 1e-6)

print("✅ Data Loaded:", X.shape)

print("class distribution:" , dict(zip(*np.unique(y, return_counts=True))))

# -----------------------------
# RESHAPE FOR CNN
# -----------------------------
X = X.reshape(X.shape[0], 40, 100, 1)

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# One-hot encoding (fixes your error)
y_categorical = to_categorical(y_encoded)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# -----------------------------
# CNN MODEL
# -----------------------------

from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(40,100,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(len(EMOTIONS), activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("emotion_model.h5")
print("✅ Model saved!")

# -----------------------------
# SAVE GRAPH DATA
# -----------------------------
np.save("train_acc.npy", history.history['accuracy'])
np.save("val_acc.npy", history.history['val_accuracy'])
np.save("train_loss.npy", history.history['loss'])
np.save("val_loss.npy", history.history['val_loss'])

print("✅ Training data saved!")

# -----------------------------
# FINAL EVALUATION
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Final Accuracy: {acc*100:.2f}%")
