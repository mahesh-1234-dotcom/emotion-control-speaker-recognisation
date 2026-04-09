import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("results.csv")

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. EMOTION DISTRIBUTION GRAPH
# -----------------------------
emotion_counts = df['predicted'].value_counts()

plt.figure()
emotion_counts.plot(kind='bar')
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.savefig("outputs/emotion_distribution.png")
plt.close()

# -----------------------------
# 2. CONFIDENCE GRAPH
# -----------------------------
plt.figure()
plt.plot(df['confidence'])
plt.title("Confidence Over Time")
plt.xlabel("Sample")
plt.ylabel("Confidence")
plt.savefig("outputs/confidence_plot.png")
plt.close()

# -----------------------------
# 3. CONFUSION MATRIX FIGURE
# -----------------------------
y_true = df['actual']
y_pred = df['predicted']

cm = confusion_matrix(y_true, y_pred)
labels = sorted(df['actual'].unique())

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

# annotate numbers
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# -----------------------------
# 4. TABLE AS IMAGE
# -----------------------------
plt.figure(figsize=(8, 4))
plt.axis('off')

table_data = df.tail(10)  # last 10 rows
table = plt.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title("Live Prediction Table")
plt.savefig("outputs/table_results.png")
plt.close()

print("✅ All figures saved in 'outputs/' folder")

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -----------------------------
# 5. CLASSIFICATION REPORT FIGURE
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

# convert to dataframe
report_df = pd.DataFrame(report).transpose()

# keep only needed rows
report_df = report_df.iloc[:-1, :]  # remove 'accuracy' row for table clarity

# round values
report_df = report_df.round(2)

# create figure
plt.figure(figsize=(10, 5))
plt.axis('off')

table = plt.table(
    cellText=report_df.values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc='center',
    cellLoc='center'
)

# styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

plt.title("Classification Report", fontsize=16)

# save
plt.savefig("outputs/classification_report.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Classification report saved!")
