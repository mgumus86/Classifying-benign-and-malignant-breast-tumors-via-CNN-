#%%
import torch
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/yolo_two_class_final/weights/best.pt')  # custom trained model

# Establish counters
total = 0
correct = 0
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

correctLabels = []
finalPrediction = []

# Loop through images
for im in os.listdir('.\\yolo_data\\images\\test'):

    # Pull corresponding label file
    label = f'.\\yolo_data\\labels\\test\\{im[:-4]}.txt'
    with open(label) as f:
        lines = [line for line in f]
        label = int(lines[0].split()[0])

        if label == 1:
            label = 'malignant'
        else:
            label = 'benign'

    # Inference
    results = model(os.path.join('.\\yolo_data\\images\\test',im))
    try:
        inferred_label = results.pandas().xyxy[0].name.values[0]
    except:
        inferred_label = "not_labelled"

    # Assemble statistics
    if (label == 'malignant') and (inferred_label == 'malignant'):
        truePositive += 1
    elif (label == 'benign') and (inferred_label == 'benign'):
        trueNegative += 1
    elif (label == 'malignant') and (inferred_label == 'benign'):
        falsePositive += 1
    elif (label == 'benign') and (inferred_label == 'malignant'):
        falseNegative += 1

    correctLabels.append(inferred_label)
    finalPrediction.append(label)




    if label == inferred_label:
        correct += 1
    total += 1

# Calc stats
if (truePositive + falsePositive > 0):
    precision = truePositive / (truePositive + falsePositive)
else:
    precision = 0.0

if (truePositive + falseNegative > 0):
    recall = truePositive / (truePositive + falseNegative)
else:
    recall = 0.0

if (precision+recall > 0):
    f1 = 2*((precision*recall)/(precision+recall))
else:
    f1 = 0.0

print(f"Total Accuracy: {correct/total}")
print(f"Precision {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")

correctLabels = [1 if i=='malignant' else 0 for i in correctLabels ]
finalPrediction = [1 if i=='malignant' else 0 for i in finalPrediction ]

cf_matrix = confusion_matrix(correctLabels, finalPrediction)

classes =  ('Benign', 'Malignant')
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])

plt.figure()
sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



# %%
