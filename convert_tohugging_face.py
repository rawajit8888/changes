import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 1. Load JSON from Label Studio
# ----------------------------
def load_labelstudio(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for task in data:
        text = task["data"]["text"]

        task_labels = []
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") == "taxonomy":
                    cats = res["value"].get("taxonomy", [])
                    if len(cats) > 0:
                        for c in cats[0]:      # because taxonomy is list-of-list
                            task_labels.append(c)

        if len(task_labels) > 0:
            texts.append(text)
            labels.append(list(set(task_labels)))

    return texts, labels


# ----------------------------
# 2. Train/Test Split
# ----------------------------
def prepare_data(texts, labels, test_size=0.2):
    return train_test_split(texts, labels, test_size=test_size, random_state=42)


# ----------------------------
# 3. Load Model + Encoders
# ----------------------------
def load_model(model_path, encoder_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        problem_type="multi_label_classification"
    )

    # Load MultiLabelBinarizer
    with open(encoder_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    return tokenizer, model, mlb


# ----------------------------
# 4. Predict multilabel outputs
# ----------------------------
def predict(texts, tokenizer, model, threshold=0.5):
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
    predictions = (probs >= threshold).astype(int)
    return predictions


# ----------------------------
# 5. Evaluation + Save CSV + Confusion Matrix
# ----------------------------
def evaluate(y_true, y_pred, mlb):

    # Convert back to label names
    true_labels = mlb.inverse_transform(y_true)
    pred_labels = mlb.inverse_transform(y_pred)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0)
    print(report)

    # Save evaluation CSV
    df = pd.DataFrame({
        "true_labels": [", ".join(x) for x in true_labels],
        "predicted_labels": [", ".join(x) for x in pred_labels]
    })

    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved → evaluation_results.csv")

    return report


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():

    json_path = "project.json"  # <<< Change This
    model_path = "model_folder"  # folder with config + model.safetensors
    encoder_path = "encoders/classes.json"  # your saved classes file

    print("Loading data…")
    texts, labels = load_labelstudio(json_path)

    print("Train/Test splitting…")
    X_train, X_test, y_train, y_test = prepare_data(texts, labels)

    print("Loading model + encoder…")
    tokenizer, model, mlb = load_model(model_path, encoder_path)

    print("Encoding labels…")
    y_test_bin = mlb.transform(y_test)

    print("Predicting…")
    y_pred_bin = predict(X_test, tokenizer, model)

    print("Evaluating…")
    evaluate(y_test_bin, y_pred_bin, mlb)


if __name__ == "__main__":
    main()
