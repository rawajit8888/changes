import json
import torch
import random
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = r"your_model_path_here"   # Example: r"C:\model\bert"
DATA_PATH = r"tasks.json"              # LS export file

TEST_SPLIT = 0.20   # 20% test, 80% train

# ---------------- LOAD DATA ----------------
texts = []
labels = []

with open(DATA_PATH, "r", encoding="utf8") as f:
    for line in f:
        task = json.loads(line)
        text = task["data"]["text"]
        ann = task["annotations"][0]["result"][0]["value"]["choices"][0]
        texts.append(text)
        labels.append(ann)

print(f"Loaded {len(texts)} samples")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ---------------- TRAIN/TEST SPLIT ----------------
indices = list(range(len(texts)))
random.shuffle(indices)

split_idx = int(len(indices) * (1 - TEST_SPLIT))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

train_texts = [texts[i] for i in train_idx]
train_labels = [labels_encoded[i] for i in train_idx]

test_texts = [texts[i] for i in test_idx]
test_labels = [labels_encoded[i] for i in test_idx]

print(f"Train size: {len(train_texts)}")
print(f"Test size : {len(test_texts)}")

# ---------------- LOAD MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ---------------- TOKENIZE FUNCTION ----------------
def make_loader(texts, labels):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(labels)
    )
    return DataLoader(dataset, batch_size=16)

train_loader = make_loader(train_texts, train_labels)
test_loader = make_loader(test_texts, test_labels)

# ---------------- INFERENCE FUNCTION ----------------
def evaluate(loader):
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            input_ids, mask, y_true = batch
            out = model(input_ids=input_ids, attention_mask=mask)
            pred = torch.argmax(out.logits, dim=1)

            preds.extend(pred.tolist())
            trues.extend(y_true.tolist())

    return trues, preds

# ---------------- EVALUATE ON TRAIN SET ----------------
print("\n=== TRAIN SET PERFORMANCE ===")
train_true, train_pred = evaluate(train_loader)
print(classification_report(train_true, train_pred, target_names=le.classes_))

# ---------------- EVALUATE ON TEST SET ----------------
print("\n=== TEST SET PERFORMANCE ===")
test_true, test_pred = evaluate(test_loader)
print(classification_report(test_true, test_pred, target_names=le.classes_))
