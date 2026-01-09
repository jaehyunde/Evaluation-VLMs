import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from posevit.dataset import PoseSeqDataset
from posevit.model import PoseSeqTransformer as PoseViT
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report  # ✅ Added
)

# =====================
# 설정
# =====================
dataset = "SaGA"
INDEX_PATH = "test/index_all.csv"
OUT_DIR = "test/outputs"
CKPT_PATH = os.path.join(OUT_DIR, f"ckpts/{dataset}posevit_best.pt")
REPORT_PATH = os.path.join(OUT_DIR, f"reports/{dataset}.csv")
PLOT_PATH = os.path.join(OUT_DIR, f"reports/{dataset}.png")
DETAILED_REPORT_PATH = os.path.join(OUT_DIR, f"reports/{dataset}_detailedreport.csv")  # ✅ Added

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
T, S = 32, 16
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.0005070
D_IN = 80  # feature 차원 (예: MediaPipe upper body 등)

# ✅ 현재 학습에 사용 중인 디바이스 확인
print(f"🚀 Training on device: {DEVICE} ({'GPU available' if torch.cuda.is_available() else 'CPU only'})")

os.makedirs(os.path.join(OUT_DIR, "ckpts"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

# === Step 1: 데이터 분리 ===
print("📂 Loading index files...")

index_all = pd.read_csv("test/index_all.csv")
index_test = pd.read_csv("test/index_test.csv")  # ✅ 외부 테스트 세트 사용

# train/val = 90% / 10%
train_df, val_df = train_test_split(index_all, test_size=0.1, random_state=42)

train_df.to_csv("test/index_train.csv", index=False)
val_df.to_csv("test/index_val.csv", index=False)

print(f"✅ Train: {len(train_df)} clips, Val: {len(val_df)}, Test (external): {len(index_test)}")

# =====================
# 2️⃣ Dataset & Dataloader
# =====================
train_ds = PoseSeqDataset("test/index_train.csv", T=T, S=S)
val_ds = PoseSeqDataset("test/index_val.csv", T=T, S=S)
test_ds = PoseSeqDataset("test/index_test.csv", T=T, S=S)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =====================
# 3️⃣ 모델 정의
# =====================
model = PoseViT(d_in=D_IN, d_model=64, nhead=2, num_layers=4, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# 4️⃣ 학습 루프
# =====================
best_f1 = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for Xb, yb in tqdm(train_loader, desc=f"Train {epoch}"):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits.view(-1, 2), yb.view(-1))
        loss.backward()
        optimizer.step()

    # === Validation ===
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb).argmax(dim=-1)
            y_true.extend(yb.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())

    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print(f"[Val] ep{epoch} F1={f1:.4f} P={prec:.4f} R={rec:.4f} Acc={acc:.4f}")

    # 최고 F1 모델 저장 (cfg 포함)
    if f1 > best_f1:
        best_f1 = f1
        ckpt = {
            "cfg": {
                "d_in": D_IN,
                "d_model": 256,
                "nhead": 8,
                "num_layers": 3,
                "num_classes": 2,
                "T": T,
                "S": S,
                "lr": LR,
                "epochs": EPOCHS
            },
            "model": model.state_dict()
        }
        torch.save(ckpt, CKPT_PATH)
        print(f"✓ Saved best model to {CKPT_PATH}")

# =====================
# 5️⃣ 테스트 평가
# =====================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

def evaluate(loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb).argmax(dim=-1)
            y_true.extend(yb.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())
    return y_true, y_pred

# ✅ Validation/Test 구분
y_true_val, y_pred_val = evaluate(val_loader)
y_true_test, y_pred_test = evaluate(test_loader)

val_metrics = {
    "precision": precision_score(y_true_val, y_pred_val),
    "recall": recall_score(y_true_val, y_pred_val),
    "f1": f1_score(y_true_val, y_pred_val),
    "accuracy": accuracy_score(y_true_val, y_pred_val)
}

test_metrics = {
    "precision": precision_score(y_true_test, y_pred_test),
    "recall": recall_score(y_true_test, y_pred_test),
    "f1": f1_score(y_true_test, y_pred_test),
    "accuracy": accuracy_score(y_true_test, y_pred_test)
}

# =====================
# 6️⃣ 결과 저장
# =====================
df = pd.DataFrame([
    {"split": "Validation", **val_metrics},
    {"split": "Test", **test_metrics}
])
df.to_csv(REPORT_PATH, index=False)
print(f"📊 Report saved to {REPORT_PATH}")

# ✅ Added: classification_report
report = classification_report(
    y_true_test,
    y_pred_test,
    target_names=["NoGesture", "Gesture"],
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df["accuracy"] = accuracy_score(y_true_test, y_pred_test)
report_df.to_csv(DETAILED_REPORT_PATH)
print(f"📄 Detailed classification report saved to {DETAILED_REPORT_PATH}")
print(report_df)

# =====================
# 7️⃣ 그래프 시각화
# =====================
plt.figure(figsize=(6, 4))
df_melt = df.melt(id_vars="split", var_name="metric", value_name="score")
for metric in ["precision", "recall", "f1", "accuracy"]:
    plt.bar(df_melt[df_melt["metric"] == metric]["split"],
            df_melt[df_melt["metric"] == metric]["score"], alpha=0.7, label=metric)
plt.legend()
plt.title("Validation vs Test Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📈 Plot saved to {PLOT_PATH}")
