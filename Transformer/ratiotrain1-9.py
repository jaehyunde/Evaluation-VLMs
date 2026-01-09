"""
train_eval_manual_alpha.py
-----------------------------------
이 스크립트는 ALPHA 값을 직접 수정하여
훈련용 데이터와 외부 데이터의 혼합 비율을 실험하기 위한 버전입니다.

예시:
    ALPHA = 0.0  # 외부 데이터 전혀 포함 안 함
    ALPHA = 0.3  # 외부 데이터 30% 포함
    ALPHA = 0.5  # 외부 데이터 절반 교환
    ALPHA = 0.7  # 외부 데이터 70% 포함
"""

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
    classification_report
)

# =====================
# 수동 설정 영역
# =====================

# 🔧 ALPHA를 직접 바꿔가며 실험할 수 있음 (0.0 ~ 1.0)
ALPHA = 0.9   # ✅ 이 값을 수동으로 변경

INDEX_PATH_ALL = "test/index_all.csv"
INDEX_PATH_EXT = "test/index_test.csv"
OUT_DIR = "test/outputs"

# ALPHA -> rate 문자열 자동 변환 (예: 0.3 -> "7:3")
train_ratio = int(round((1 - ALPHA) * 10))
ext_ratio   = int(round(ALPHA * 10))
rate = f"{train_ratio}:{ext_ratio}"
rate_tag = rate.replace(":", "-")

# 파일 경로 자동 반영
CKPT_PATH = os.path.join(OUT_DIR, f"ckpts/posevit_best_{rate_tag}.pt")
REPORT_PATH = os.path.join(OUT_DIR, f"reports/posevit_{rate_tag}.csv")
PLOT_PATH = os.path.join(OUT_DIR, f"reports/posevit_{rate_tag}.png")
DETAILED_REPORT_PATH = os.path.join(OUT_DIR, f"reports/posevit_detailed_{rate_tag}.csv")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "ckpts"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# ✅ 기존 하이퍼파라미터 유지
T, S = 32, 16
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.0005070
D_IN = 80

print(f"🚀 Training on {DEVICE}")
print(f"🔀 ALPHA={ALPHA}  → rate(train:ext)={rate}")
print(f"💾 Paths tagged with '{rate_tag}'")

# =====================
# 데이터 섞기 함수
# =====================
def mix_datasets(train_df: pd.DataFrame,
                 ext_df: pd.DataFrame,
                 alpha: float,
                 random_state: int = 42):
    n_ext = len(ext_df)
    n_train = len(train_df)

    if n_ext == 0 or n_train == 0 or alpha <= 0.0:
        return train_df.copy(), ext_df.copy()

    k = int(alpha * n_ext)
    k = max(0, min(k, n_ext, n_train))
    if k == 0:
        return train_df.copy(), ext_df.copy()

    ext_sample = ext_df.sample(n=k, random_state=random_state)
    train_sample = train_df.sample(n=k, random_state=random_state + 1)

    train_new = pd.concat(
        [train_df.drop(train_sample.index), ext_sample],
        ignore_index=True
    )
    ext_new = pd.concat(
        [ext_df.drop(ext_sample.index), train_sample],
        ignore_index=True
    )
    return train_new, ext_new

# =====================
# Step 1: index 로드 & 섞기
# =====================
index_all_orig = pd.read_csv(INDEX_PATH_ALL)
index_ext_orig = pd.read_csv(INDEX_PATH_EXT)
index_all_mixed, index_ext_mixed = mix_datasets(index_all_orig, index_ext_orig, ALPHA)

INDEX_ALL_MIXED_PATH = f"test/index_all_mixed_{rate_tag}.csv"
INDEX_EXT_MIXED_PATH = f"test/index_test_mixed_{rate_tag}.csv"
index_all_mixed.to_csv(INDEX_ALL_MIXED_PATH, index=False)
index_ext_mixed.to_csv(INDEX_EXT_MIXED_PATH, index=False)

print(f"📂 index_all_mixed saved: {INDEX_ALL_MIXED_PATH}")
print(f"📂 index_test_mixed saved: {INDEX_EXT_MIXED_PATH}")

# =====================
# Step 2: train/val split
# =====================
train_df, val_df = train_test_split(index_all_mixed, test_size=0.1, random_state=42)
INDEX_TRAIN_PATH = f"test/index_train_mixed_{rate_tag}.csv"
INDEX_VAL_PATH   = f"test/index_val_mixed_{rate_tag}.csv"
train_df.to_csv(INDEX_TRAIN_PATH, index=False)
val_df.to_csv(INDEX_VAL_PATH, index=False)

# =====================
# Dataset & Dataloader
# =====================
train_ds = PoseSeqDataset(INDEX_TRAIN_PATH, T=T, S=S)
val_ds   = PoseSeqDataset(INDEX_VAL_PATH,   T=T, S=S)
test_ds  = PoseSeqDataset(INDEX_EXT_MIXED_PATH, T=T, S=S)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# =====================
# 모델 정의
# =====================
model = PoseViT(d_in=D_IN, d_model=64, nhead=2, num_layers=4, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# 학습 루프
# =====================
best_f1 = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for Xb, yb in tqdm(train_loader, desc=f"Train {epoch}"):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits.view(-1, 2), yb.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= max(1, len(train_loader))

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb).argmax(dim=-1)
            y_true.extend(yb.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())

    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)

    print(f"[Val] ep{epoch} loss={epoch_loss:.4f} | F1={f1:.4f} P={prec:.4f} R={rec:.4f} Acc={acc:.4f}")

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
                "epochs": EPOCHS,
                "alpha": ALPHA,
                "rate": rate,
            },
            "model": model.state_dict()
        }
        torch.save(ckpt, CKPT_PATH)
        print(f"✓ Saved best model to {CKPT_PATH} (Val F1={best_f1:.4f}, alpha={ALPHA}, rate={rate})")

# =====================
# 테스트 평가
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

y_true_val, y_pred_val   = evaluate(val_loader)
y_true_test, y_pred_test = evaluate(test_loader)

val_metrics = {
    "precision": precision_score(y_true_val, y_pred_val),
    "recall":    recall_score(y_true_val, y_pred_val),
    "f1":        f1_score(y_true_val, y_pred_val),
    "accuracy":  accuracy_score(y_true_val, y_pred_val)
}
test_metrics = {
    "precision": precision_score(y_true_test, y_pred_test),
    "recall":    recall_score(y_true_test, y_pred_test),
    "f1":        f1_score(y_true_test, y_pred_test),
    "accuracy":  accuracy_score(y_true_test, y_pred_test)
}

df = pd.DataFrame([
    {"split": "Validation", **val_metrics},
    {"split": "Test",       **test_metrics}
])
df.to_csv(REPORT_PATH, index=False)
print(f"📊 Report saved to {REPORT_PATH}")

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

plt.figure(figsize=(6, 4))
df_melt = df.melt(id_vars="split", var_name="metric", value_name="score")
for metric in ["precision", "recall", "f1", "accuracy"]:
    plt.bar(df_melt[df_melt["metric"] == metric]["split"],
            df_melt[df_melt["metric"] == metric]["score"],
            alpha=0.7,
            label=metric)
plt.legend()
plt.title(f"Validation vs Test (alpha={ALPHA}, rate={rate})")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📈 Plot saved to {PLOT_PATH}")
