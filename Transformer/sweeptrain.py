import os
import pandas as pd
import numpy as np
import torch
import wandb
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
# 0️⃣ W&B 초기화
# =====================
wandb.init(project="EnvisionTransformer")
config = wandb.config

# Sweep에서 전달받은 값 또는 기본값 사용
LR = config.LR if "LR" in config else 3e-4
BATCH_SIZE = config.BATCH_SIZE if "BATCH_SIZE" in config else 16
EPOCHS = config.EPOCHS if "EPOCHS" in config else 20
d_model = config.d_model if "d_model" in config else 128
nhead = config.nhead if "nhead" in config else 4
num_layers = config.num_layers if "num_layers" in config else 2

# =====================
# 1️⃣ 기본 설정
# =====================
OUT_DIR = "outputs"
CKPT_PATH = os.path.join(OUT_DIR, "ckpts/posevit_best.pt")
REPORT_PATH = os.path.join(OUT_DIR, "reports/posevit_report.csv")
DETAILED_REPORT_PATH = os.path.join(OUT_DIR, "reports/posevit_detailed_report.csv")
PLOT_PATH = os.path.join(OUT_DIR, "reports/posevit_comparison.png")

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
T, S = 32, 16
D_IN = 80  # feature dimension

print(f"🚀 Training on device: {DEVICE} ({'GPU available' if torch.cuda.is_available() else 'CPU only'})")

os.makedirs(os.path.join(OUT_DIR, "ckpts"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

# =====================
# 2️⃣ 데이터셋 로드
# =====================
print("📂 Loading index files...")
index_all = pd.read_csv("data/index_all.csv")
index_test = pd.read_csv("data/index_test.csv")  # ✅ 외부 테스트 세트 사용

train_df, val_df = train_test_split(index_all, test_size=0.1, random_state=42)
train_df.to_csv("data/index_train.csv", index=False)
val_df.to_csv("data/index_val.csv", index=False)

print(f"✅ Train: {len(train_df)} clips, Val: {len(val_df)}, Test (external): {len(index_test)}")

train_ds = PoseSeqDataset("data/index_train.csv", T=T, S=S)
val_ds = PoseSeqDataset("data/index_val.csv", T=T, S=S)
test_ds = PoseSeqDataset("data/index_test.csv", T=T, S=S)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =====================
# 3️⃣ 모델 정의
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = PoseViT(
    d_in=D_IN,
    d_model=128,
    nhead=4,
    num_layers=2,
    num_classes=2
)

# ✅ 여러 GPU 자동 병렬 처리 추가
if torch.cuda.device_count() > 1:
    print(f"⚡ Using {torch.cuda.device_count()} GPUs for training!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

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
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb).argmax(dim=-1)
            y_true_val.extend(yb.cpu().numpy().flatten())
            y_pred_val.extend(pred.cpu().numpy().flatten())

    f1_val = f1_score(y_true_val, y_pred_val)
    prec_val = precision_score(y_true_val, y_pred_val)
    rec_val = recall_score(y_true_val, y_pred_val)
    acc_val = accuracy_score(y_true_val, y_pred_val)

    print(f"[Val] Epoch {epoch}: F1={f1_val:.4f}, P={prec_val:.4f}, R={rec_val:.4f}, Acc={acc_val:.4f}")

    # 최고 모델 저장
    if f1_val > best_f1:
        best_f1 = f1_val
        ckpt = {
            "cfg": {
                "d_in": D_IN,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
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
# 5️⃣ 평가 함수
# =====================
def evaluate(loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb).argmax(dim=-1)
            y_true.extend(yb.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())
    return y_true, y_pred

# =====================
# 6️⃣ Validation & Test 평가
# =====================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])

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
# 7️⃣ 콘솔 출력
# =====================
print("\n=== Validation Results ===")
for k, v in val_metrics.items():
    print(f"{k.capitalize():<10}: {v:.4f}")

print("\n=== Test Results ===")
for k, v in test_metrics.items():
    print(f"{k.capitalize():<10}: {v:.4f}")

# =====================
# 8️⃣ W&B Summary 등록 (parallel coordinates 표시용)
# =====================
# ⚠️ 반드시 finish() 전에 실행되어야 함
wandb.run.summary["final_val_precision"] = val_metrics["precision"]
wandb.run.summary["final_val_recall"] = val_metrics["recall"]
wandb.run.summary["final_val_f1"] = val_metrics["f1"]
wandb.run.summary["final_val_accuracy"] = val_metrics["accuracy"]

wandb.run.summary["final_test_precision"] = test_metrics["precision"]
wandb.run.summary["final_test_recall"] = test_metrics["recall"]
wandb.run.summary["final_test_f1"] = test_metrics["f1"]
wandb.run.summary["final_test_accuracy"] = test_metrics["accuracy"]

# =====================
# 9️⃣ CSV 및 상세 리포트 저장
# =====================
df = pd.DataFrame([
    {"split": "Validation", **val_metrics},
    {"split": "Test", **test_metrics}
])
df.to_csv(REPORT_PATH, index=False)
print(f"\n📊 Report saved to {REPORT_PATH}")

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
# 🔟 그래프 시각화
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

# ✅ 반드시 호출 (summary 확정)
wandb.finish()
