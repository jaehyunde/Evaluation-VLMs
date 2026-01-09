import os
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from posevit.dataset import PoseSeqDataset
from posevit.model import PoseSeqTransformer as PoseViT
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import wandb

# =====================
# 기본 설정
# =====================
INDEX_PATH_ALL = "test/index_all.csv"
INDEX_PATH_EXT = "test/index_test.csv"
OUT_DIR = "test/outputs"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "ckpts"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

T, S = 32, 16
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.00050701
D_IN = 80

# Parameters
ALPHA = 1.0         # 고정
N_RUNS = 10         # 반복 횟수
BASE_SEED = 42

print(f"🚀 Device: {DEVICE}")
print(f"🔁 Plan: alpha={ALPHA}, run_id=0~{N_RUNS-1}")

# =====================
# 유틸 함수들
# =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mix_datasets(train_df, ext_df, alpha, random_state=42):
    n_ext = len(ext_df)
    n_train = len(train_df)

    if n_ext == 0 or n_train == 0 or alpha <= 0.0:
        return train_df.copy(), ext_df.copy(), 0

    k = int(alpha * n_ext)
    k = max(0, min(k, n_ext, n_train))
    if k == 0:
        return train_df.copy(), ext_df.copy(), 0

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

    return train_new, ext_new, k


def evaluate(loader, model, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).argmax(dim=-1)
            y_true.extend(yb.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())
    return np.array(y_true), np.array(y_pred)


# =====================
# 반복 실행 (run_id=0~9)
# =====================
for run_id in range(N_RUNS):
    seed = BASE_SEED + run_id
    set_seed(seed)

    wandb.init(
        project="RatioTrain",
        config={
            "alpha": ALPHA,
            "run_id": run_id,
            "seed": seed,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        }
    )

    print("\n" + "=" * 60)
    print(f"🔥 ALPHA={ALPHA:.1f}, RUN={run_id}, SEED={seed}")
    print("=" * 60)

    # Step 1. index 로드 & 섞기
    index_all_orig = pd.read_csv(INDEX_PATH_ALL)
    index_ext_orig = pd.read_csv(INDEX_PATH_EXT)

    index_all_mixed, index_ext_mixed, k_exchanged = mix_datasets(
        index_all_orig, index_ext_orig, ALPHA, random_state=seed
    )

    print(f"📂 Train(original): {len(index_all_orig)}, Ext(original): {len(index_ext_orig)}")
    print(f"🔁 Mixed Train: {len(index_all_mixed)}, Mixed Ext: {len(index_ext_mixed)}, exchanged={k_exchanged}")

    # Step 2. train/val split
    train_df, val_df = train_test_split(index_all_mixed, test_size=0.1, random_state=seed)

    tag = f"a{int(ALPHA * 10)}_r{run_id}"
    INDEX_TRAIN_PATH = f"test/index_train_mixed_{tag}.csv"
    INDEX_VAL_PATH = f"test/index_val_mixed_{tag}.csv"
    INDEX_TEST_PATH = f"test/index_test_mixed_{tag}.csv"

    train_df.to_csv(INDEX_TRAIN_PATH, index=False)
    val_df.to_csv(INDEX_VAL_PATH, index=False)
    index_ext_mixed.to_csv(INDEX_TEST_PATH, index=False)

    # Step 3. Dataset / Dataloader
    train_ds = PoseSeqDataset(INDEX_TRAIN_PATH, T=T, S=S)
    val_ds = PoseSeqDataset(INDEX_VAL_PATH, T=T, S=S)
    test_ds = PoseSeqDataset(INDEX_TEST_PATH, T=T, S=S)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Step 4. 모델 정의
    model = PoseViT(d_in=D_IN, d_model=64, nhead=2, num_layers=4, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    CKPT_PATH = os.path.join(OUT_DIR, "ckpts", f"posevit_best_{tag}.pt")

    # Step 5. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"[α={ALPHA:.1f}, run={run_id}] Train {epoch}"):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits.view(-1, 2), yb.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_loader))

        # Validation
        y_true_val, y_pred_val = evaluate(val_loader, model, DEVICE)
        f1_val = f1_score(y_true_val, y_pred_val)
        prec_val = precision_score(y_true_val, y_pred_val)
        rec_val = recall_score(y_true_val, y_pred_val)
        acc_val = accuracy_score(y_true_val, y_pred_val)

        wandb.log({
            "epoch": epoch,
            "val_loss": epoch_loss,
            "val_f1": f1_val,
            "val_precision": prec_val,
            "val_recall": rec_val,
            "val_accuracy": acc_val,
            "alpha": ALPHA,
            "run_id": run_id,
        })

        print(f"[Val] ep{epoch} loss={epoch_loss:.4f} | F1={f1_val:.4f} "
              f"P={prec_val:.4f} R={rec_val:.4f} Acc={acc_val:.4f}")

        if f1_val > best_f1:
            best_f1 = f1_val
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
                    "run_id": run_id,
                },
                "model": model.state_dict()
            }
            torch.save(ckpt, CKPT_PATH)
            print(f"✓ Saved best model: {CKPT_PATH} (Val F1={best_f1:.4f})")

    # Step 6. Test 평가
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    y_true_test, y_pred_test = evaluate(test_loader, model, DEVICE)

    test_metrics = {
        "precision": precision_score(y_true_test, y_pred_test),
        "recall": recall_score(y_true_test, y_pred_test),
        "f1": f1_score(y_true_test, y_pred_test),
        "accuracy": accuracy_score(y_true_test, y_pred_test),
    }

    wandb.log({
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
    })

    print(f"\n📊 Test results (run_id={run_id}): {test_metrics}")

    # 결과 누적 저장
    df_path = os.path.join(OUT_DIR, "reports", "wandb_repeat_all_results.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame()

    new_row = {
        "alpha": ALPHA,
        "run": run_id,
        **test_metrics,
        "seed": seed,
        "k_exchanged": k_exchanged,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(df_path, index=False)

    # 평균/표준편차 갱신
    summary = (
        df.groupby("alpha")
        .agg({"f1": ["mean", "std"], "accuracy": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = ["alpha", "f1_mean", "f1_std", "acc_mean", "acc_std"]

    summary_path = os.path.join(OUT_DIR, "reports", "wandb_repeat_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("✅ Updated Summary:")
    print(summary)

    wandb.finish()

# =====================
# 끝
# =====================
