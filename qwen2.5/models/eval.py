import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --------------------------------
# 설정
# --------------------------------
cls_list = ["8class", "binary"]
avg_list = ["macro", "weighted"]

root_list = {
    "Qwen2.5-VL": "qwen2.5/models/ergebnisse",
    "LLaVA-OneVision": "LLaVA-OneVision/output",
    "LLaVA-NeXT": "LLaVA-NeXT/output",
}

video_bases1 = ["edab_new", "video41_new", "video42_new", "video43_new"]
video_bases = ["angle_3"]

BASE_PATH = "/home/stud_homes/s6010479/Jayproject"

# --------------------------------
# helper
# --------------------------------
def load_and_merge(true_path: str, pred_path: str) -> pd.DataFrame:
    true_df = pd.read_csv(true_path)
    pred_df = pd.read_csv(pred_path)

    merged = pd.merge(
        true_df,
        pred_df,
        on="index",
        suffixes=("_true", "_pred")
    )
    return merged


# --------------------------------
# evaluation loop
# --------------------------------
for model_name, root in root_list.items():
    print("\n" + "#" * 80)
    print(f"MODEL: {model_name}  |  ROOT: {root}")
    print("#" * 80)

    for cls in cls_list:
        # true / pred 파일 리스트
        true_files = [
            f"testdata/labels/edab_new{cls}.csv"
            for v in video_bases
        ]

        pred_files = [
            f"{BASE_PATH}/{root}/{v}{cls}.csv"
            for v in video_bases
        ]

        assert len(true_files) == len(pred_files), "true / pred 파일 개수가 다릅니다."

        # merge & concat
        merged_list = []
        for t_path, p_path in zip(true_files, pred_files):
            merged_list.append(load_and_merge(t_path, p_path))

        merged_all = pd.concat(merged_list, ignore_index=True)

        # 라벨 정리
        y_true = merged_all["label_true"].astype(str).str.strip()
        y_pred = (
    merged_all["label_pred"].astype(str)
    .str.strip()
    .replace({"nogesture": "NoGesture"})
)

        print("\n" + "-" * 70)
        print(f"[CLS: {cls}]  samples={len(merged_all)}  videos={len(video_bases)}")

        # metrics
        for art in avg_list:
            f1 = f1_score(y_true, y_pred, average=art, zero_division=0)
            precision = precision_score(y_true, y_pred, average=art, zero_division=0)
            recall = recall_score(y_true, y_pred, average=art, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            print(f"\n  [average = {art}]")
            print(f"    f1        : {f1:.4f}")
            print(f"    precision : {precision:.4f}")
            print(f"    recall    : {recall:.4f}")
            print(f"    accuracy  : {accuracy:.4f}")
