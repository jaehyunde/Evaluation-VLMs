import os
import pandas as pd

# === 경로 설정 ===
FEATURE_DIR = "test/testfeatures"
LABEL_DIR = "test/testlabels"
OUTPUT_PATH = "test/index_test.csv"

os.makedirs("data", exist_ok=True)

# === 파일 목록 수집 ===
feature_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")])
rows = []

for f in feature_files:
    clip_id = f.replace(".mp4_features.npy", "")
    feature_path = os.path.join(FEATURE_DIR, f)
    label_path = os.path.join(LABEL_DIR, f"{clip_id}.csv")

    if not os.path.exists(label_path):
        print(f"⚠️ 라벨 없음: {label_path}")
        continue

    rows.append({
        "clip": clip_id,
        "feature_path": feature_path,
        "label_path": label_path
    })

# === 저장 ===
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ index_all.csv 생성 완료 ({len(rows)}개 클립)")
    print(f"📁 저장 경로: {OUTPUT_PATH}")
else:
    print("❌ 일치하는 feature-label 쌍을 찾지 못했습니다.")
