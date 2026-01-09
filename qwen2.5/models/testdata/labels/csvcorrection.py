import pandas as pd
import os

# 원본 파일 경로
input_path = "2Clinician_id1_vid1.csv"

# CSV 불러오기
df = pd.read_csv(input_path)

# 1️⃣ index 열 추가 (0부터 시작)
df.insert(0, "index", range(len(df)))

# 2️⃣ start_t, end_t 열 제거 (존재할 경우만)
df = df.drop(columns=["start_t", "end_t"], errors="ignore")

# 3️⃣ 파일 이름 변경 후 저장
base, ext = os.path.splitext(input_path)
output_path = f"{base}_index{ext}"
df.to_csv(output_path, index=False)

print(f"✅ 파일 저장 완료: {output_path}")
