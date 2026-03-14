import pandas as pd

videoname =  "edab"
# 파일 경로
input_path = f"{videoname}.csv"

# CSV 파일 불러오기
df = pd.read_csv(input_path)

# index, label 열만 남기기 (존재할 경우만)
columns_to_keep = [col for col in ["index", "label"] if col in df.columns]
df = df[columns_to_keep]

# 결과 저장 (덮어쓰지 않으려면 파일명 변경 가능)
output_path = f"{videoname}pure.csv"
df.to_csv(output_path, index=False)

print(f"✅ 정리 완료! 저장된 파일: {output_path}")
