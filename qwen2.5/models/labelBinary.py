
import pandas as pd
import numpy as np

filename = "edab_new"

# 1. CSV 파일 불러오기
file_path = f'testdata/labels/{filename}.csv'
df = pd.read_csv(file_path)

# 2. label 컬럼 NaN → 'NoGesture'
df['label'] = df['label'].fillna('NoGesture')

# 3. Gesture / NoGesture 이진화
df['label'] = df['label'].apply(lambda x: 'NoGesture' if x == 'NoGesture' else 'Gesture')

# 4. 필요한 열만 남기기 (index, label)
# 만약 파일에 index가 없으면 자동 생성해줄 수도 있음 → 원하시면 추가해줄게요
columns_to_keep = [col for col in ['index', 'label'] if col in df.columns]
df = df[columns_to_keep]

# 5. 새 CSV 저장
output_path = f'testdata/labels/{filename}binary.csv'
df.to_csv(output_path, index=False)

print("✅ 변환 완료!")
print(f"📁 saved in {output_path}")
print("\n🔍 최종 미리보기:")
print(df.head())
