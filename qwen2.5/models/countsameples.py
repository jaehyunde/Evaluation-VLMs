import os
import pandas as pd

# 1. 환경 설정 및 경로 정의
gt_list = ['edab_new8class', 'video41_new8class', 'video42_new8class', 'video43_new8class']
# 사용자 홈 디렉토리(~)를 포함한 경로 확장
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse')

print(f"{'File Name':<25} | {'Total Samples':>15}")
print("-" * 43)

total_all_samples = 0

# 2. 파일 순회 및 샘플 수 계산
for file_name in gt_list:
    file_path = os.path.join(gt_path, f"{file_name}.csv")
    
    if os.path.exists(file_path):
        try:
            # CSV 로드
            df = pd.read_csv(file_path)
            sample_count = len(df)
            total_all_samples += sample_count
            print(f"{file_name:<25} | {sample_count:>15,}")
        except Exception as e:
            print(f"{file_name:<25} | Error reading file")
    else:
        print(f"{file_name:<25} | File not found")

print("-" * 43)
print(f"{'TOTAL SUM':<25} | {total_all_samples:>15,}")
