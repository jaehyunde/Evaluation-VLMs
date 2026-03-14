import os
import pandas as pd

# 1. 환경 설정 (기존 코드와 동일)
file_list = ['edab_new8class', 'video41_new8class', 'video42_new8class', 'video43_new8class']
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

target_classes = [
    'emblematic', 'indexing', 'representing', 'molding', 
    'acting', 'drawing', 'beat', 'other', 'nogesture'
]

def get_clean_label(label):
    if pd.isna(label): return "EMPTY_CELL" # 비어있는 셀 처리
    return str(label).lower().strip().replace(".", "")

print("="*80)
print(f"{'File Name':<20} | {'Index':<8} | {'Original GT Label':<20} | {'Reason'}")
print("-" * 80)

total_gt_rows = 0
invalid_gt_found = 0

for video in file_list:
    gt_file = os.path.join(gt_path, f"{video}.csv")
    
    if not os.path.exists(gt_file):
        print(f"⚠️ {video}.csv 파일을 찾을 수 없습니다.")
        continue
        
    df_gt = pd.read_csv(gt_file)
    total_gt_rows += len(df_gt)
    
    for _, row in df_gt.iterrows():
        raw_gt = get_clean_label(row['label'])
        
        # GT 라벨이 유효 클래스에 없는 경우 탐색
        if raw_gt not in target_classes:
            invalid_gt_found += 1
            print(f"{video:<20} | {row['index']:<8} | {str(row['label']):<20} | Not in target_classes")

print("-" * 80)
print(f"📊 총 검사한 GT 행 수: {total_gt_rows}")
print(f"❌ 유효하지 않은 GT 라벨 수: {invalid_gt_found}")
print("="*80)

if invalid_gt_found == 0:
    print("💡 모든 GT 라벨이 정상입니다. 파일 간 인덱스 누락(Merge 실패)을 확인해보세요.")
