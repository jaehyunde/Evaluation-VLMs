import os
import pandas as pd

# 1. 대상 폴더 설정
target_dir = os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse')

# 수정된 파일 목록을 저장할 리스트
modified_files = []

# 2. 폴더 내 모든 파일 순회
if not os.path.exists(target_dir):
    print(f"❌ 폴더를 찾을 수 없습니다: {target_dir}")
else:
    file_list = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    print(f"📂 총 {len(file_list)}개의 CSV 파일을 검사합니다...\n")

    for filename in file_list:
        file_path = os.path.join(target_dir, filename)
        
        try:
            # CSV 로드
            df = pd.read_csv(file_path)
            
            if 'index' in df.columns and len(df) > 0:
                # 첫 번째 인덱스 값이 1인지 확인
                first_val = df.iloc[0]['index']
                
                if first_val == 1:
                    # 모든 인덱스 값에서 1을 뺌
                    df['index'] = df['index'] - 1
                    
                    # 원본 파일에 덮어쓰기 (overwrite)
                    df.to_csv(file_path, index=False)
                    modified_files.append(filename)
                    print(f"✅ 수정 완료: {filename} (1 -> 0 시작)")
                else:
                    # 0으로 시작하거나 다른 경우 스킵
                    pass
        
        except Exception as e:
            print(f"⚠️ 에러 발생 ({filename}): {e}")

# 3. 결과 요약 출력
print("\n" + "="*50)
print(f"📊 작업 요약")
print("-"*50)
if modified_files:
    print(f"✨ 인덱스가 1로 시작되어 수정된 파일 ({len(modified_files)}개):")
    for f in modified_files:
        print(f"  - {f}")
else:
    print("검사 결과, 수정한 파일이 없습니다 (모두 0으로 시작함).")
print("="*50)
