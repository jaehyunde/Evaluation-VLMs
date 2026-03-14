import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
# [수정 1] file_list에서 "_new8class" 제거
file_list = ['edab', 'video41', 'video42', 'video43']

# [수정 2] models_info path 끝에 "/pure8class" 추가
models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse/pure8class'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output/pure8class'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output/pure8class')
}

# [수정 3] target_classes에서 'nogesture' 제거 (순수 8종)
target_classes = [
    'emblematic', 'indexing', 'representing', 'molding', 
    'acting', 'drawing', 'beat', 'other'
]

# [수정 5] 저장 경로 변경
output_dir = "result/distribution/pure"
os.makedirs(output_dir, exist_ok=True)

def get_clean_label(label):
    if pd.isna(label): return None
    return str(label).lower().strip().replace(".", "")

# -------------------------------------------------------
# 2. 데이터 수집 및 집계
# -------------------------------------------------------
distribution_data = []

for model_name, pred_dir in models_info.items():
    print(f"📊 Analyzing pure distribution for: {model_name}...")
    
    counts = {cls: 0 for cls in target_classes}
    outliers = {}
    total_valid = 0
    
    for video in file_list:
        pred_file = os.path.join(pred_dir, f"{video}.csv")
        
        if not os.path.exists(pred_file):
            print(f"  ⚠️ Warning: {pred_file} not found.")
            continue
            
        df = pd.read_csv(pred_file)
        if 'label' not in df.columns:
            continue
            
        for raw_val in df['label']:
            label = get_clean_label(raw_val)
            
            # 정제된 라벨이 8종 안에 있는 경우만 카운트
            if label in target_classes:
                counts[label] += 1
                total_valid += 1
            else:
                # nogesture를 포함한 모든 예외 라벨을 Outlier로 처리
                outlier_name = label if label else "None/Empty"
                outliers[outlier_name] = outliers.get(outlier_name, 0) + 1
    
    # 행 데이터 구성
    row = {"Model": model_name}
    row.update(counts)
    row["Valid_Samples"] = total_valid
    row["Outlier_Details"] = ", ".join([f"{k}: {v}" for k, v in outliers.items()]) if outliers else "None"
    
    distribution_data.append(row)

# DataFrame 생성
df_dist = pd.DataFrame(distribution_data)

# -------------------------------------------------------
# 3. 결과 출력 및 CSV 저장
# -------------------------------------------------------
print("\n" + "="*150)
print(f"{'Pure 8-Class Label Distribution Summary':^150}")
print("="*150)
print(df_dist.to_string(index=False))
print("="*150)

# [수정 4] 저장 파일 이름 변경: distribution_pure.csv
csv_path = os.path.join(output_dir, "distribution_pure.csv")
df_dist.to_csv(csv_path, index=False)
print(f"🚀 CSV saved to: {csv_path}")

# -------------------------------------------------------
# 4. 막대그래프 시각화
# -------------------------------------------------------
# 시각화용 데이터 정제 (추가 정보 제외)
plot_cols = ["Model"] + target_classes
df_plot = df_dist[plot_cols].melt(id_vars='Model', var_name='Label', value_name='Anzahl')

plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_plot, x='Label', y='Anzahl', hue='Model')

plt.title('Pure 8-Gesture Distribution by Model', fontsize=16, pad=20)
plt.xlabel('Label Name', fontsize=15)
plt.ylabel('Anzahl', fontsize=15)
plt.xticks(rotation=45)

# Y축 25 단위 설정 유지
max_count = df_plot['Count'].max() if not df_plot.empty else 0
plt.yticks(range(0, int(max_count) + 50, 25))

# 막대 상단 카운트 기입
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9, fontweight='bold')

plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# [수정 4] 저장 파일 이름 변경: distributionpure.png
graph_path = os.path.join(output_dir, "distributionpure.png")
plt.savefig(graph_path)
plt.close()

print(f"✅ Bar chart saved to: {graph_path}")
