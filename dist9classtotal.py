import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
file_list = ['edab_new8class', 'video41_new8class', 'video42_new8class', 'video43_new8class']

models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output')
}

# [추가] Ground Truth 경로 정의
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

target_classes = [
    'emblematic', 'indexing', 'representing', 'molding', 
    'acting', 'drawing', 'beat', 'other', 'nogesture'
]

output_dir = "result/distribution/withnogesture/"
os.makedirs(output_dir, exist_ok=True)

def get_clean_label(label):
    if pd.isna(label): return None
    return str(label).lower().strip().replace(".", "")

# -------------------------------------------------------
# 2. 데이터 수집 및 집계
# -------------------------------------------------------
distribution_data = []

# --- [추가] Ground Truth 분포 먼저 계산 ---
print(f"📊 Analyzing distribution for: Ground Truth...")
gt_counts = {cls: 0 for cls in target_classes}
gt_total_valid = 0

for video in file_list:
    gt_file = os.path.join(gt_path, f"{video}.csv")
    
    if os.path.exists(gt_file):
        df_gt = pd.read_csv(gt_file)
        if 'label' in df_gt.columns:
            for raw_val in df_gt['label']:
                label = get_clean_label(raw_val)
                if label in target_classes:
                    gt_counts[label] += 1
                    gt_total_valid += 1

# GT 행 추가
gt_row = {"Model": "Ground Truth"}
gt_row.update(gt_counts)
gt_row["Valid_Samples"] = gt_total_valid
gt_row["Outlier_Details"] = "None"
distribution_data.append(gt_row)

# --- 기존 모델들의 예측 데이터 집계 ---
for model_name, pred_dir in models_info.items():
    print(f"📊 Analyzing distribution for: {model_name}...")
    
    counts = {cls: 0 for cls in target_classes}
    outliers = {}
    total_valid = 0
    
    for video in file_list:
        pred_file = os.path.join(pred_dir, f"{video}.csv")
        
        if not os.path.exists(pred_file):
            continue
            
        df = pd.read_csv(pred_file)
        if 'label' not in df.columns:
            continue
            
        for raw_val in df['label']:
            label = get_clean_label(raw_val)
            
            if label in target_classes:
                counts[label] += 1
                total_valid += 1
            else:
                outlier_name = label if label else "None/Empty"
                outliers[outlier_name] = outliers.get(outlier_name, 0) + 1
    
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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*150)
print(f"{'GT vs Predicted Label Distribution':^150}")
print("="*150)
print(df_dist.to_string(index=False))
print("="*150)

csv_path = os.path.join(output_dir, "distribution_9class.csv")
df_dist.to_csv(csv_path, index=False)
print(f"🚀 CSV saved to: {csv_path}")

# -------------------------------------------------------
# 4. 막대그래프 시각화
# -------------------------------------------------------
plot_cols = ["Model"] + target_classes
df_plot = df_dist[plot_cols].melt(id_vars='Model', var_name='Label', value_name='Count')

plt.figure(figsize=(18, 9)) # 그래프 크기를 조금 더 키움
sns.set_style("whitegrid")

# 막대 그래프 생성 (hue에 Model을 넣어 GT와 각 모델이 나란히 표시됨)
ax = sns.barplot(data=df_plot, x='Label', y='Count', hue='Model')

plt.title('Modellvorhersagen vs. Ground Truth (9-Klassen)', fontsize=20, pad=20)
plt.xlabel('Label Name', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(rotation=45,fontsize=16)

# Y축 눈금 및 수치 표시 (25 단위 유지)
max_count = df_plot['Count'].max()
plt.yticks(range(0, int(max_count) + 50, 25),fontsize=16)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 12), 
                    textcoords='offset points',
                    fontsize=16, fontweight='normal')

plt.legend(title='Category')
plt.tight_layout()

graph_path = os.path.join(output_dir, "distribution_9class.png")
plt.savefig(graph_path)
plt.close()

print(f"✅ Bar chart with GT saved to: {graph_path}")
