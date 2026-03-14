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
# 2. 데이터 수집 및 집계 (CSV 저장용 정보 추가)
# -------------------------------------------------------
distribution_data = []

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
    
    # [추가 기능] CSV 행 데이터 구성
    row = {"Model": model_name}
    row.update(counts) # 각 라벨별 카운트 추가
    row["Valid_Samples"] = total_valid # 지표 계산에 사용된 총 샘플 수
    # 이상치 종류와 개수를 문자열 형태로 저장 (예: 'error: 5, failed: 2')
    row["Outlier_Details"] = ", ".join([f"{k}: {v}" for k, v in outliers.items()]) if outliers else "None"
    
    distribution_data.append(row)

# DataFrame 생성
df_dist = pd.DataFrame(distribution_data)

# -------------------------------------------------------
# 3. 결과 출력 및 CSV 저장
# -------------------------------------------------------
# 터미널 출력 (가독성을 위해 모든 컬럼 표시)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*150)
print(f"{'Predicted Label Distribution & Additional Info':^150}")
print("="*150)
print(df_dist.to_string(index=False))
print("="*150)

# CSV 저장 (추가 정보 포함)
csv_path = os.path.join(output_dir, "distribution_with_nogesture.csv")
df_dist.to_csv(csv_path, index=False)
print(f"🚀 CSV with extra info saved to: {csv_path}")

# -------------------------------------------------------
# 4. 막대그래프 시각화 (기존 동일)
# -------------------------------------------------------
# 시각화 시에는 추가 정보(Valid_Samples, Outlier_Details) 제외하고 멜트(Melt) 처리
plot_cols = ["Model"] + target_classes
df_plot = df_dist[plot_cols].melt(id_vars='Model', var_name='Label', value_name='Count')

plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_plot, x='Label', y='Count', hue='Model')

plt.title('Verteilung der Modellvorhersagen in der 9-Klassen-Konfiguration', fontsize=16, pad=20)
plt.xlabel('Label Name', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)

# Y축 25 단위 설정
max_count = df_plot['Count'].max()
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

graph_path = os.path.join(output_dir, "distributionwithnogesture.png")
plt.savefig(graph_path)
plt.close()

print(f"✅ Bar chart saved to: {graph_path}")
