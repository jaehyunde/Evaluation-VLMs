import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
# [수정] 분석할 파일 리스트에 "_new8class" 추가
file_list = ['edab_new8class', 'video41_new8class', 'video42_new8class', 'video43_new8class']

# 입력 파일의 경로 (testdata)
input_dir = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

# [수정] 분석 대상 클래스에 'nogesture' 추가
target_classes = [
    'emblematic', 'indexing', 'representing', 'molding', 
    'acting', 'drawing', 'beat', 'other', 'nogesture'
]

# [수정] 저장 경로 변경
output_dir = "result/distribution/withnogesture"
os.makedirs(output_dir, exist_ok=True)

def get_clean_label(label):
    if pd.isna(label): return None
    # .lower()를 통해 대소문자 구분 없이 인식 가능하게 처리
    return str(label).lower().strip().replace(".", "")

# -------------------------------------------------------
# 2. 데이터 수집 및 집계
# -------------------------------------------------------
print(f"📊 Analyzing distribution for Ground Truth data in: {input_dir}")

counts = {cls: 0 for cls in target_classes}
outliers = {}
total_valid = 0

for video in file_list:
    file_path = os.path.join(input_dir, f"{video}.csv")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️ Warning: {file_path} not found.")
        continue
        
    df = pd.read_csv(file_path)
    if 'label' not in df.columns:
        continue
        
    for raw_val in df['label']:
        label = get_clean_label(raw_val)
        
        # target_classes 내에 있는 경우만 카운트
        if label in target_classes:
            counts[label] += 1
            total_valid += 1
        else:
            # 그 외의 값은 Outlier 처리
            outlier_name = label if label else "None/Empty"
            outliers[outlier_name] = outliers.get(outlier_name, 0) + 1

# 결과 데이터 구성 (Source 명칭을 GT_Data로 표기)
distribution_data = []
row = {"Source": "GT_Data"}
row.update(counts)
row["Valid_Samples"] = total_valid
row["Outlier_Details"] = ", ".join([f"{k}: {v}" for k, v in outliers.items()]) if outliers else "None"
distribution_data.append(row)

df_dist = pd.DataFrame(distribution_data)

# -------------------------------------------------------
# 3. 터미널 출력 및 CSV 저장
# -------------------------------------------------------
print("\n" + "="*150)
print(f"{'Ground Truth Label Distribution Summary':^150}")
print("="*150)
print(df_dist.to_string(index=False))
print("="*150)

# [수정] 파일 이름: distgt.csv
csv_path = os.path.join(output_dir, "distgt.csv")
df_dist.to_csv(csv_path, index=False)
print(f"🚀 CSV saved to: {csv_path}")

# -------------------------------------------------------
# 4. 막대그래프 시각화
# -------------------------------------------------------
plot_cols = ["Source"] + target_classes
df_plot = df_dist[plot_cols].melt(id_vars='Source', var_name='Label', value_name='Anzahl')

plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# 막대 그래프
ax = sns.barplot(data=df_plot, x='Label', y='Anzahl', color='grey')

plt.title('Klassenverteilung des Datensatzes', fontsize=20, pad=20)
plt.xlabel('Label Name', fontsize=16)
plt.ylabel('Anzahl', fontsize=16)
plt.xticks(rotation=45)

# Y축 25 단위 설정
max_count = df_plot['Anzahl'].max() if not df_plot.empty else 0
plt.yticks(range(0, int(max_count) + 50, 25))

plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18, rotation=0)

# 막대 상단 수치 기입
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=20, fontweight='normal')

plt.tight_layout()

# [수정] 파일 이름: distgt.png
graph_path = os.path.join(output_dir, "distgt.png")
plt.savefig(graph_path)
plt.close()

print(f"✅ Bar chart saved to: {graph_path}")
