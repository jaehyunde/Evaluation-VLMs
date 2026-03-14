import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 환경 설정 및 경로 정의
file_list = ['angle_1','angle_2','angle_3']
models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse/pure8class'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output/pure8class'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output/pure8class')
}
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

# 분석 대상인 8개 순수 제스처 클래스
target_classes = [
    'gesture','nogesture'
]

def get_clean_label(label):
    if pd.isna(label): return None
    return str(label).lower().strip().replace(".", "")

# 2. 분석 루프 시작
overall_results = []
outlier_reports = [] 

for model_name, pred_dir in models_info.items():
    all_y_true = []
    all_y_pred = []
    model_outlier_count = 0
    
    print(f"\n🔍 Processing Model: {model_name}...")
    
    for video in file_list:
        gt_file = os.path.join(gt_path, f"{video}.csv")
        pred_file = os.path.join(pred_dir, f"{video}.csv")
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            continue
            
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
        
        df_merged = pd.merge(df_gt, df_pred, on='index', how='left', suffixes=('_gt', '_pred'))
        
        for _, row in df_merged.iterrows():
            raw_gt = get_clean_label(row['label_gt'])
            raw_pred = get_clean_label(row['label_pred'])
            
            # Strict Filtering: 8개 클래스 이외의 값 탐지
            if raw_pred not in target_classes:
                outlier_reports.append({
                    "Model": model_name,
                    "Video": video,
                    "Index": row['index'],
                    "Label": row['label_pred']
                })
                model_outlier_count += 1
            
            # 지표 계산용 데이터 수집 (GT와 Pred 모두 8개 안에 있을 때만)
            if raw_gt in target_classes and raw_pred in target_classes:
                all_y_true.append(raw_gt)
                all_y_pred.append(raw_pred)

    # 3. 모든 지표 계산 (수정된 부분)
    if all_y_true:
        metrics = {
            "Model": model_name,
            "Valid_Samples": len(all_y_true),
            "Outliers": model_outlier_count,
            "Accuracy": accuracy_score(all_y_true, all_y_pred),
            "Prec(m)": precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "Rec(m)": recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "F1(m)": f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "Prec(w)": precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            "Rec(w)": recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            "F1(w)": f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
        }
        overall_results.append(metrics)
# Confusion Matrix 계산
        cm = confusion_matrix(all_y_true, all_y_pred, labels=target_classes)
        plt.figure(figsize=(12, 10)) # 가독성을 위해 도화지 크기 확대
        # 1. 히트맵 내부 숫자(Annot) 크기 및 스타일 설정
        ax = sns.heatmap(cm, 
                         annot=True, 
                         fmt='d', 
                         cmap='Blues', 
                         xticklabels=target_classes, 
                         yticklabels=target_classes,
                         annot_kws={"size": 28, "weight": "normal"}) # 숫자 크기 확대 및 레귤러 설정

        # 2. 제목 설정 (폰트 크기 확대 및 레귤러)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=20, pad=25, fontweight='normal')

        # 3. 축 이름(Label) 설정
        plt.xlabel('Predicted Label', fontsize=18, labelpad=15, fontweight='normal')
        plt.ylabel('True Label', fontsize=18, labelpad=15, fontweight='normal')

        # 4. 클래스 이름(Ticks) 설정
        plt.xticks(fontsize=18, rotation=45, fontweight='normal')
        plt.yticks(fontsize=18, rotation=0, fontweight='normal')

        # 여백 조정 및 저장
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}_cm.png"), dpi=300) # 고해상도 저장
        plt.close()


# 4. Strict Filtering 보고서 출력
print("\n" + "!"*30 + " [ STRICT FILTERING REPORT ] " + "!"*30)
if outlier_reports:
    df_outliers = pd.DataFrame(outlier_reports)
    for model in df_outliers['Model'].unique():
        print(f"\n📌 Model: {model}")
        model_data = df_outliers[df_outliers['Model'] == model]
        for video in model_data['Video'].unique():
            video_data = model_data[model_data['Video'] == video]
            print(f"  🎬 Video: {video}")
            for _, row in video_data.iterrows():
                print(f"    - Index {row['Index']:03d}: Label -> {row['Label']}")
        print(f"  📊 Total Outliers for {model}: {len(model_data)}개")

# 5. 전체 지표 요약 출력 및 저장 (모든 지표 포함)
print("\n" + "="*140)
print(f"{'Final Performance Summary (8-Class Only)':^140}")
print("="*140)
final_df = pd.DataFrame(overall_results)
# 출력 컬럼 순서 고정
col_order = ["Model", "Valid_Samples", "Outliers", "Accuracy", "Prec(m)", "Rec(m)", "F1(m)", "Prec(w)", "Rec(w)", "F1(w)"]
final_df = final_df[col_order]

print(final_df.to_string(index=False, float_format="{:.4f}".format))
final_df.to_csv("final_summary_report.csv", index=False)
print("="*140)
print("🚀 Results saved to 'final_summary_report.csv' and CM images saved.")
