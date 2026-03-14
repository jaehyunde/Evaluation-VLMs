import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 환경 설정 및 경로 정의
# [수정] 저장 위치 정의
save_path = os.path.expanduser('~/Jayproject/result/llmbinary')
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_list = ['edab_newbinary', 'video41_newbinary', 'video42_newbinary', 'video43_newbinary']

models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output')
}
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

target_classes = ['gesture', 'nogesture']

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
            
            if raw_pred not in target_classes:
                outlier_reports.append({
                    "Model": model_name,
                    "Video": video,
                    "Index": row['index'],
                    "Label": row['label_pred']
                })
                model_outlier_count += 1
            
            if raw_gt in target_classes and raw_pred in target_classes:
                all_y_true.append(raw_gt)
                all_y_pred.append(raw_pred)

    # 3. 모든 지표 계산
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
# 4. 보고서 출력 및 저장
final_df = pd.DataFrame(overall_results)
col_order = ["Model", "Valid_Samples", "Outliers", "Accuracy", "Prec(m)", "Rec(m)", "F1(m)", "Prec(w)", "Rec(w)", "F1(w)"]
final_df = final_df[col_order]

print("\n" + "="*140)
print(final_df.to_string(index=False, float_format="{:.4f}".format))
# [수정] 지정된 경로에 CSV 저장
final_df.to_csv(os.path.join(save_path, "final_summary_report.csv"), index=False)
print("="*140)
print(f"🚀 Results saved to '{save_path}'")
