import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 환경 설정 및 경로 정의
file_list = ['edab_new8class', 'video41_new8class', 'video42_new8class', 'video43_new8class']
models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output')
}
gt_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')

target_classes = ['emblematic', 'indexing', 'representing', 'molding', 'acting', 'drawing', 'beat', 'other', 'nogesture']

def get_clean_label(label):
    if pd.isna(label): return None
    return str(label).lower().strip().replace(".", "")

overall_results = []
outlier_reports = [] 

for model_name, pred_dir in models_info.items():
    all_y_true, all_y_pred = [], []
    model_outlier_count = 0
    
    print(f"🔍 Processing Model: {model_name}...")
    
    for video in file_list:
        gt_file = os.path.join(video, f"{video}.csv") # 원본 코드 유지 (단, gt_path 활용이 필요할 수 있음)
        gt_file = os.path.join(gt_path, f"{video}.csv")
        pred_file = os.path.join(pred_dir, f"{video}.csv")
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"  ⚠️ Warning: File missing for {video}")
            continue
            
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
        df_merged = pd.merge(df_gt, df_pred, on='index', how='left', suffixes=('_gt', '_pred'))
        
        for _, row in df_merged.iterrows():
            raw_gt = get_clean_label(row['label_gt'])
            raw_pred = get_clean_label(row['label_pred'])
            
            # [수정] 유효하지 않은 예측값 처리 로직
            if raw_pred not in target_classes:
                final_pred = 'other' # 유효하지 않으면 'other'로 치환
                outlier_reports.append({"Model": model_name, "Video": video, "Index": row['index'], "Label": row['label_pred']})
                model_outlier_count += 1
            else:
                final_pred = raw_pred
            
            # [수정] 모든 샘플을 지표 계산 리스트에 추가 (단, GT가 유효할 때)
            if raw_gt in target_classes:
                all_y_true.append(raw_gt)
                all_y_pred.append(final_pred)

    # 데이터가 수집된 경우에만 지표 계산
    if len(all_y_true) > 0:
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
        
        cm = confusion_matrix(all_y_true, all_y_pred, labels=target_classes)
        plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=target_classes, yticklabels=target_classes)
        plt.title(f'CM - {model_name}'); plt.savefig(f"{model_name}_cm.png"); plt.close()
    else:
        print(f"  ❌ No valid data for {model_name}.")

# 5. 결과 요약 및 저장
print("\n" + "="*140)
if overall_results:
    final_df = pd.DataFrame(overall_results)
    col_order = ["Model", "Valid_Samples", "Outliers", "Accuracy", "Prec(m)", "Rec(m)", "F1(m)", "Prec(w)", "Rec(w)", "F1(w)"]
    available_cols = [c for c in col_order if c in final_df.columns]
    final_df = final_df[available_cols]

    print(f"{'Final Performance Summary (Invalid as Other)':^140}")
    print("="*140)
    print(final_df.to_string(index=False, float_format="{:.4f}".format))
    final_df.to_csv("final_summary_report.csv", index=False)
else:
    print("❌ No results to display.")
print("="*140)
