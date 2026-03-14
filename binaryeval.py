import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
gt_list = ['edab', 'video41', 'video42', 'video43']
models_info = {
    'qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse'),
    'LLaVA-OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output'),
    'LLaVA-NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output')
}

gt_base_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# 표준 이진 라벨 정의
binary_classes = ['Gesture', 'NoGesture']

def get_standard_label(label):
    """라벨의 공백, 대소문자, 마침표 제거 후 표준화"""
    if pd.isna(label): return None
    val = str(label).strip().lower().replace(".", "")
    if val == 'gesture': return 'Gesture'
    if val == 'nogesture': return 'NoGesture'
    return val # 이상치 식별을 위해 원본값(소문자화) 반환

# -------------------------------------------------------
# 2. 메인 분석 루프
# -------------------------------------------------------
binary_overall_results = []
outlier_details = [] # 이상치 상세 정보를 담을 리스트

for model_name, pred_dir in models_info.items():
    all_y_true = []
    all_y_pred = []
    outlier_count = 0
    
    print(f"\n📊 Analyzing Binary Performance: {model_name}...")
    
    for gt_name in gt_list:
        filename = f"{gt_name}_newbinary.csv"
        gt_file = os.path.join(gt_base_path, filename)
        pred_file = os.path.join(pred_dir, filename)
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            continue
            
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
        df_merged = pd.merge(df_gt, df_pred, on='index', how='inner', suffixes=('_gt', '_pred'))
        
        for _, row in df_merged.iterrows():
            # 정답과 예측값 표준화
            true_label = get_standard_label(row['label_gt'])
            pred_label = get_standard_label(row['label_pred'])
            
            # --- 이상치(Outlier) 체크 ---
            # 예측값이 Gesture/NoGesture가 아니면 이상치로 분류
            if pred_label not in binary_classes:
                outlier_details.append({
                    "Model": model_name,
                    "Video": gt_name,
                    "Index": row['index'],
                    "Label": row['label_pred']
                })
                outlier_count += 1
                continue # 지표 계산에서 제외
            
            # 정답이 유효한 경우에만 지표 리스트에 추가
            if true_label in binary_classes:
                all_y_true.append(true_label)
                all_y_pred.append(pred_label)

    # 3. 지표 계산 (유효 데이터 기준)
    if all_y_true:
        metrics = {
            "Model": model_name,
            "Total_Samples": len(all_y_true) + outlier_count, # 전체 데이터 합계
            "Valid_Samples": len(all_y_true),               # 지표에 사용된 데이터
            "Outliers": outlier_count,                        # 이상치 수
            "Accuracy": accuracy_score(all_y_true, all_y_pred),
            "Prec(m)": precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "Rec(m)": recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "F1(m)": f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
            "Prec(w)": precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            "Rec(w)": recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            "F1(w)": f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
        }
        binary_overall_results.append(metrics)

        # Confusion Matrix 생성
        cm = confusion_matrix(all_y_true, all_y_pred, labels=binary_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=binary_classes, yticklabels=binary_classes)
        plt.title(f'Binary CM - {model_name}')
        plt.savefig(os.path.join(result_dir, f"{model_name}_binary_cm.png"))
        plt.close()

# -------------------------------------------------------
# 4. Strict Filtering 리포트 (이상치 상세 출력)
# -------------------------------------------------------
print("\n" + "!"*30 + " [ BINARY OUTLIER REPORT ] " + "!"*30)
if outlier_details:
    df_out = pd.DataFrame(outlier_details)
    for model in df_out['Model'].unique():
        m_data = df_out[df_out['Model'] == model]
        print(f"\n📌 Model: {model} (Total Outliers: {len(m_data)})")
        for video in m_data['Video'].unique():
            v_data = m_data[m_data['Video'] == video]
            indices = ", ".join([f"#{r['Index']}({r['Label']})" for _, r in v_data.iterrows()])
            print(f"  🎬 {video}: {indices}")
else:
    print("\n✅ No outliers found in binary classification.")

# -------------------------------------------------------
# 5. 최종 결과 요약표 출력 및 저장
# -------------------------------------------------------
if binary_overall_results:
    final_df = pd.DataFrame(binary_overall_results)
    # 컬럼 순서 설정 (합계와 이상치 포함)
    cols = ["Model", "Total_Samples", "Valid_Samples", "Outliers", "Accuracy", 
            "Prec(m)", "Rec(m)", "F1(m)", "Prec(w)", "Rec(w)", "F1(w)"]
    final_df = final_df[cols]
    
    summary_csv = os.path.join(result_dir, "binary_performance_summary.csv")
    final_df.to_csv(summary_csv, index=False)

    print("\n" + "="*160)
    print(f"{'Final Binary Performance Summary':^160}")
    print("="*160)
    print(final_df.to_string(index=False, float_format="{:.4f}".format))
    print("="*160)
    print(f"🚀 결과가 '{result_dir}/' 폴더에 저장되었습니다.")
