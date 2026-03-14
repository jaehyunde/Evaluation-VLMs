import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
video_list = ['angle_1', 'angle_2', 'angle_3']
# [수정] 8class와 binary 두 가지 카테고리 처리를 위한 모드 설정
modes = ["8class", "binary"]

gt_root = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels')
pred_root = os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse')
save_base_dir = os.path.expanduser('~/Jayproject/result/angles')
os.makedirs(save_base_dir, exist_ok=True)

def get_clean_label(label):
    if pd.isna(label): return 'other'
    val = str(label).lower().strip().replace(".", "")
    return val

# -------------------------------------------------------
# 2. 분석 루프 시작 (8class, binary 순차 실행)
# -------------------------------------------------------
for mode in modes:
    cls_suffix = mode
    # 각 카테고리별로 결과 폴더를 분리하여 저장
    save_dir = os.path.join(save_base_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    
    if mode == "8class":
        target_classes = [
            'emblematic', 'indexing', 'representing', 'molding', 
            'acting', 'drawing', 'beat', 'other', 'nogesture'
        ]
    else: # binary 모드
        target_classes = ['gesture', 'nogesture']

    total_y_true = []
    total_y_pred = []
    summary_results = []

    print(f"\n🚀 Starting {mode.upper()} Evaluation... Output directory: {save_dir}")
    print("-" * 175)

    for video in video_list:
        gt_file = os.path.join(gt_root, f"{video}{cls_suffix}.csv")
        pred_file = os.path.join(pred_root, f"{video}{cls_suffix}.csv")
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"  ⚠️ Warning: Missing files for {video} (Mode: {mode}). Skipping...")
            continue
            
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
        
        # index 기준 병합
        df_merged = pd.merge(df_gt, df_pred, on='index', how='inner', suffixes=('_gt', '_pred'))
        
        y_true_vid = []
        y_pred_vid = []
        valid_count = 0
        invalid_count = 0
        
        for _, row in df_merged.iterrows():
            raw_gt = get_clean_label(row['label_gt'])
            raw_pred = get_clean_label(row['label_pred'])
            
            # [수정] binary 모드일 때 8종 제스처 라벨이 섞여있을 경우 'gesture'로 통합
            gesture_subclasses = ['emblematic', 'indexing', 'representing', 'molding', 'acting', 'drawing', 'beat', 'other']
            if mode == "binary":
                if raw_gt in gesture_subclasses: raw_gt = 'gesture'
                if raw_pred in gesture_subclasses: raw_pred = 'gesture'

            # GT 처리: 유효하지 않으면 other
            clean_gt = raw_gt if raw_gt in target_classes else 'other'
            
            # Pred 처리: 유효하지 않으면 invalid 카운트 후 other 치환
            if raw_pred not in target_classes:
                invalid_count += 1
                clean_pred = 'other'
            else:
                valid_count += 1
                clean_pred = raw_pred
                
            y_true_vid.append(clean_gt)
            y_pred_vid.append(clean_pred)
        
        total_y_true.extend(y_true_vid)
        total_y_pred.extend(y_pred_vid)
        
        # 지표 계산
        metrics = {
            "Video": video,
            "Total_Samples": len(y_true_vid),
            "Valid_Samples": valid_count,
            "Invalid_Samples": invalid_count,
            "Accuracy": accuracy_score(y_true_vid, y_pred_vid),
            "Prec(m)": precision_score(y_true_vid, y_pred_vid, average='macro', zero_division=0),
            "Rec(m)": recall_score(y_true_vid, y_pred_vid, average='macro', zero_division=0),
            "F1(m)": f1_score(y_true_vid, y_pred_vid, average='macro', zero_division=0),
            "Prec(w)": precision_score(y_true_vid, y_pred_vid, average='weighted', zero_division=0),
            "Rec(w)": recall_score(y_true_vid, y_pred_vid, average='weighted', zero_division=0),
            "F1(w)": f1_score(y_true_vid, y_pred_vid, average='weighted', zero_division=0)
        }
        summary_results.append(metrics)
        
        # 개별 CSV 및 CM 저장
        pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, f"{video}_{mode}.csv"), index=False)
        cm = confusion_matrix(y_true_vid, y_pred_vid, labels=target_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes)
        plt.title(f'Confusion Matrix ({mode}) - {video}')
        plt.savefig(os.path.join(save_dir, f"{video}_cm.png"))
        plt.close()
        
        print(f"  ✅ {video} | Valid: {valid_count}, Invalid: {invalid_count} | Acc: {metrics['Accuracy']:.4f}")

    # -------------------------------------------------------
    # 3. 전체(TOTAL) 통계 및 분포 시각화 (기능 추가)
    # -------------------------------------------------------
    if total_y_true:
        # [신규] 분포 요약 출력
        print(f"\n📊 {mode.upper()} Label Distribution Summary")
        print("-" * 60)
        gt_dist = pd.Series(total_y_true).value_counts().reindex(target_classes, fill_value=0)
        pred_dist = pd.Series(total_y_pred).value_counts().reindex(target_classes, fill_value=0)
        dist_df = pd.DataFrame({'Label': target_classes, 'Ground_Truth': gt_dist.values, 'Prediction': pred_dist.values})
        print(dist_df.to_string(index=False))
        print("-" * 60)

        # [신규] 분포 그래프 저장 (distribution.png)
        plt.figure(figsize=(12, 6))
        dist_melted = dist_df.melt(id_vars='Label', var_name='Type', value_name='Count')
        sns.barplot(data=dist_melted, x='Label', y='Count', hue='Type', palette='Blues_d')
        plt.title(f'Label Distribution ({mode.upper()})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution.png"))
        plt.close()

        # 전체 지표 요약
        total_metrics = {
            "Video": "TOTAL",
            "Total_Samples": len(total_y_true),
            "Valid_Samples": sum(r['Valid_Samples'] for r in summary_results),
            "Invalid_Samples": sum(r['Invalid_Samples'] for r in summary_results),
            "Accuracy": accuracy_score(total_y_true, total_y_pred),
            "Prec(m)": precision_score(total_y_true, total_y_pred, average='macro', zero_division=0),
            "Rec(m)": recall_score(total_y_true, total_y_pred, average='macro', zero_division=0),
            "F1(m)": f1_score(total_y_true, total_y_pred, average='macro', zero_division=0),
            "Prec(w)": precision_score(total_y_true, total_y_pred, average='weighted', zero_division=0),
            "Rec(w)": recall_score(total_y_true, total_y_pred, average='weighted', zero_division=0),
            "F1(w)": f1_score(total_y_true, total_y_pred, average='weighted', zero_division=0)
        }
        summary_results.append(total_metrics)
        
        # 전체 CM 저장
        cm_total = confusion_matrix(total_y_true, total_y_pred, labels=target_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes)
        plt.title(f'Overall Confusion Matrix ({mode.upper()})')
        plt.savefig(os.path.join(save_dir, "total_cm.png"))
        plt.close()

        final_df = pd.DataFrame(summary_results)
        print("\n" + "="*175)
        print(f"{'Final Performance Summary Report (' + mode.upper() + ')':^175}")
        print("="*175)
        print(final_df.to_string(index=False, float_format="{:.4f}".format))
        print("="*175)
        
        final_df.to_csv(os.path.join(save_dir, "total_summary_report.csv"), index=False)
        print(f"🚀 {mode.upper()} Results saved to: {save_dir}")
