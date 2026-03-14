import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# 1. 환경 설정 및 경로 정의
# -------------------------------------------------------
video_list = ['angle_1', 'angle_2', 'angle_3']
modes = ["8class", "binary"] # 8class와 binary 두 가지 카테고리 수행

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
    save_dir = os.path.join(save_base_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    
    if mode == "8class":
        target_classes = [
            'emblematic', 'indexing', 'representing', 'molding', 
            'acting', 'drawing', 'beat', 'other', 'nogesture'
        ]
    else: # binary 모드
        target_classes = ['gesture', 'nogesture']

    print(f"\n🚀 Starting {mode.upper()} Evaluation... Output: {save_dir}")
    print("-" * 150)

    summary_results = []

    for video in video_list:
        gt_file = os.path.join(gt_root, f"{video}{mode}.csv")
        pred_file = os.path.join(pred_root, f"{video}{mode}.csv")
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"  ⚠️ Warning: Missing files for {video} ({mode}). Skipping...")
            continue
            
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
        
        df_merged = pd.merge(df_gt, df_pred, on='index', how='inner', suffixes=('_gt', '_pred'))
        
        y_true_vid = []
        y_pred_vid = []
        valid_count, invalid_count = 0, 0
        
        for _, row in df_merged.iterrows():
            raw_gt = get_clean_label(row['label_gt'])
            raw_pred = get_clean_label(row['label_pred'])
            
            # Binary 모드일 때 세부 제스처 라벨을 'gesture'로 통합
            gesture_subclasses = ['emblematic', 'indexing', 'representing', 'molding', 'acting', 'drawing', 'beat', 'other']
            if mode == "binary":
                if raw_gt in gesture_subclasses: raw_gt = 'gesture'
                if raw_pred in gesture_subclasses: raw_pred = 'gesture'

            clean_gt = raw_gt if raw_gt in target_classes else 'other'
            
            if raw_pred not in target_classes:
                invalid_count += 1
                clean_pred = 'other'
            else:
                valid_count += 1
                clean_pred = raw_pred
                
            y_true_vid.append(clean_gt)
            y_pred_vid.append(clean_pred)

        # [기능 추가] 각 비디오별 Distribution 집계 및 저장
        gt_series = pd.Series(y_true_vid).value_counts().reindex(target_classes, fill_value=0)
        pred_series = pd.Series(y_pred_vid).value_counts().reindex(target_classes, fill_value=0)
        dist_df = pd.DataFrame({'Label': target_classes, 'Ground_Truth': gt_series.values, 'Prediction': pred_series.values})
        dist_df.to_csv(os.path.join(save_dir, f"{video}_distribution.csv"), index=False)

        # [수정] 막대 위에 수치를 기입하는 그래프 생성
        plt.figure(figsize=(12, 6))
        dist_melted = dist_df.melt(id_vars='Label', var_name='Type', value_name='Count')
        ax = sns.barplot(data=dist_melted, x='Label', y='Count', hue='Type', palette='Blues_d')
        
        # 수치 기입 로직 추가
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9, fontweight='bold')

        plt.title(f'Distribution ({mode}) - {video}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{video}_distribution.png"))
        plt.close()

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
        
        # 지표 CSV 및 CM 저장
        pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, f"{video}_{mode}.csv"), index=False)
        cm = confusion_matrix(y_true_vid, y_pred_vid, labels=target_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes)
        plt.title(f'CM ({mode}) - {video}')
        plt.savefig(os.path.join(save_dir, f"{video}_cm.png"))
        plt.close()
        
        print(f"  ✅ {video} | Distribution & Metrics saved.")

    # 터미널 요약 출력
    if summary_results:
        print(f"\n📊 {mode.upper()} Summary Report")
        print(pd.DataFrame(summary_results).to_string(index=False, float_format="{:.4f}".format))
