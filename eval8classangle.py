import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

# 1. 환경 설정 및 경로 정의
models_info = {
    'Qwen2.5': os.path.expanduser('~/Jayproject/qwen2.5/models/ergebnisse/pure8class'),
    'OneVision': os.path.expanduser('~/Jayproject/LLaVA-OneVision/output/pure8class'),
    'NeXT': os.path.expanduser('~/Jayproject/LLaVA-NeXT/output/pure8class')
}

angles = ['angle_1', 'angle_2', 'angle_3']
gt_file_path = os.path.expanduser('~/Jayproject/qwen2.5/models/testdata/labels/edabpure.csv')  # 공통 GT 파일
save_root = os.path.expanduser('~/Jayproject/result/pureangles')

# nogesture가 제거된 8개 타겟 클래스
target_classes = [
    'emblematic', 'indexing', 'representing', 'molding', 
    'acting', 'drawing', 'beat', 'other'
]

def get_clean_label(label):
    if pd.isna(label): return 'other'
    return str(label).lower().strip().replace(".", "")

# 2. 분석 루프 시작
for model_name, pred_dir in models_info.items():
    model_save_dir = os.path.join(save_root, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n🚀 모델 분석 중: {model_name} (NoGesture 제거 버전)")
    print("=" * 110)

    for angle in angles:
        pred_file_path = os.path.join(pred_dir, f"{angle}.csv")
        
        if not os.path.exists(gt_file_path) or not os.path.exists(pred_file_path):
            print(f"⚠️ 파일 누락: {angle} (건너뜀)")
            continue

        # 데이터 로드
        df_gt = pd.read_csv(gt_file_path)
        df_pred = pd.read_csv(pred_file_path)
        df_merged = pd.merge(df_gt, df_pred, on='index', how='inner', suffixes=('_gt', '_pred'))

        y_true, y_pred = [], []
        invalid_list = []
        valid_count = 0

        for _, row in df_merged.iterrows():
            raw_gt = get_clean_label(row['label_gt'])
            raw_pred = get_clean_label(row['label_pred'])

            # [수정] nogesture 관련 데이터 완전 제외 (GT가 nogesture인 경우 skip)
            if raw_gt == 'nogesture':
                continue
            
            # GT가 target_classes에 없는 경우 'other'로 간주 (단, nogesture는 위에서 제외됨)
            clean_gt = raw_gt if raw_gt in target_classes else 'other'
            
            # [수정] Prediction 처리: target_classes 외의 라벨은 invalid로 기록 후 'other' 치환
            if raw_pred not in target_classes:
                invalid_list.append(raw_pred)
                clean_pred = 'other'
            else:
                valid_count += 1
                clean_pred = raw_pred
            
            y_true.append(clean_gt)
            y_pred.append(clean_pred)

        # Invalid 샘플 분석 (종류와 갯수)
        invalid_counts = dict(Counter(invalid_list))
        invalid_detail_str = ", ".join([f"'{k}': {v}" for k, v in invalid_counts.items()])

        # 3. 지표 계산 (5자리에서 반올림 -> 4자리 표기)
        metrics = {
            "Angle": angle,
            "Valid_Samples": valid_count,
            "Invalid_Samples": len(invalid_list),
            "Invalid_Details": invalid_detail_str if invalid_detail_str else "None",
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Prec(m)": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
            "Rec(m)": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
            "F1(m)": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            "Prec(w)": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            "Rec(w)": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            "F1(w)": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
        }

        # 터미널 출력
        print(f"📊 [{angle}] 지표:")
        for k, v in metrics.items(): print(f"  {k}: {v}")
        print("-" * 60)

        # CSV 저장
        pd.DataFrame([metrics]).to_csv(os.path.join(model_save_dir, f"{angle}_results.csv"), index=False)

        # 4. 시각화 (CM, Distribution)
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=target_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes,annot_kws={"size":18,"weight":"normal"})
        plt.title(f'CM ({model_name} - {angle})', fontsize=20, pad=25, fontweight='normal')

        # 3. 축 이름(Label) 폰트 조절
        plt.xlabel('Predicted Label', fontsize=18, labelpad=15, fontweight='normal')
        plt.ylabel('True Label', fontsize=18, labelpad=15, fontweight='normal')

        # 4. 축 눈금(Ticks - 클래스 이름) 폰트 조절
        plt.xticks(fontsize=16, rotation=45, fontweight='normal')
        plt.yticks(fontsize=16, rotation=0, fontweight='normal')

        # 레이아웃 최적화 및 저장
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_dir, f"{angle}_cm.png"), dpi=300)
        plt.close()

        # Distribution
        gt_dist = pd.Series(y_true).value_counts().reindex(target_classes, fill_value=0)
        pred_dist = pd.Series(y_pred).value_counts().reindex(target_classes, fill_value=0)
        dist_df = pd.DataFrame({'Label': target_classes, 'GT': gt_dist.values, 'Pred': pred_dist.values})
        
        plt.figure(figsize=(12, 6))
        dist_melted = dist_df.melt(id_vars='Label', var_name='Type', value_name='Anzahl')
        ax = sns.barplot(data=dist_melted, x='Label', y='Anzahl', hue='Type', palette='Blues_d')
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=16, fontweight='normal')
        plt.title(f'Distribution ({model_name} - {angle})')
        plt.xticks(rotation=45,fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_dir, f"{angle}_distribution.png"))
        plt.close()

print(f"\n✅ 분석 완료! 결과 저장: {save_root}")
