import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. 기본 설정
angles = ['1', '2', '3']
modes = ['8class', 'binary']
models_info = {
    'qwen2.5': 'qwen2.5/models/ergebnisse',
    'LLaVA-NeXT': 'LLaVA-NeXT/output',
    'LLaVA-Onevision': 'LLaVA-OneVision/output'
}
gt_base_path = 'qwen2.5/models/testdata/labels'
save_base_path = 'result/angles'

# 클래스 정의
target_classes_8 = [
    'emblematic', 'indexing', 'representing', 'molding',
    'acting', 'drawing', 'beat', 'other', 'nogesture'
]
target_classes_binary = ['gesture', 'nogesture'] # 이진 분류용 예시

# 디렉토리 생성
os.makedirs(save_base_path, exist_ok=True)

def get_clean_label(label):
    if pd.isna(label): return None
    return str(label).lower().strip().replace(".", "")

# 2. 분석 루프
for angle in angles:
    for mode in modes:
        # 모드별 설정 (8class vs binary)
        target_classes = target_classes_8 if mode == '8class' else target_classes_binary
        
        for model_name, pred_dir in models_info.items():
            gt_file = os.path.join(gt_base_path, f"angle_{angle}{mode}.csv")
            pred_file = os.path.join(pred_dir, f"angle_{angle}{mode}.csv")
            
            if not os.path.exists(gt_file) or not os.path.exists(pred_file):
                print(f"⚠️ File not found: {gt_file} or {pred_file}")
                continue
            
            # 데이터 로드
            df_gt = pd.read_csv(gt_file)
            df_pred = pd.read_csv(pred_file)
            
            # 'index' 기준으로 병합 (기존 코드 로직 유지)
            df_merged = pd.merge(df_gt, df_pred, on='index', how='left', suffixes=('_gt', '_pred'))
            
            all_y_true = [get_clean_label(l) for l in df_merged['label_gt']]
            all_y_pred = [get_clean_label(l) for l in df_merged['label_pred']]

            # 유효 데이터 필터링 (target_classes 내에 있는 것만)
            filtered_true = []
            filtered_pred = []
            for t, p in zip(all_y_true, all_y_pred):
                if t in target_classes and p in target_classes:
                    filtered_true.append(t)
                    filtered_pred.append(p)

            if not filtered_true:
                print(f"❌ No valid data for {model_name} - Angle {angle} - {mode}")
                continue

            # 3. 시각화 섹션
            cm = confusion_matrix(filtered_true, filtered_pred, labels=target_classes)
            plt.figure(figsize=(12, 10))

            if mode == '8class':
                # --- 8class용 상세 설정 (Regular 스타일) ---
                ax = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                                 xticklabels=target_classes, yticklabels=target_classes,
                                 annot_kws={"size": 28, "weight": "normal"})
                
                plt.title(f'Confusion Matrix - {model_name} (Angle {angle})', fontsize=20, pad=20, fontweight='normal')
                plt.xlabel('Predicted Label', fontsize=18, labelpad=15, fontweight='normal')
                plt.ylabel('True Label', fontsize=18, labelpad=15, fontweight='normal')
                plt.xticks(fontsize=18, rotation=45, fontweight='normal')
                plt.yticks(fontsize=18, rotation=0, fontweight='normal')

            else:
                # --- binary용 상세 설정 (Bold 스타일) ---
                ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                 xticklabels=target_classes, yticklabels=target_classes,
                                 annot_kws={"size": 28, "weight": "normal"})
                
                plt.title(f'Confusion Matrix - {model_name} (Angle {angle})', fontsize=20, pad=25, fontweight='normal')
                plt.xlabel('Predicted Label', fontsize=16, labelpad=15)
                plt.ylabel('True Label', fontsize=16, labelpad=15)
                plt.xticks(fontsize=13, rotation=45)
                plt.yticks(fontsize=13, rotation=0)

            plt.tight_layout()
            
            # 저장 경로 설정: result/angles/{model}_{angle}_{mode}_cm.png
            save_filename = f"angle_{angle}_{mode}{model_name}cm.png"
            plt.savefig(os.path.join(save_base_path, save_filename), dpi=300)
            plt.close()
            print(f"✅ Saved: {save_filename}")
