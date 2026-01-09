import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# CSV 파일 불러오기
true_df = pd.read_csv("testdata/labels/edabbinary.csv")
pred_df = pd.read_csv("output/edab_framebyframe3.csv")

# index 기준으로 merge
merged_df = pd.merge(true_df, pred_df, on="index", suffixes=('_true', '_pred'))

# 레이블 추출
y_true = merged_df["label_true"]
y_pred = merged_df["label_pred"]

# 다중 클래스 분류 → average='macro' 사용
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)

print("📊 Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
