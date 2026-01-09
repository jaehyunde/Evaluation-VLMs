import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# 파일 경로 설정 (3개로 확장)
true_files = [
    "testdata/labels/video41_newbinary.csv",
    "testdata/labels/video42_newbinary.csv",
    "testdata/labels/video43_newbinary.csv"
]

output = "new_result"
root = "LLaVA-OneVision/output"
pred_files = [
    f"/home/stud_homes/s6010479/Jayproject/{root}/video41_{output}.csv",
    f"/home/stud_homes/s6010479/Jayproject/{root}/video42_{output}.csv",
    f"/home/stud_homes/s6010479/Jayproject/{root}/video43_{output}.csv"
]

# 쌍 개수 체크
assert len(true_files) == len(pred_files), "true / pred 파일 개수가 다릅니다."

merged_list = []

# 1) 각 쌍을 index 기준으로 merge
for t_path, p_path in zip(true_files, pred_files):
    true_df = pd.read_csv(t_path)
    pred_df = pd.read_csv(p_path)

    merged = pd.merge(true_df, pred_df, on="index", suffixes=('_true', '_pred'))
    merged_list.append(merged)

# 2) 하나의 데이터프레임으로 통합
merged_all = pd.concat(merged_list, ignore_index=True)

# 3) 레이블 추출
y_true = merged_all["label_true"]
y_pred = merged_all["label_pred"]

# 4) metric 계산 (요청 순서: f1 → precision → recall → accuracy)
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)


# 5) 출력
print(f"f1: {f1:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"accuracy: {accuracy:.4f}")
