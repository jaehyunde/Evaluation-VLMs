import os
import numpy as np
from collections import defaultdict

features_dir = "data/features"

# 집계할 타입들(표시 순서)
KNOWN_TYPES = ["M3D", "9", "V", "P", "S", "1", "2", "3"]

frames_by_type = defaultdict(int)
files_by_type  = defaultdict(int)

def detect_type(basename: str) -> str:
    """
    타입 판정 규칙:
      - 'M3E' 또는 'M3D'로 시작하면 'M3E' 버킷으로 집계
      - 그 외에는 첫 글자가 9/V/P/S/1/2/3 중 하나면 그 글자
      - 아니면 'OTHER'
    """
    if basename.startswith("M3E") or basename.startswith("M3D"):
        return "M3D"
    first = basename[0]
    if first in {"9", "V", "P", "S", "1", "2", "3"}:
        return first
    return "OTHER"

overall_total_frames = 0
unknown_examples = []

for f in sorted(os.listdir(features_dir)):
    if not f.endswith(".npy"):
        continue

    path = os.path.join(features_dir, f)
    arr = np.load(path, allow_pickle=False)
    n_frames = int(arr.shape[0])

    type_key = detect_type(f)
    frames_by_type[type_key] += n_frames
    files_by_type[type_key]  += 1
    overall_total_frames     += n_frames

    if type_key == "OTHER" and len(unknown_examples) < 5:
        unknown_examples.append(f)

# ---- 타입별 결과 ----
print("=== Frame totals by type ===")
for t in KNOWN_TYPES:
    print(f"{t}: frames={frames_by_type[t]:,}  files={files_by_type[t]:,}")

if frames_by_type["OTHER"] > 0:
    print(f"OTHER: frames={frames_by_type['OTHER']:,}  files={files_by_type['OTHER']:,}")
    if unknown_examples:
        print("  examples:", ", ".join(unknown_examples))

# ---- 데이터셋 이름으로 묶은 합산 ----
GESRes     = frames_by_type["1"] + frames_by_type["2"] + frames_by_type["3"]
MULTISIMO  = frames_by_type["P"] + frames_by_type["S"]
ZHUBO      = frames_by_type["9"]
SaGA       = frames_by_type["V"]
M3D_TED    = frames_by_type["M3D"]

print("\n=== Frame totals by dataset groups ===")
print(f"GESRes (1+2+3): {GESRes:,}")
print(f"MULTISIMO (P+S): {MULTISIMO:,}")
print(f"ZHUBO (9): {ZHUBO:,}")
print(f"SaGA (V): {SaGA:,}")
print(f"M3D_TED (M3E): {M3D_TED:,}")

# ---- 마지막: 전체 프레임 합산 ----
print("\n=== Overall total frames (ALL) ===")
print(f"TOTAL: {overall_total_frames:,}")
