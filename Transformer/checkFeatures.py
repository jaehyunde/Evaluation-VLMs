import os
import numpy as np

def check_feature_dimensions(features_dir="data/features"):
    """
    data/features 폴더 내 모든 .npy 파일의 feature 차원(F)을 확인하는 함수.
    각 파일의 shape과 feature dimension을 출력하고,
    전체적으로 F 값이 일관된지 검증한다.
    """
    feature_dims = set()
    file_count = 0

    print(f"🔍 Checking feature files in: {features_dir}\n")

    for fname in sorted(os.listdir(features_dir)):
        if fname.endswith(".npy"):
            path = os.path.join(features_dir, fname)
            try:
                arr = np.load(path)
                if arr.ndim != 2:
                    print(f"⚠️  {fname}: Unexpected shape {arr.shape}")
                    continue
                num_frames, num_features = arr.shape
                feature_dims.add(num_features)
                file_count += 1
                print(f"{fname:<50} → shape: {arr.shape}  (F={num_features})")
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")

    print("\n─────────────────────────────────────────────")
    print(f"📂 총 파일 수: {file_count}")
    if len(feature_dims) == 1:
        print(f"✅ 모든 파일의 feature dimension(F)이 동일합니다: F = {list(feature_dims)[0]}")
    else:
        print(f"⚠️ 파일마다 feature dimension이 다릅니다: {sorted(feature_dims)}")
    print("─────────────────────────────────────────────")

# 실행
if __name__ == "__main__":
    check_feature_dimensions("data/features")
