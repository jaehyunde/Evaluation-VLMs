import os
import torch
import csv
import cv2   # 프레임 세는 용도로만 사용 (가볍고 빠름)
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# ---------------------------------------
# 1. 환경 설정
# ---------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# ---------------------------------------
# 2. 비디오 폴더
# ---------------------------------------
video_dir = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/video42/"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

# ---------------------------------------
# 3. CSV 출력 설정
# ---------------------------------------
os.makedirs("output", exist_ok=True)
csv_path = "output/video42_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "label"])

# ---------------------------------------
# 4. 간단한 프레임 수 계산 함수
# ---------------------------------------
def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames

# ---------------------------------------
# 5. 프롬프트 구성
# ---------------------------------------
def build_conversation(path):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                    (
                        "Analyze ONLY the hand movements.\n"
                        "Output exactly ONE token: Gesture or NoGesture."
                    ),
                },
                {"type": "video", "path": path},
            ],
        }
    ]

# ---------------------------------------
# 6. 모든 비디오 처리
# ---------------------------------------
desired_frames = 25  # 원래 목표 프레임 수

for idx, filename in enumerate(video_files):
    video_path = os.path.join(video_dir, filename)
    print(f"\n=== Processing {idx}: {filename} ===")

    # (A) 실제 프레임 수 계산
    total_frames = count_frames(video_path)
    print(f" total_frames = {total_frames}")

    # (B) 영상 길이에 따라 num_frames 자동 조정
    used_frames = min(desired_frames, total_frames)
    print(f" using num_frames = {used_frames}")

    # (C) 프롬프트 생성
    conversation = build_conversation(video_path)

    # (D) LLaVA 입력 생성
    inputs = processor.apply_chat_template(
        conversation,
        num_frames=used_frames,   # ← 여기만 조정됨!
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 입력을 GPU로 이동
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)

    # 모델 실행
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

    # "Gesture" 단어가 들어있으면 Gesture로 판단
    label = "Gesture" if "Gesture" in decoded else "NoGesture"
    print(" → Result:", label)

    # CSV 저장
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([idx, label])

print("\n=== Done ===")
print("Results saved to:", csv_path)
