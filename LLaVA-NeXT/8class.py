import os
import torch
import csv
import cv2   # check number of frames
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
video_name = "angle_3"
video_dir = f"/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/{video_name}/"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

# ---------------------------------------
# 3. CSV 출력 설정
# ---------------------------------------
os.makedirs("output", exist_ok=True)
csv_path = f"output/pure8class/{video_name}.csv"

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
                    ("You are an expert in gesture classification. The input is a video of a 3D human-like character created from motion capture data, performing a single hand gesture. Your task is to classify the gesture into one of these eight categories: emblematic, indexing, representing, molding, acting, drawing, beat, other. Focus only on hand movement and shape. Ignore facial expressions, eye gaze, or body posture. Carefully observe the gesture, determine its communicative function, and output only the final label in lowercase. Do not include explanations or any extra words — only one label from the list."
                    ),
                },
                {"type": "video", "path": path},
            ],
        }
    ]

# ---------------------------------------
# 6. 모든 비디오 처리
# ---------------------------------------
desired_frames = 32  # 원래 목표 프레임 수

for idx, filename in enumerate(video_files):
    video_path = os.path.join(video_dir, filename)
    print(f"\n=== Processing {idx}: {filename} ===")

    # --------------------------------------------------------
    # 🔥 IndexError 방지: 이 블록 전체를 try로 감싸기
    # --------------------------------------------------------
    try:
        # (A) 실제 프레임 수 계산
        total_frames = count_frames(video_path)
        print(f" total_frames = {total_frames}")

        # (B) 부족하면 자동 조정
        used_frames = min(desired_frames, total_frames)
        print(f" using num_frames = {used_frames}")

        # (C) 프롬프트 생성
        conversation = build_conversation(video_path)

        # (D) LLaVA 입력 생성
        inputs = processor.apply_chat_template(
            conversation,
            num_frames=used_frames,
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
  
        # ✅ 생성된 토큰만 추출 (중요!)
        prompt_len = inputs["input_ids"].shape[1]
        generated = output[:, prompt_len:]

        # ✅ 모델 응답(assistant)만 디코딩
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # ✅ 필터링 삭제: 모델 출력 그대로 저장
        label = decoded

        print(f" → Raw result: {repr(label)}")

    # --------------------------------------------------------
    # 🔥 IndexError 발생 시 → 건너뛰고 다음 영상으로 진행
    # --------------------------------------------------------
    except IndexError as e:
        print(f" !!! IndexError on {filename}: {repr(e)}")
        label = "IndexError"   # 기록은 남겨두자

    # --------------------------------------------------------
    # (선택적으로) 기타 오류는 그대로 터지도록 한다.
    # except Exception as e:
    #     raise e
    # --------------------------------------------------------

    # CSV 저장
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([idx, label])

print("\n=== Done ===")
print("Results saved to:", csv_path)
