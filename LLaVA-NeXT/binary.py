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
video_name = "video41_new"
video_dir = f"/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/{video_name}/"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

# ---------------------------------------
# 3. CSV 출력 설정
# ---------------------------------------
os.makedirs("output", exist_ok=True)
csv_path = f"output/{video_name}binary.csv"

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
def build_conversation(video_path):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                    (
"**Video Analysis Task: High-Recall Hand Gesture Detection**\n\n"
    "You are a highly specialized computer vision expert and **intent-focused temporal analyst**. "
    "Your primary mission is to identify **ANY presence** of a **clear and intentional human Hand Gesture** throughout the video. Focus exclusively on the hands.\n\n"
    "**1. Target Definition (Valid Hand Gesture):**\n"
    "A **'Hand Gesture'** is defined by its **INTENT**— a purposeful movement of the hands or fingers for communication, manipulation, or expression. **Duration is secondary to Intent.**\n\n"
    "**2. Exclusion Definition (NoGesture/Filtering Rules):**\n"
    "A **'NoGesture'** scenario includes movements that are non-intentional and MUST be filtered out:\n"
    "a. Hands at rest, hanging passively, or static position changes (e.g., placing hand on lap).\n"
    "b. Incidental, non-communicative movement such as **nervous jitter, accidental slight twitching, or motion caused by external forces**.\n"
    "c. Holding an object without **active interaction** with it.\n\n"
    "**3. Judgement Criteria (Prioritizing Intent):**\n"
    "Analyze the video across the entire timeline. If a purposeful hand movement is observed, even if it is **brief** or **short-lived**, classify it as 'Gesture'. Only classify as 'NoGesture' if the hands are completely passive or the movement is clearly accidental/non-intentional. **Intent is the absolute final deciding factor.**\n\n"
    "**4. Instruction & Output Format (STRICT):**\n"
    "If a defined 'Hand Gesture' is observed, output **Gesture**. Otherwise, output **NoGesture**. "
    "**OUTPUT ONLY ONE WORD: Gesture or NoGesture. NO other text, punctuation, or explanation is permitted.**"
                    ),
                },
                {"type": "video", "path": video_path},
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
