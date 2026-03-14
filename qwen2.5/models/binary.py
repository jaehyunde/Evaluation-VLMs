import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

video="video41_new"

INPUT_DIR = f"testdata/videos/{video}"
OUTPUT_CSV = f"ergebnisse/{video}binary.csv"

# CSV 헤더 생성
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "label"])

# 영상 파일 리스트
video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".mp4")]

for idx, file_name in enumerate(video_files, start=1):
    video_path = os.path.join(INPUT_DIR, file_name)
    print(f"🎥 Processing ({idx}/{len(video_files)}): {file_name}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 32.0,
                },
                {
                    "type": "text",
		    "text": ("**Video Analysis Task: High-Recall Hand Gesture Detection**\n\n"
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
		)
                },
            ],
        }
    ]

    # 입력 준비
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 모델 실행
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(f"🧠 Output: {output_text}")

    # CSV 저장 (index 포함)
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([idx, output_text])

print("\n✅ 모든 영상 처리 완료!")
print(f"결과 저장 경로: {OUTPUT_CSV}")
