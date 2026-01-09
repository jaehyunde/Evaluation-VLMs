import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# -------------------------------
# 🔥 GPU fallback: cuda:0 → cuda:1 자동 전환
# -------------------------------
if torch.cuda.is_available():
    try:
        device = torch.device("cuda:0")
        print("Trying cuda:0 ...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map={"": "cuda:0"}
        )
        print("✔ Using auto")
    except torch.cuda.OutOfMemoryError:
        print("❌ cuda:0 OOM → switching to cuda:1")
        torch.cuda.empty_cache()
        device = torch.device("cuda:1")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map={"": "cuda:2"}
        )
        print("✔ Using cuda:1")
else:
    device = torch.device("cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    )
    print("✔ Using CPU")

# processor는 GPU 영향 없음 → 그대로
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# -------------------------------
# 기존 코드 (변경 없음)
# -------------------------------
INPUT_DIR = "testdata/videos/edab"
OUTPUT_CSV = "output/edab_framebyframe1.csv"

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
                    "text": ("Frame-level gesture detection task. You are given a SINGLE video frame. Analyze ONLY the hands. Definitions: Gesture = intentional communicative hand movement. NoGesture = resting hands, non-communicative random motion, or transitional movement between gestures. Task: Classify THIS SINGLE FRAME as either 'Gesture' or 'NoGesture'. Output rule: Do NOT describe the frame. Do NOT explain. Output EXACTLY ONE of the following tokens and nothing else: Gesture or NoGesture."
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
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        except torch.cuda.OutOfMemoryError:
            print("❌ OOM during generation → retrying on cuda:1")
            torch.cuda.empty_cache()
            device = torch.device("cuda:1")
            model.to(device)
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)

    # 결과 후처리
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
