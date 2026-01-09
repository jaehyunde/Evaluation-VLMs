import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# -----------------------------------------------------------
# 1. 모델 및 프로세서 로드
# -----------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# -----------------------------------------------------------
# 2. 입력 / 출력 경로
# -----------------------------------------------------------
INPUT_DIR = "testdata/videos/edab"             # 모든 .mp4 파일이 들어있는 폴더
OUTPUT_CSV = "edab_output1.csv"      # 결과 저장 파일

# -----------------------------------------------------------
# 3. CSV 헤더 생성
# -----------------------------------------------------------
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "filename", "label"])

# -----------------------------------------------------------
# 4. 영상 파일 반복 처리 (최대 10개)
# -----------------------------------------------------------
#video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".mp4")]
#video_files = video_files[:10]  # ✅ 앞의 10개 파일만 선택

#for file_name in video_files:
#    video_path = os.path.join(INPUT_DIR, file_name)
#    print(f"🎥 Processing: {file_name}")

for file_name in sorted(os.listdir(INPUT_DIR)):
    if not file_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(INPUT_DIR, file_name)
    print(f"🎥 Processing: {file_name}")

    # 메시지 구성 (원본 구조 유지)
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
		    "text": ("Classify the hand gesture performed by the 3D character in the input video into one of the following categories: emblematic, indexing, representing, molding, acting, drawing, beat, or other. Output only one label."
                    ),
                },
            ],
        }
    ]

    # 인퍼런스 입력 준비
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
        clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"🧠 Model Output: {output_text}")

    # CSV에 저장
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([file_name, output_text])

print("\n✅ 10개 영상 처리 완료!")
print(f"결과 저장 경로: {OUTPUT_CSV}")
