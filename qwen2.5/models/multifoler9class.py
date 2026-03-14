import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# GPU 메모리 정리
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 프로세서 로드
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# [수정] 처리할 비디오 리스트 정의
videos_to_process = ["edab_new", "video41_new", "video42_new", "video43_new"]

for video_name in videos_to_process:
    INPUT_DIR = f"testdata/videos/{video_name}"
    OUTPUT_CSV = f"ergebnisse/{video_name}8class.csv"
    
    # 폴더 존재 여부 확인
    if not os.path.exists(INPUT_DIR):
        print(f"⚠️ Warning: Directory not found: {INPUT_DIR}. Skipping...")
        continue

    # CSV 헤더 생성 (매 비디오마다 새로 생성)
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label"])

    # 영상 파일 리스트
    video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".mp4")]

    print(f"\n🚀 Starting processing for video folder: {video_name} ({len(video_files)} files)")

    for idx, file_name in enumerate(video_files, start=1):
        video_path = os.path.join(INPUT_DIR, file_name)
        print(f"🎥 [{video_name}] Processing ({idx}/{len(video_files)}): {file_name}")

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
                        "text": (
                            "You are an expert in gesture classification. The input is a video of a 3D human-like character "
                            "created from motion capture data, performing a single hand gesture. Your task is to classify "
                            "the gesture into one of these eight categories: emblematic, indexing, representing, molding, "
                            "acting, drawing, beat, other, NoGesture. Focus only on hand movement and shape. Ignore facial expressions, "
                            "eye gaze, or body posture. Carefully observe the gesture, determine its communicative function, "
                            "and output only the final label in lowercase. Do not include explanations or any extra words — "
                            "only one label from the list."
                        )
                    },
                ],
            }
        ]

        # 입력 준비 및 모델 실행
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

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

    print(f"✅ Finished folder: {video_name} -> Saved to {OUTPUT_CSV}")

print("\n🎉 모든 비디오 리스트 처리 완료!")
