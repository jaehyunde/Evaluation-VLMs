import os
import csv
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


# -------------------------------------------------------
# 1. 모델 로드
# -------------------------------------------------------
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

print("Loading model...")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)

# Tokenizer 설정
processor.tokenizer.padding_side = "left"

# 🔥 pad_token_id 경고 제거 (중요)
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

model.generation_config.pad_token_id = processor.tokenizer.pad_token_id


# -------------------------------------------------------
# 2. 입력 비디오 폴더 설정
# -------------------------------------------------------
video_dir = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/video42/"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

print(f"Found {len(video_files)} video files.")


# -------------------------------------------------------
# 3. Gesture/NoGesture 프롬프트 생성 함수
# -------------------------------------------------------
def build_conversation(video_path):
    prompt_text = (
        "You are an expert gesture recognition model.\n"
        "Task: Determine whether the person in the video is performing a communicative hand gesture.\n\n"
        "Rules:\n"
        "- Output only one token: 'Gesture' or 'NoGesture'.\n"
        "- Do NOT describe the video.\n"
        "- Do NOT explain.\n"
        "- If unclear or ambiguous, output 'NoGesture'."
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return conversation


# -------------------------------------------------------
# 4. 단일 비디오 인퍼런스 함수
# -------------------------------------------------------
def predict_gesture(video_path):
    conversation = build_conversation(video_path)

    # Tokenize + 영상 프레임 인코딩
    inputs = processor.apply_chat_template(
        conversation,
        num_frames=25,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, torch.float16)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )

    # assistant 답변만 추출
    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_len:]

    # 디코딩
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    # 방어적 필터링
    if "Gesture" in text:
        return "Gesture"
    if "NoGesture" in text:
        return "NoGesture"

    return "NoGesture"


# -------------------------------------------------------
# 5. 모든 비디오 처리 후 CSV 저장
# -------------------------------------------------------
output_path = "output/gestures.csv"
os.makedirs("output", exist_ok=True)

rows = []

for idx, file in enumerate(video_files):
    full_path = os.path.join(video_dir, file)
    print(f"[{idx}] Processing:", full_path)

    label = predict_gesture(full_path)

    rows.append([idx, label])

# CSV 저장
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "label"])
    writer.writerows(rows)

print("\nDone! Saved to:", output_path)

