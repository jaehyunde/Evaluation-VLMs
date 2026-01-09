import os
import csv
import cv2
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


# -------------------------------------------------------
# 1. 모델 로드
# -------------------------------------------------------
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

print("Loading model...")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.float16,      # torch_dtype → dtype (deprecation fix)
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"

# pad_token 설정
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

videoname = "angle_1"
# -------------------------------------------------------
# 2. 입력 비디오 폴더
# -------------------------------------------------------
video_dir = (
    f"/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/{videoname}/"
)

video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
print(f"Found {len(video_files)} video files.")


# -------------------------------------------------------
# 3. Gesture/NoGesture 프롬프트
# -------------------------------------------------------
def build_conversation(video_path):
    prompt_text = (
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
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


# -------------------------------------------------------
# 4. 단일 비디오 인퍼런스 (FULL SAFE VERSION)
# -------------------------------------------------------
def predict_gesture(video_path):

    # --- A) total frames 체크 (IndexError-safe) ---
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        print(f"   !! Frame read error: {e}, skipping")
        return None

    if total_frames <= 0:
        print("   !! total_frames=0 or invalid → skip")
        return None

    desired_num_frames = 32
    num_frames = min(desired_num_frames, total_frames)
    print(f"   total_frames={total_frames}, using num_frames={num_frames}")

    # --- B) 프롬프트 준비 (IndexError-safe) ---
    conversation = build_conversation(video_path)

    try:
        inputs = processor.apply_chat_template(
            conversation,
            num_frames=num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, torch.float16)
    except IndexError as e:
        print(f"   !! IndexError during sampling: {e} → skip")
        return None
    except Exception as e:
        print(f"   !! apply_chat_template error: {e} → skip")
        return None

    # --- C) 모델 generate (IndexError-safe) ---
    try:
        print("   -> generating...")
        out = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
        )
        print("   -> generation done")
    except IndexError as e:
        print(f"   !! IndexError during generation: {e} → skip")
        return None
    except Exception as e:
        print(f"   !! generate error: {e} → skip")
        return None

    # --- D) slicing assistant output (IndexError-safe) ---
    try:
        prompt_len = inputs["input_ids"].shape[1]
        generated = out[:, prompt_len:]
    except IndexError as e:
        print(f"   !! IndexError slicing output: {e} → skip")
        return None
    except Exception as e:
        print(f"   !! Unexpected slicing error: {e} → skip")
        return None

    # --- E) decode output (IndexError-safe) ---
    try:
        text = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()
    except IndexError as e:
        print(f"   !! IndexError decoding output: {e} → skip")
        return None
    except Exception as e:
        print(f"   !! Unexpected decode error: {e} → skip")
        return None

    print(f"   raw output: {repr(text)}")

    # --- F) Gesture/NoGesture 필터링 ---
    t = text.lower().replace(" ", "")

    if t.startswith("gesture"):
        return "Gesture"
    if t.startswith("nogesture"):
        return "NoGesture"

    if "gesture" in t:
        return "Gesture"
    if "nogesture" in t:
        return "NoGesture"

    return "NoGesture"


# -------------------------------------------------------
# 5. 전체 비디오 처리 + CSV 저장
# -------------------------------------------------------
output_path = f"output/{videoname}binary.csv"
os.makedirs("output", exist_ok=True)

rows = []

for idx, file in enumerate(video_files):
    full_path = os.path.join(video_dir, file)
    print(f"[{idx}] Processing:", full_path)

    label = predict_gesture(full_path)

    if label is None:
        print("   → Skipped (error)")
        continue

    # 콘솔에도 결과 바로 출력
    print(f"   → result: {label}")

    rows.append([idx, label])

# CSV 저장
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "label"])
    writer.writerows(rows)

print("\nDone! Saved to:", output_path)
