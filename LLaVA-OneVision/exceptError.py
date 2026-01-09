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
    "You are an expert in gesture classification. The input is a video of a 3D human-like character created from motion capture data, performing a single hand gesture. Your task is to classify the gesture into one of these eight categories: emblematic, indexing, representing, molding, acting, drawing, beat, other, NoGesture. Focus only on hand movement and shape. Ignore facial expressions, eye gaze, or body posture. Carefully observe the gesture, determine its communicative function, and output only the final label in lowercase. Do not include explanations or any extra words — only one label from the list."
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
    return "NoGesture" if text == "no gesture" else text


# -------------------------------------------------------
# 5. 전체 비디오 처리 + CSV 저장
# -------------------------------------------------------
output_path = f"output/{videoname}8class.csv"
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
