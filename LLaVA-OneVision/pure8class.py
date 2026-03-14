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
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"

# pad_token 설정
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

# -------------------------------------------------------
# 2. 입력 설정 (폴더명 수정 가능)
# -------------------------------------------------------
videoname = "video43" # 분석할 폴더명
video_dir = f"/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/{videoname}/"

video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
print(f"Found {len(video_files)} video files in {video_dir}")

# -------------------------------------------------------
# 3. 프롬프트 구성
# -------------------------------------------------------
def build_conversation(video_path):
    prompt_text = ("You are an expert in gesture classification. The input is a video of a 3D human-like character created from motion capture data, performing a single hand gesture. Your task is to classify the gesture into one of these eight categories: emblematic, indexing, representing, molding, acting, drawing, beat or other. Focus only on hand movement and shape. Ignore facial expressions, eye gaze, or body posture. Carefully observe the gesture, determine its communicative function, and output only the final label in lowercase. Do not include explanations or any extra words — only one label from the list."
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
# 4. 단일 비디오 인퍼런스
# -------------------------------------------------------
def predict_gesture(video_path):
    # A) 프레임 수 체크
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        print(f"   !! Frame read error: {e}")
        return None

    if total_frames <= 0:
        return None

    desired_num_frames = 32
    num_frames = min(desired_num_frames, total_frames)
    print(f"   total_frames={total_frames}, using num_frames={num_frames}")

    # B) 입력 준비
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

        # C) 모델 생성
        print("   -> generating...")
        out = model.generate(
            **inputs,
            max_new_tokens=10, # 라벨이 조금 길 수 있으므로 소폭 상향
            do_sample=False,
        )
        
        # D) 결과 슬라이싱 및 디코딩
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.batch_decode(
            out[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()
        
        print(f"   raw output: {repr(text)}")
        
        # E) 후처리: 소문자화 및 문장 부호 제거 (raw output 보존)
        return text.lower().replace(".", "").strip()

    except Exception as e:
        print(f"   !! Error during inference: {e}")
        return None

# -------------------------------------------------------
# 5. 전체 비디오 처리 + CSV 저장
# -------------------------------------------------------
output_path = f"output/pure8class/{videoname}.csv"
os.makedirs("output", exist_ok=True)

rows = []

for idx, file in enumerate(video_files):
    full_path = os.path.join(video_dir, file)
    print(f"\n[{idx}] Processing: {full_path}")

    label = predict_gesture(full_path)

    # 수정된 부분: label이 생성되었다면 필터링 없이 그대로 저장
    if label is not None:
        print(f"   → result: {label}")
        rows.append([idx, label])
    else:
        # 실패한 경우에도 인덱스를 유지하기 위해 에러 기록
        print(f"   → Failed (index {idx})")
        rows.append([idx, "error_failed"])

# CSV 저장
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "label"])
    writer.writerows(rows)

print(f"\n✅ Done! Results saved to: {output_path}")
