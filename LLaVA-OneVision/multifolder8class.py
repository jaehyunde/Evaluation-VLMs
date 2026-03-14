import os
import csv
import cv2
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# -------------------------------------------------------
# 1. 모델 로드 (반복문 밖에서 한 번만 수행)
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
# 2. 인퍼런스 보조 함수
# -------------------------------------------------------
def build_conversation(video_path):
    prompt_text = (
        "You are an expert in gesture classification. The input is a video of a 3D human-like character created from motion capture data, performing a single hand gesture. Your task is to classify the gesture into one of these eight categories: emblematic, indexing, representing, molding, acting, drawing, beat or other. Focus only on hand movement and shape. Ignore facial expressions, eye gaze, or body posture. Carefully observe the gesture, determine its communicative function, and output only the final label in lowercase. Do not include explanations or any extra words — only one label from the list."
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

    # B) 프롬프트 준비
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

        # C) 모델 generate
        print("   -> generating...")
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
        print("   -> generation done")

        # D) 결과 디코딩
        prompt_len = inputs["input_ids"].shape[1]
        generated = out[:, prompt_len:]
        text = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        print(f"   raw output: {repr(text)}")
        
        # 소문자 변환 및 마침표 제거하여 반환
        return text.lower().replace(".", "").strip()

    except Exception as e:
        print(f"   !! Error: {e}")
        return None

# -------------------------------------------------------
# 3. 다중 폴더 순회 및 작업 수행
# -------------------------------------------------------
# 분석할 폴더 리스트
videolist = ['angle_1','angle_2','angle_3','edab', 'video41', 'video42', 'video43']
# 비디오들이 들어있는 기본 경로
base_path = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/"
# 결과를 저장할 폴더 (요청하신 경로)
output_root = "output/pure8class"
os.makedirs(output_root, exist_ok=True)

for videoname in videolist:
    video_dir = os.path.join(base_path, videoname)
    
    # 해당 비디오 폴더가 실제로 있는지 확인
    if not os.path.exists(video_dir):
        print(f"\n⚠️ Directory not found: {video_dir}, skipping...")
        continue

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    
    print(f"\n" + "="*60)
    print(f"📂 Folder: {videoname} | Files: {len(video_files)}")
    print("="*60)

    # 개별 폴더 결과를 저장할 파일 경로
    output_path = f"{output_root}/{videoname}.csv"

    rows = []
    for idx, file in enumerate(video_files):
        full_path = os.path.join(video_dir, file)
        print(f"\n[{idx}] Processing: {full_path}")

        label = predict_gesture(full_path)

        if label is not None:
            print(f"   → result: {label}")
            rows.append([idx, label])
        else:
            print(f"   → Failed (index {idx})")
            rows.append([idx, "error_failed"])

    # CSV 파일로 저장
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label"])
        writer.writerows(rows)

    print(f"\n✅ Finished {videoname}! Result: {output_path}")

print("\n🚀 All tasks completed successfully!")
