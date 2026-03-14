import os
import torch
import csv
import cv2
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# ---------------------------------------
# 1. 환경 설정 및 모델 로드
# ---------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LLaVA-NeXT-Video model...")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# ---------------------------------------
# 2. 다중 폴더 리스트 설정
# ---------------------------------------
videolist = ['edab', 'video41', 'video42', 'video43']
base_path = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/"
output_root = "output/newprompt8class"
os.makedirs(output_root, exist_ok=True)

# ---------------------------------------
# 3. 폴더 순회 작업 시작
# ---------------------------------------
for video_name in videolist:
    video_dir = os.path.join(base_path, video_name)
    
    if not os.path.exists(video_dir):
        print(f"\n⚠️ Directory not found: {video_dir}, skipping...")
        continue

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    
    print(f"\n" + "="*60)
    print(f"📂 Processing Folder: {video_name} ({len(video_files)} files)")
    print("="*60)

    # 결과 CSV 경로 설정
    csv_path = f"{output_root}/{video_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label"])

    # 개별 비디오 처리 루프
    for idx, filename in enumerate(video_files):
        video_path = os.path.join(video_dir, filename)
        print(f"\n[{idx}] Processing: {video_path}")

        try:
            # 프레임 수 확인
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames <= 0:
                print(f" !!! Skip {filename}: total_frames <= 0")
                label = "error_frame"
            else:
                # LLaVA-NeXT-Video의 권장 프레임 수 (일반적으로 16~32 프레임)
                used_frames = min(total_frames, 32)
                
                # 최적화된 단답형 프롬프트 구성
                prompt_text = (
                    "Task: Classify the hand gesture in the video of a 3D character.\n\n"
                    "Instructions:\n"
                    "1. Focus exclusively on hand trajectory, finger orientation, and hand shape.\n"
                    "2. Ignore facial expressions, gaze, and body posture.\n"
                    "3. Determine the gesture's communicative function among these categories: \n"
                    "list=[emblematic, indexing, representing, molding, acting, drawing, beat, other].\n\n"
                    "Constraint:\n"
                    "- Output ONLY the final label in lowercase.\n"
                    "- Do NOT include reasoning, explanations, or any other words.\n"
                    "- Your entire response must be exactly one label from the list above.\n\n"
                    "Format:\n"
                    "<label>"
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

                # 입력 준비
                inputs = processor.apply_chat_template(
                    conversation,
                    num_frames=used_frames,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                for k in inputs:
                    if torch.is_tensor(inputs[k]):
                        inputs[k] = inputs[k].to(device)

                # 모델 실행
                with torch.no_grad():
                    # 단답형이므로 max_new_tokens는 작게 설정 (효율성)
                    output = model.generate(**inputs, max_new_tokens=15)
          
                # 생성된 텍스트만 추출
                prompt_len = inputs["input_ids"].shape[1]
                generated = output[:, prompt_len:]
                decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

                # 후처리: 소문자화 및 문장 부호 제거
                label = decoded.lower().replace(".", "").replace(",", "").strip()
                print(f"   → Result Label: {repr(label)}")

        except IndexError as e:
            print(f" !!! IndexError on {filename}: {repr(e)}")
            label = "error_index"
        except Exception as e:
            print(f" !!! Unexpected error on {filename}: {repr(e)}")
            label = "error_failed"

        # 실시간 CSV 저장
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, label])

    print(f"\n✅ Finished {video_name}! Saved to: {csv_path}")

print("\n🚀 All folders processed successfully!")
