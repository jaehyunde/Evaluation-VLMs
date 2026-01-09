from huggingface_hub import hf_hub_download
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# -----------------------------
# 1. 모델 및 프로세서 로드
# -----------------------------
# GPU 여부 확인
generation_device = "cuda" if torch.cuda.is_available() else "cpu"

# LLaVA-NeXT-Video 모델 로드 (반정밀도 + device_map 자동)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16 if generation_device == "cuda" else torch.float32,
    device_map="auto" if generation_device == "cuda" else None,
)

processor = LlavaNextVideoProcessor.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf"
)

# -----------------------------
# 2. 비디오 파일 준비
# -----------------------------
# HuggingFace Hub에서 데모 비디오 받기
video_path = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/video42/050.mp4"

# -----------------------------
# 3. 대화(프롬프트) 구성
# -----------------------------
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Video Analysis Task: High-Recall Hand Gesture Detection**\n\n"
    "You are a highly specialized computer vision expert and **intent-focused temporal analyst**. "
    "Your primary mission is to identify **ANY presence** of a **clear and intentional human Hand Gesture** throughout the video. Focus exclusive"
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
    "**OUTPUT ONLY ONE WORD: Gesture or NoGesture. NO other text, punctuation, or explanation is permitted.**"},
            {"type": "video", "path": video_path},
        ],
    },
]

# -----------------------------
# 4. 입력 텐서 생성
# -----------------------------
inputs = processor.apply_chat_template(
    conversation,
    num_frames=25,                 # 샘플링할 프레임 수
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# ⚠ 중요: 입력 텐서를 모델이 올라가 있는 디바이스로 옮기기
inputs = {
    k: (v.to(generation_device) if torch.is_tensor(v) else v)
    for k, v in inputs.items()
}

# -----------------------------
# 5. 생성 실행
# -----------------------------
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=60,
    )

# -----------------------------
# 6. 디코딩 및 출력
# -----------------------------
answers = processor.batch_decode(
    out,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("=== MODEL OUTPUT ===")
for i, ans in enumerate(answers):
    print(f"[{i}] {ans}")
