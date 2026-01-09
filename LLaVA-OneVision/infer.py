from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

video_path = "/home/stud_homes/s6010479/Jayproject/qwen2.5/models/testdata/videos/video42/000.mp4"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "The Movement in this video is a kind of Gesture or NoGesture?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    num_frames=8,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.float16)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=60)

# 🔹 프롬프트 길이만큼 잘라서 assistant 답변만 남기기
prompt_len = inputs["input_ids"].shape[1]
generated_only = out[:, prompt_len:]

answers = processor.batch_decode(
    generated_only,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("ASSISTANT:", answers[0])
