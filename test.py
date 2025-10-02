from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit 
import base64
import io
from PIL import Image

name = '/data/hezhuangzhuang-p/vr-zero/storage/temp_results/temp_2_1759069365899_42447.json'
with open(name, 'r') as f:
    data = json.load(f)

model_path = '/data/hezhuangzhuang-p/llm/Qwen2.5-VL-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = vllm.LLM(
    model=model_path,
    tokenizer=model_path,
    gpu_memory_utilization=0.8,
)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # Generate 10 candidate answers for each question
)

questions = [item.get('question', '') for item in data]
answers   = [item.get('answer',   '') for item in data]
types     = [item.get('types',    '') for item in data]
image     = [item.get('image',    '') for item in data]

def base64_to_pil(b64_string):
    # 去掉 "data:image/png;base64," 头（如果有的话）
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

# 转换 image 列表
pil_images = []
for img_b64 in image:
    if img_b64:
        try:
            pil_images.append(base64_to_pil(img_b64))
        except Exception as e:
            print(f"[warning] Image decode failed: {e}")
            pil_images.append(None)
    else:
        pil_images.append(None)


# (Data preparation logic remains unchanged)
valid_chats = []
for i, (q, a, t, img) in enumerate(zip(questions, answers, types, pil_images)):
    if q and a and t and img:
        # valid_chats.append(
            # {'role': 'system', 'content': 'You are an AI visual question answering assistant. Given a question and its type, provide a clear and concise answer based only on the visual content described. Do not assume information not given and put your final answer within \\boxed{}.'},
            # {'role': 'user',   'content': [
            #     {"type": "text", "text": "This is a question: " + q + ". The question type is: " + t + "."},
            #     {"type": "image", "image": img} # 加入图像
            # ]}
            # {
            # "prompt": "USER: <image>\nYou are an AI visual question answering assistant. Given a question and its type, provide a clear and concise answer based only on the visual content described. Do not assume information not given and put your final answer within \\boxed{}. This is a question:" + q + "The question type is: " + t + "\nASSISTANT:",
            # "multi_modal_data": {"image": img},
            
            # }
        # )
        prompt = (
            "<|im_start|>system\n"
            "You are an AI visual question answering assistant. "
            "Answer questions based only on the visual content provided. "
            "You **must only output your final answer inside \\boxed{}**. "
            "Do not write explanations or any other text.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"This is a question: {q}. The question type is: {t}.\n"
            "**IMPORTANT:** Only output your answer in the form \\boxed{{answer}}.Do NOT include any units; provide only the numeric value or option.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        valid_chats.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        })
print('[server] Valid chat prompts have been prepared.')

# ---------- vLLM Generation ----------
# (vLLM generation logic remains unchanged)

# prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, add_special_tokens=True) for chat in valid_chats]
responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)

# responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)

for o in responses:
    generated_text = o.outputs[0].text
    print(generated_text)