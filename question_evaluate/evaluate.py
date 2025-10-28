#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    This script evaluates generated answers against golden answers for a set of questions.
    It uses vLLM for efficient generation and a robust, timed grading mechanism to score the results.
    The script is designed to run as a batch job, often in parallel across multiple GPUs.

Refactoring Notes:
    - Replaced 'timeout-decorator' with the thread-safe 'stopit' library to provide robust
      timeout protection for the grading function without causing errors.
    - Optimized the answer comparison logic to perform cheap checks first, only calling the
      expensive grading function when necessary.
    - Improved error handling and code structure for better readability and stability.

Setup:
    pip install stopit transformers torch vllm

Example Usage (in a shell script):
    # This would run the script for GPU 0, with a specific model and save name.
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "Qwen/Qwen3-4B-Base" --suffix 0 --save_name "my_experiment" &
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os
import stopit  # Use the robust, thread-safe stopit library for timeouts
from mathruler.grader import extract_boxed_content, grade_answer
import base64
from io import BytesIO
from PIL import Image

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate generated questions using vLLM.")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=9, help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH", "/apdcephfs_sh2/share_300000800/user/heyicheng")
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"

# --- Timeout-Protected Grading Function ---
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    Wraps the mathruler 'grade_answer' function with a timeout.
    If the function takes too long, it returns 'TIMED_OUT' instead of hanging.
    """
    # The actual timeout value is passed as a keyword argument on each call.
    return grade_answer(res1, res2)

# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    # Clean up the input file immediately after loading to save space
    os.remove(INPUT_FILE)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
    exit()

# Adapt to new input format: each item has question, question_type, image (base64), answer (golden)
questions = [item.get("question", "") for item in data]
answers = [item.get("answer", "") for item in data]
question_types = [item.get("question_type", "") for item in data]
images_base64 = [item.get("image", "") for item in data]

# Filter out empty questions to avoid unnecessary generation
filtered = [(q, a, qt, img) for q, a, qt, img in zip(questions, answers, question_types, images_base64) if q]
if not filtered:
    print(f"[{args.suffix}] No valid questions found. Exiting.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

questions, answers, question_types, images_base64 = zip(*filtered)
print(f"[{args.suffix}] Found {len(questions)} questions to process.")

# 2. Initialize Model and Tokenizer
print(f"[{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    seed=int(args.suffix),
)
sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# 3. Generate Responses
print(f"[{args.suffix}] Generating {args.num_samples} samples for each question...")

# Build prompts for Qwen2.5-VL with image placeholder
placeholder = "<|image_pad|>"
prompts = [
    (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for q in questions
]

# Decode base64 images to PIL Images for vLLM multi-modal input
def b64_to_image(b64_str):
    try:
        img_bytes = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

images_pil = [b64_to_image(b64) for b64 in images_base64]

# Prepare valid chats with prompts and images
valid_chats = []
for prompt, img in zip(prompts, images_pil):
    if img is not None:
        valid_chat = {
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        }
        valid_chats.append(valid_chat)

# Generate responses using vLLM
responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)
print(f"[{args.suffix}] Generation complete.")

# 4. Process and Grade Responses
results_all = []
print(f"[{args.suffix}] Grading responses...")
for response, golden_answer, question, image_b64, question_type in zip(responses, answers, questions, images_base64, question_types):
    try:
        # Extract the boxed content from all generated samples
        results = [extract_boxed_content(output.text) for output in response.outputs]
        results = [res for res in results if res] # Filter out None/empty results

        if not results:
            print(f"[{args.suffix}] WARNING: No valid boxed answers found for question: '{question[:50]}...'")
            continue

        answer_counts = {}
        for result in results:
            matched = False
            for existing_answer in answer_counts:
                # OPTIMIZATION: Perform cheap string comparisons first.
                if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                    answer_counts[existing_answer] += 1
                    matched = True
                    break
                
                # If cheap checks fail, use the expensive, timed grader.
                # Check both directions (A vs B and B vs A).
                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{result[:30]}...' vs '{existing_answer[:30]}...'")
                    continue # Skip to the next existing_answer
                
                if match_1:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{existing_answer[:30]}...' vs '{result[:30]}...'")
                    continue

                if match_2:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

            if not matched:
                answer_counts[result] = 1

        if not answer_counts:
            continue

        # Determine the majority answer and its score
        majority_answer = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_answer]
        score = max_count / len(results)

        # Skip certain question types that are hard to grade automatically
        if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
            continue

        results_all.append({
            "question": question,
            "answer": majority_answer,
            "score": score,
            "image": image_b64,
            "question_type": question_type,
            'results': results
        })

    except Exception as e:
        print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
        continue

# 5. Save Final Results
print(f"[{args.suffix}] Processed {len(results_all)} questions. Saving results to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{args.suffix}] Script finished.")