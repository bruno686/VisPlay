#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

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
import stopit  # 1. Import the thread-safe 'stopit' library
import sys; sys.path.insert(0, "/data/hezhuangzhuang-p/vr-zero")
from verl.utils.tokenizer import get_processor
from PIL import Image

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='6000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
processor = get_processor(args.model_path, trust_remote_code=True)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    limit_mm_per_prompt={"image": 1},
    disable_mm_preprocessor_cache=True,
)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # Generate 10 candidate answers for each question
)

valid_chats.append([
    {'role': 'system', 'content': 'You are an AI visual question answering assistant. Given a question and its type, provide a clear and concise answer based only on the visual content described. Do not assume information not given and put your final answer within \\boxed{}.'},
    {
    "role": "user",
    "content": [
        {"type": "text", "text": "This is a question: " + q + ". The question type is: " + t + "."},
        {"type": "image", "image": image}   # 加入图像
    ]
    }            
])


# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]
    types     = [item.get('types',   '')  for item in data]
    images    = [item.get('image',   None) for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_types, valid_images, valid_chats = [], [], [], [], [], []
    multi_modal_data = []
    for i, (q, a, t, image) in enumerate(zip(questions, answers, types, images)):
        if q and a and t and image:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_types.append(t)
            valid_images.append(image)
            # Build chat with image token if available
            valid_chats.append([
                {'role': 'system', 'content': 'You are an AI visual question answering assistant. Given a question and its type, provide a clear and concise answer based only on the visual content described. Do not assume information not given and put your final answer within \\boxed{}.'},
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a question: " + q + ". The question type is: " + t + "."},
                    {"type": "image", "image": image}   # 加入图像
                ]
                }            
    ])
            # Ensure image is PIL; convert to RGB only if necessary
            try:
                if isinstance(image, Image.Image):
                    img = image if image.mode == 'RGB' else image.convert('RGB')
                else:
                    # If somehow not PIL, try opening from path-like
                    img = Image.open(image)
                    img = img if img.mode == 'RGB' else img.convert('RGB')
                multi_modal_data.append({"image": [img]})
            except Exception as e:
                print(f"[server] Failed to handle image at index {i}: {e}")
                valid_indices.pop()
                valid_questions.pop()
                valid_answers.pop()
                valid_types.pop()
                valid_images.pop()
                valid_chats.pop()
                continue
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # Prefer processor.apply_chat_template for multimodal models; fallback to tokenizer
    if valid_chats:
        try:
            prompts = [
                processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in valid_chats
            ]
        except Exception as e:
            print(f"[server] processor.apply_chat_template failed, fallback to tokenizer: {e}")
            if tokenizer.chat_template:
                prompts = [
                    tokenizer.apply_chat_template(chat, tokenize=False,
                                                  add_generation_prompt=True, add_special_tokens=True)
                    for chat in valid_chats
                ]
            else:
                prompts = [
                    'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content'][0]['text'] + ' <image>'
                    for chat in valid_chats
                ]
        # Sanity logs
        print(f"[server] Prepared prompts: {len(prompts)}, multi_modal_data: {len(multi_modal_data)}, valid_chats: {len(valid_chats)}")
        if prompts:
            _preview = (prompts[0][:160]).replace('\n', ' ')
            print(f"[server] First prompt preview: {_preview} ...")
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True, multi_modal_data=multi_modal_data)
    else:
        responses = []
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        if not any(results):
            raw_preview = response.outputs[0].text[:200].replace('\n', ' ') if response.outputs else ''
            print(f"[server] WARN: empty extracted results for question preview '{question[:60]}...'. Raw first output: {raw_preview}")
        # print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res: continue # Skip empty results
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score if majority_ans == golden_answer and score > 0.1 else 0,
            'results':  results
        }

    results_all = []
    response_idx = 0
    valid_idx_set = set(valid_indices)
    for i, (q, a) in enumerate(zip(questions, answers)):
        try:
            if i in valid_idx_set:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')