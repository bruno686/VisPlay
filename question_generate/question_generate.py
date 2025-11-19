import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.datasets_loader import get_dataset_handler
import json
import regex as re
import os
from datasets import Dataset
import base64
from io import BytesIO
from PIL import Image
STORAGE_PATH = os.getenv("STORAGE_PATH")

def load_vqa_dataset(data_path, max_samples=None):
    """Load VQA dataset from parquet files"""
    datasets = []
    parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    parquet_files.sort()  # Sort to ensure consistent order
    
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file}...")
        dataset = Dataset.from_parquet(os.path.join(data_path, parquet_file))
        datasets.append(dataset)
    
    # Concatenate all datasets
    if len(datasets) > 1:
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(datasets)
    else:
        combined_dataset = datasets[0]
    
    if max_samples:
        combined_dataset = combined_dataset.select(range(min(max_samples, len(combined_dataset))))
    
    print(f"Total samples loaded: {len(combined_dataset)}")
    return combined_dataset

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

import math
from PIL import Image
from io import BytesIO

def process_image_for_vllm(image, max_pixels: int = 4194304, min_pixels: int = 262144):
    """
    Process image for vLLM multi-modal input, following ImageProcessMixin pattern.

    Args:
        image: PIL.Image.Image or dict with key "bytes" or raw bytes.
        max_pixels: maximum allowed total pixels (width * height). If exceeded, image will be downscaled.
        min_pixels: minimum allowed total pixels. If smaller, image will be upscaled.

    Returns:
        PIL.Image.Image in RGB mode, resized to be within [min_pixels, max_pixels].
    """
    # Accept dict / bytes / PIL.Image
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure image is loaded
    image.load()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.width, image.height
    total_pixels = width * height

    # Downscale if too large
    if total_pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # Upscale if too small
    elif total_pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    return image


def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results

def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        # gpu_memory_utilization=0.8,
        seed=int(args.suffix),
    )
    
    # Load VQA dataset
    print(f"Loading VQA dataset from {args.data_path}...")
    vqa_dataset = load_vqa_dataset(args.data_path, max_samples=args.max_samples)
    
    # Process each sample in the dataset
    results = []
    for i, sample in enumerate(vqa_dataset):
        if i >= args.num_samples:
            break
            
        print(f"Processing sample {i+1}/{min(args.num_samples, len(vqa_dataset))}")
        
        # Process image for vLLM and convert to base64 for storage
        processed_image = process_image_for_vllm(sample['images'])
        image_base64 = image_to_base64(processed_image)
        
        # Create prompt for qwen2.5-VL model
        system_prompt = """You are an intelligent Question Generator. Your task is to create a question based on the given image.  

                        **Requirements (must follow exactly):**  

                        1. Analyze the image carefully and understand all details.  
                        2. Generate **exactly one question** that is directly related to the image.  
                        3. Choose the question type from **only one** of the following:  
                        - `multiple choice` (Yes/No or four options labeled A, B, C, D; only one correct answer)  
                        - `numerical` (requires a specific numeric answer)  
                        - `regression` (requires predicting a continuous value, such as a measurement, quantity, or coordinate)  
                        4. The question must require analysis or reasoning, not just description.  
                        5. Provide the correct answer. Include units if applicable.  
                        6. **Output must be strictly in this format, with nothing else:**
                        7. Question type must be **only one** of: `multiple choice`, `numerical`, `regression`.  

                        The following THREE blocks:                     
                        <type>X</type>
                        <question>Y</question>
                        <answer>Z</answer>

                        **Strict rules:**  
                        - Do **not** use any other labels, punctuation, or formatting.  
                        - Do **not** add commentary, explanations, or extra text.  
                        - `X` must be exactly one of: `multiple choice`, `numerical`, or `regression`.  
                        - Always use the exact three-line structure above.  
                        - Do NOT include any units; provide only the numeric value or option. 
                        **Example of correct output:**   
                        <type>numerical</type>
                        <question>How many clubs are there in Florida?</question>
                        <answer>5.7M</answer>"""

        user_question = "Generate one new, challenging reasoning question based on this image. Remember to format the output exactly as instructed."
        
        # Create prompt in qwen2.5-VL format
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{user_question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        sample_params = vllm.SamplingParams(
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            n=1,
            stop_token_ids=[tokenizer.eos_token_id],
        )

        # Generate response for this sample
        # Prepare valid chat with prompt and processed image
        valid_chat = {
            "prompt": prompt,
            "multi_modal_data": {"image": processed_image}
        }
        
        completions: List[RequestOutput] = model.generate(
            [valid_chat], 
            sampling_params=sample_params
        )
        
        for completion in completions:
            response = completion.outputs[0].text
            try:
                # Extract question type, question, and answer
                question_types = re.findall(r"<type>(.*?)</type>", response, re.DOTALL)
                questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)

                if questions and answers:
                    question_type = question_types[-1].strip() if question_types else "unknown"
                    question = questions[-1].strip()
                    answer = answers[-1].strip()
                    results.append({
                        "question_type": question_type,
                        "question": question,
                        "answer": answer,
                        "image": image_base64
                    })
                else:
                    results.append({
                        "question_type": "unknown",
                        "question": response,
                        "answer": "",
                        "image": image_base64
                    })
            except Exception as e:
                print(f"Error processing response: {e}")
                results.append({
                    "question_type": "error",
                    "question": response,
                    "answer": "",
                    "image": image_base64
                })
    
    # Save results
    output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} questions and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="yourpath/datasets/Vision-SR1-47K", help="Path to VQA dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to load from dataset")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for output file")
    args = parser.parse_args()

    main(args) 