import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
from evaluation.datasets_loader import get_dataset_handler
import json
import regex as re
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")

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
    dataset_handler = get_dataset_handler("math")
    questions, answers = dataset_handler.load_data()
    question = questions[0]
    answer = answers[0]
    chat = [
        {
            "role": "system",
            "content": (
                """
                    You are an intelligent Question Generator. Your task is to create a question based on the given image.  

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
                    **Example of correct output:**   
                    <type>numerical</type>
                    <question>How many clubs are there in Florida?</question>
                    <answer>5.7M</answer>  """
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat, 
            tokenize=False,
            add_generation_prompt=True, 
            add_special_tokens=True
        )
    else:
        prompt = "system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"]
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    completions: List[RequestOutput] = model.generate([prompt]*args.num_samples, sampling_params=sample_params)
    results=[]
    for completion in completions:
        response = completion.outputs[0].text
        try:
            questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
            answers = extract_boxed(response)

            if questions and answers:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer, "score": 0})
            else:
                results.append({"question": response, "answer": "", "score": -1})
        except:
            results.append({"question": response, "answer": "", "score": -1})
    with open(f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1250, help="Number of samples to generate")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="", help="")
    args = parser.parse_args()

    main(args) 