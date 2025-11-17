#!/usr/bin/env python3
"""
Evaluation script: Extract content within \\boxed{} from model output responses, and use LLM Judge to compare with ground truth answers to calculate accuracy
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from zai import ZhipuAiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='3B-VL-Qwen2.5-solver_1', help='Specify the model name to load')
args = parser.parse_args()

# args.model = '3B-VL-Qwen2.5'

GLM_MODEL = "GLM-4-Flash-250414"

def extract_boxed_content(text: str) -> str:
    if not text:
        return ""

    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last match (usually the final answer)
        return matches[-1].strip()
    
    # If no boxed format found, return the first 100 characters of the response (stripped)
    # This handles cases where the answer is directly output (e.g., "B", "C", etc.)
    return text.strip()[:100]


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    
    answer = str(answer).strip()
    answer = answer.upper()
    answer = re.sub(r'[,.\s]+', '', answer)
    return answer



@lru_cache(maxsize=10000)
def judge_answer_with_llm(predicted_answer: str, ground_truth_answer: str, question: str = "") -> bool:
    client = ZhipuAiClient(api_key="")
    
    user_content = f"""Please judge whether the following two answers express the same meaning. Please only answer "correct" or "incorrect". Correct answer: {ground_truth_answer}. Answer to be judged: {predicted_answer}. Judgment result (only answer "correct" or "incorrect"). You don't need to reason, if correct return 1, if incorrect return 0. Don't say anything else.""" 
    response = client.chat.completions.create( 
        model=GLM_MODEL, 
        messages=[ 
            {"role": "system", "content": "You are an answer evaluation assistant. Your task is to judge whether two answers are substantially equivalent. When evaluating, you should ignore superficial differences such as format, spaces, punctuation, case, etc., and focus on whether they are consistent in core content, logical meaning and information expression. The judgment criteria should be lenient and inclusive, as long as the expressed meaning is basically the same, it is considered equivalent."}, 
            {"role": "user", "content": user_content} 
            ], 
            temperature=0.6, 
        )
    
    result = response.choices[0].message.content.strip().lower()

    if result == '1' or result == 'correct' or result == 'yes' or result == 'true':
        return True
    else:
        return False


def load_ground_truth(parquet_file: str, file_name_filter: str = None) -> Dict[int, str]:
    df = pd.read_parquet(parquet_file)
    
    if file_name_filter and 'file_name' in df.columns:
        df = df[df['file_name'] == file_name_filter]
    
    ground_truth = {}
    for idx, row in df.iterrows():
        if 'answer' in row:
            ground_truth[idx] = str(row['answer']).strip()
    
    return ground_truth


def evaluate_dataset(predictions_file: str, ground_truth_file: str, use_llm_judge: bool = True, max_workers: int = 32, file_name_filter: str = None) -> Tuple[int, int, List[Dict]]:
    # Load ground truth answers
    ground_truth = load_ground_truth(ground_truth_file, file_name_filter)
    
    # Read all prediction results
    samples = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset_index = data.get('dataset_index', -1)
            if dataset_index in ground_truth:
                response = data.get('response', '')
                question = data.get('question', '')
                predicted = extract_boxed_content(response)
                samples.append({
                    'dataset_index': dataset_index,
                    'predicted': predicted,
                    'true_answer': ground_truth[dataset_index],
                    'question': question,
                    'response': response
                })
    
    total = len(samples)
    correct = 0
    errors = []
    
    if not use_llm_judge:
        # Traditional method: normalize answers for comparison
        for sample in samples:
            pred_normalized = normalize_answer(sample['predicted'])
            true_normalized = normalize_answer(sample['true_answer'])
            if pred_normalized == true_normalized:
                correct += 1
            else:
                errors.append({
                    'index': sample['dataset_index'],
                    'predicted': sample['predicted'],
                    'true_answer': sample['true_answer'],
                    'question': sample['question'][:200] + '...' if len(sample['question']) > 200 else sample['question'],
                    'response': sample['response'][:200] + '...' if len(sample['response']) > 200 else sample['response']
                })
    else:
        # Use multi-threaded concurrent LLM Judge calls
        def judge_sample(sample):
            if not sample['predicted']:
                return sample, False
            is_correct = judge_answer_with_llm(sample['predicted'], sample['true_answer'], sample['question'])
            return sample, is_correct
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(judge_sample, sample): sample for sample in samples}
            for future in as_completed(futures):
                sample, is_correct = future.result()
                if is_correct:
                    correct += 1
                else:
                    errors.append({
                        'index': sample['dataset_index'],
                        'predicted': sample['predicted'],
                        'true_answer': sample['true_answer'],
                        'question': sample['question'][:200] + '...' if len(sample['question']) > 200 else sample['question'],
                        'response': sample['response'][:200] + '...' if len(sample['response']) > 200 else sample['response']
                    })
    
    return correct, total, errors


def main():
    datasets = {
        'MMMU': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/MMMU.jsonl',
            '/yourpath/datasets/MMMU/data/test-00000-of-00001.parquet',
            None
        ),
        'MLLM_test/mm-vet': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
            '/yourpath/datasets/MLLM_test/test.parquet',
            'mm-vet'
        ),
        'MLLM_test/realWorldQA': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
            '/yourpath/datasets/MLLM_test/test.parquet',
            'realWorldQA'
        ),
        'visnumbench': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/visnumbench.jsonl',
            '/yourpath/datasets/visnumbench/data/test-00000-of-00001.parquet',
            None
        ),
        'MLLM_test/mathverse': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
            '/yourpath/datasets/MLLM_test/test.parquet',
            'mathverse'
        ),
        'MLLM_test/mathvision': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
            '/yourpath/datasets/MLLM_test/test.parquet',
            'mathvision'
        ),
        'hallusionbench': (
            f'/yourpath/vr-zero/Evaluation/Raw-Outputs/{args.model}/hallusionbench.jsonl',
            '/yourpath/datasets/hallusionbench/data/test-00000-of-00001.parquet',
            None
        ),
    }
    
    # Collect statistics for all datasets
    all_results = {}
    total_correct = 0
    total_samples = 0
    
    # Create results file in advance
    output_file = f'/yourpath/vr-zero/Evaluation/{args.model}_results2.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    print("=" * 80)
    print("Starting evaluation of accuracy for each dataset...")
    print(f"Using GLM Judge: {GLM_MODEL}")
    print(f"Results will be saved to: {output_file}")
    print("=" * 80)
    print()
    
    for dataset_name, (pred_file, truth_file, file_name_filter) in datasets.items():
        print(f"Evaluating: {dataset_name}")
        print(f"  Prediction file: {pred_file}")
        print(f"  Ground truth: {truth_file}")
        if file_name_filter:
            print(f"  Filter condition: file_name={file_name_filter}")
        
        try:
            correct, total, errors = evaluate_dataset(pred_file, truth_file, use_llm_judge=True, file_name_filter=file_name_filter)
            accuracy = correct / total * 100 if total > 0 else 0
            
            all_results[dataset_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy,
                'errors': errors
            }
            
            total_correct += correct
            total_samples += total
            
            print(f"  ✓ Correct: {correct}/{total}")
            print(f"  ✓ Accuracy: {accuracy:.2f}%")
            
            # Append results to file immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{dataset_name}: {correct}/{total} = {accuracy:.2f}%\n")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Evaluation error: {str(e)}")
            print()
            continue
    
    # Print overall statistics
    print("=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print()
    
    for dataset_name, result in all_results.items():
        print(f"{dataset_name:20s}: {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%")
    
    print("-" * 80)
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':20s}: {total_correct:4d}/{total_samples:4d} = {overall_accuracy:6.2f}%")
    print()
    
    # Append overall results to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Overall: {total_correct}/{total_samples} = {overall_accuracy:.2f}%\n")
    
    print(f"All results saved to: {output_file}")
    print("=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

