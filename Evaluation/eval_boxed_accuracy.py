#!/usr/bin/env python3
"""
评估脚本：从模型输出的 response 中提取 \\boxed{} 内的内容，并使用 LLM Judge 与真实答案比对，计算准确率
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
parser.add_argument('--model', type=str, default='3B-VL-Qwen2.5-solver_1', help='指定要加载的模型名称')
args = parser.parse_args()

# args.model = '3B-VL-Qwen2.5'

GLM_MODEL = "GLM-4-Flash-250414"

def extract_boxed_content(text: str) -> str:
    if not text:
        return ""

    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # 返回最后一个匹配（通常是最终答案）
        return matches[-1].strip()
    
    # 如果没有boxed格式，返回整个response的前100个字符（去除首尾空格）
    # 这样可以处理直接输出答案的情况（如 "B", "C" 等）
    return text.strip()[:100]


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    
    # 转为字符串并去除空格
    answer = str(answer).strip()
    
    # 转为大写以忽略大小写差异
    answer = answer.upper()
    
    # 移除常见的标点符号
    answer = re.sub(r'[,.\s]+', '', answer)
    
    return answer



@lru_cache(maxsize=10000)
def judge_answer_with_llm(predicted_answer: str, ground_truth_answer: str, question: str = "") -> bool:
    client = ZhipuAiClient(api_key="fb0e37380d124c0291982435fc3303e3.WgIM85O5ycMPHYtS")
    
    user_content = f"""请判断下面两个答案是否表达了相同的意思。请仅回答"正确"或"错误"。正确答案：{ground_truth_answer}.待评判答案：{predicted_answer}.判断结果（只回答"正确"或"错误"）.你不需要推理，如果正确返回 1，错误返回 0.不要说任何其他的东西。""" 
    response = client.chat.completions.create( 
        model=GLM_MODEL, 
        messages=[ 
            {"role": "system", "content": "你是一名答案评判助手。你的任务是判断两个答案是否实质等价。在评判时，应忽略格式、空格、标点、大小写等表面差异，重点关注两者在核心内容、逻辑含义和信息表达上是否一致。判断标准应宽松、包容，只要表达的意思基本相同，即视为等价。"}, 
            {"role": "user", "content": user_content} 
            ], 
            temperature=0.6, 
        )
    
    result = response.choices[0].message.content.strip().lower()
    print('result:', result)
    if result == '1' or result == '正确' or result == 'yes' or result == 'true' or result=="correct":
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
    # 加载真实答案
    ground_truth = load_ground_truth(ground_truth_file, file_name_filter)
    
    # 读取所有预测结果
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
        # 传统方法：规范化答案进行比较
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
        # 使用多线程并发调用 LLM Judge
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
    # 定义数据集映射
    # 格式：{预测文件名: (预测文件路径, 真实答案文件路径, file_name_filter)}
    datasets = {
        # 'mm-vet': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/mm-vet.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/mm-vet/test.parquet',
        #     None
        # ),
        # 'MLLM_test/clevr_count_70k': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'clevr_count_70k'
        # ),
        # 'MLLM_test/mathverse': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'mathverse'
        # ),
        # 'MLLM_test/mathvision': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'mathvision'
        # ),
        # 'MLLM_test/mathvista': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'mathvista'
        # ),
        # 'MLLM_test/mm-vet': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'mm-vet'
        # ),
        # 'MLLM_test/mmmu-pro': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
        #     'mmmu-pro'
        # ),
        'MLLM_test/realWorldQA': (
            f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MLLM_test.jsonl',
            '/data/hezhuangzhuang-p/datasets/MLLM_test/test.parquet',
            'realWorldQA'
        ),
        # 'visnumbench': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/visnumbench.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/visnumbench/data/test-00000-of-00001.parquet',
        #     None
        # ),
        # 'mmmu_pro_10options': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/mmmu_pro_10options.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/mmmu_pro_10options/data/test-00000-of-00001.parquet',
        #     None
        # ),
        # 'mmmu-pro-vision': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/mmmu-pro-vision.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/mmmu-pro-vision/data/test-00000-of-00001.parquet',
        #     None
        # ),
        # 'hallusionbench': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/hallusionbench.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/hallusionbench/data/test-00000-of-00001.parquet',
        #     None
        # ),
        # 'MMMU': (
        #     f'/data/hezhuangzhuang-p/vr-zero/Evaluation/Raw-Outputs/{args.model}/MMMU.jsonl',
        #     '/data/hezhuangzhuang-p/datasets/MMMU/data/test-00000-of-00001.parquet',
        #     None
        # ),
    }
    
    # 统计所有数据集的结果
    all_results = {}
    total_correct = 0
    total_samples = 0
    
    # 提前创建结果文件
    output_file = f'/data/hezhuangzhuang-p/vr-zero/Evaluation/{args.model}_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    print("=" * 80)
    print("开始评估各数据集的准确率...")
    print(f"使用 GLM Judge: {GLM_MODEL}")
    print(f"结果将保存到: {output_file}")
    print("=" * 80)
    print()
    
    for dataset_name, (pred_file, truth_file, file_name_filter) in datasets.items():
        print(f"正在评估: {dataset_name}")
        print(f"  预测文件: {pred_file}")
        print(f"  真实答案: {truth_file}")
        if file_name_filter:
            print(f"  过滤条件: file_name={file_name_filter}")
        
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
            
            print(f"  ✓ 正确: {correct}/{total}")
            print(f"  ✓ 准确率: {accuracy:.2f}%")
            
            # 立即追加写入结果文件
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{dataset_name}: {correct}/{total} = {accuracy:.2f}%\n")
            
            print()
            
        except Exception as e:
            print(f"  ✗ 评估出错: {str(e)}")
            print()
            continue
    
    # 打印总体统计
    print("=" * 80)
    print("总体统计")
    print("=" * 80)
    print()
    
    for dataset_name, result in all_results.items():
        print(f"{dataset_name:20s}: {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%")
    
    print("-" * 80)
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'总体':20s}: {total_correct:4d}/{total_samples:4d} = {overall_accuracy:6.2f}%")
    print()
    
    # 追加总体结果到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"总体: {total_correct}/{total_samples} = {overall_accuracy:.2f}%\n")
    
    print(f"所有结果已保存到: {output_file}")
    print("=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

