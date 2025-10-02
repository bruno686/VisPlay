# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This reward function is for regular [CoT] -> [Answer] GRPO finetuning
'''

import re, os, json
from typing import Dict, List, Optional
import time
import random
from mathruler.grader import extract_boxed_content, grade_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
STORAGE_PATH = os.getenv("STORAGE_PATH")
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^\s*<type>(multiple choice|numerical|regression)</type>\s*"
        r"<question>.+?</question>\s*"
        r"<answer>.+?</answer>\s*$",
        re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict.strip()) else 0.0

def extract_description(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<description>([\s\S]*?)</description>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()

def extract_answer(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<answer>([\s\S]*?)</answer>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()

def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    # answer = extract_answer(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def match(generation):
    pattern = r"<type>(.*?)</type>.*?<question>(.*?)</question>.*?<answer>(.*?)</answer>"
    match = re.search(pattern, generation, re.DOTALL)

    if match:
        result = {
            "type": match.group(1).strip(),
            "question": match.group(2).strip(),
            "answer": match.group(3).strip()
        }
    return result

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def fetch(index,i):
    response = requests.get(f"http://0.0.0.0:{6000+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    datas = split_list(data,4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(4)]

        for future in as_completed(futures):
            print(future.result())

    for future in as_completed(futures):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))

    return final_results


def compute_score(predicts: List[str], ground_truths: List[str], questions: List[str], description_answers: List[str], format_weight: float = 0.1, images: Optional[List[object]] = None) -> List[Dict[str, float]]:
    scores = []
    results = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        # format_score = format_reward(predict)
        # accuracy_score = accuracy_reward(predict, ground_truth)
        dirty_results = match(predict)
        questions = dirty_results['question']
        types = dirty_results['type']
        answers = dirty_results['answer']


        if questions and answers and types:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer, "types": types})
            except:
                results.append({"question": "", "answer": "", "types": ""})
        else:
            results.append({"question": "", "answer": "", "types": ""})
        
        final_results = generate_results(results)
        penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)

        assert len(penalty) == len(final_results)
        scores = []
        for i in range(len(final_results)):
            final_score = (min(final_results[i]["score"],1-final_results[i]["score"]) if final_results[i]['question'] else -1)-penalty[i]
            scores.append({"overall": final_score,"format": 1 if final_results[i]['question'] else 0,"accuracy": penalty[i]})
        return scores

