import datasets
import json
import re
import random
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None

def extract_last_final_answer(text):
    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = r'The answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None

def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)

def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

def get_prediction(output):
    solution = extract_solution(output)
    if solution is None:
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    for option in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        if option in solution:
            return option
    return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_file", type=str, default="outputs.json", help="File to save results")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=4,gpu_memory_utilization=0.85)
    dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')
    
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']
    # For each category store [correct_count, incorrect_count]
    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    answers = []
    
    print('----------------- Start Answering -------------------')
    
    for category in categories:
        category_entries = [entry for entry in dataset['test'] if entry['category'] == category]
        prompts = []
        for entry in category_entries:
            query = entry['question'] + '\n' + form_options(entry['options']) + '\n'
            messages = [{
                "role": "user",
                "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
            }]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                prompt = "user: " + query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
            prompts.append(prompt)
        
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=8192)
        outputs = llm.generate(prompts, sampling_params)
        
        for entry, output in zip(category_entries, outputs):
            answer = output.outputs[0].text
            entry['solution'] = answer
            answers.append(entry)
            
            prediction = get_prediction(answer)
            if entry["answer"] == prediction:
                success += 1
                per_category_accuracy[category][0] += 1
            else:
                fail += 1
                per_category_accuracy[category][1] += 1
            
        # Print category accuracy as soon as it's computed
        total_cat = per_category_accuracy[category][0] + per_category_accuracy[category][1]
        cat_accuracy = per_category_accuracy[category][0] / total_cat if total_cat > 0 else 0.0
        print(f"{category}: {cat_accuracy:.4f}")
    
    # Save all the answers in a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(answers, f, indent=2)
    
    # Calculate per-category report, micro average, and macro average
    print("\n----- Accuracy Report -----")
    category_accuracy_report = {}
    for category in categories:
        correct, incorrect = per_category_accuracy[category]
        total = correct + incorrect
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0
        category_accuracy_report[category] = accuracy
        print(f"{category}: {correct}/{total} -> {accuracy*100:.2f}% accuracy")
        
    total_predictions = success + fail
    micro_avg = success / total_predictions if total_predictions > 0 else 0.0
    print(f"\nMicro Average Accuracy: {micro_avg*100:.2f}%")
    with open('final_results.jsonl', 'a') as f:
        json.dump({"dataset": "mmlupro", "model": args.model_path, "accuracy": round(micro_avg*100, 2)}, f, indent=2)
    valid_categories = [cat for cat in categories if (per_category_accuracy[cat][0] + per_category_accuracy[cat][1] > 0)]
    if valid_categories:
        macro_avg = sum(category_accuracy_report[cat] for cat in valid_categories) / len(valid_categories)
    else:
        macro_avg = 0.0
    print(f"Macro Average Accuracy: {macro_avg*100:.2f}%")
