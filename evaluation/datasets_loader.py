from abc import ABC, abstractmethod
import re
from math_verify import parse, verify
import pandas
from datasets import load_dataset
import random
ANSWER_PATTERN_MULTICHOICE = r"(?:\$\$\s*)?\\boxed\{[^}]*?([A-Z])[^}]*\}(?:\s*\$\$)?|(?:\*{0,2}\s*)?(?:Final|Correct)\s*Answer:\s*([A-Z])\."
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"

class DatasetHandler(ABC):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED, num_examples: int = None):
        self.answer_pattern = answer_pattern
        self.num_examples = num_examples if num_examples is not None else 1

    @abstractmethod
    def load_data(self):
        """
        Load the dataset and return a tuple: (splits_dict, answer_type).

        splits_dict: A dictionary where each key is a split name (e.g., 'train', 'test')
                     and the value is the corresponding dataset or data structure.
        answer_type: A string describing the type of the answer, e.g.:
                     'number', 'text', 'option letter', etc.
        """
        pass

    def extract_answer(self, response: str) -> str:
        try:
            return re.search(self.answer_pattern, response).group(1)
        except:
            return None

    def compare_answer(self, response: str, answer: str) -> bool:
        response_answer = self.extract_answer(response)
        answer = str(answer)
        response_answer = str(response_answer)    
        if response_answer is None:
            return False
        if self.answer_pattern == ANSWER_PATTERN_MULTICHOICE:
            return response_answer == answer
        return verify(parse(answer), parse(response_answer))

    def get_score(self, responses: str, answers: str) -> float:
        scores = []
        for r,a in zip(responses, answers):
            if self.compare_answer(r,a):
                scores.append(1)
            else:
                scores.append(0)
        return scores, sum(scores)/len(scores)

class MathDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)

    def load_data(self):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        questions = [example['Question'] for example in examples]
        answers = [example['Answer'] for example in examples]

        return questions, answers

class Gsm8kDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("openai/gsm8k", 'main', split='test')
        examples = [row for row in dataset]
        questions = [example['question'] for example in examples]
        answers = [example["answer"].split('#### ')[-1] for example in examples]
        return questions, answers

class AmcDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("zwhe99/amc23", split='test')
        examples = [row for row in dataset]
        questions = [example['question'] for example in examples]  *32
        answers = [example['answer'] for example in examples]  *32

        return questions, answers

class MinervaDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("zwhe99/simplerl-minerva-math", split='test')
        examples = [row for row in dataset]
        questions = [example['problem'] for example in examples]
        answers = [example['answer'] for example in examples]

        return questions, answers

class OlympiadDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("zwhe99/simplerl-OlympiadBench", split='test')
        examples = [row for row in dataset]
        questions = [example['question'] for example in examples]
        answers = [example['final_answer'][0] for example in examples]

        return questions, answers

class Aime2024DatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("HuggingFaceH4/aime_2024", split='train')
        examples = [row for row in dataset]
        questions = [example['problem'] for example in examples]*32
        answers = [example['answer'] for example in examples]*32

        return questions, answers


class Aime2025DatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("yentinglin/aime_2025", 'default')['train']
        examples = [row for row in dataset]
        questions = [example['problem'] for example in examples]*32
        answers = [example['answer'] for example in examples]*32

        return questions, answers

class MmluProDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_MULTICHOICE):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
        examples = []
        for row in dataset:
            example = {
                'question': row['question'],
                'options': row['options'],
                'answer': row['answer'],
                'answer_index': row['answer_index'],
                'category': row['category'],
                'cot_content': row['cot_content'],
                'src': row['src']
            }
            examples.append(example)
        random.shuffle(examples)
        examples = examples[:1000]
        questions = []
        answers = []
        for example in examples:
            # Format question with options
            question = example['question'] + "\n\nOptions:\n"
            for i, opt in enumerate(example['options']):
                question += f"{chr(65+i)}. {opt}\n"
            
            questions.append(question)
            answers.append(example['answer'])
            
        return questions, answers

class bbehDatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_BOXED):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("MrLight/bbeh-eval", split='train')
        examples = [row for row in dataset]
        random.shuffle(examples)
        examples = examples[:1000]
        questions = [example['question'] for example in examples]
        answers = [example['answer'] for example in examples]

        return questions, answers

class SuperGPQADatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_MULTICHOICE):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset('m-a-p/SuperGPQA')
        examples = []
        for row in dataset['train']:
            example = {
                'question': row['question'],
                'options': row['options'],
                'answer': row['answer_letter']
            }
            examples.append(example)        
        random.shuffle(examples)
        examples = examples[:1000]
        
        questions = []
        answers = []
        for example in examples:
            # Format question with options
            question = example['question'] + "\n\nOptions:\n"
            for i, opt in enumerate(example['options']):
                question += f"{chr(65+i)}. {opt}\n"
            
            questions.append(question)
            answers.append(example['answer'])
            
        return questions, answers

class GPQA_DatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str = ANSWER_PATTERN_MULTICHOICE):
        super().__init__(answer_pattern)
    
    def load_data(self):
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond",'train')
        examples = []
        
        for row in dataset:
            # Get the question and answers
            question = row['Question']
            options = [
                row['Correct Answer'],
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3']
            ]
            # Shuffle options to randomize correct answer position
            random.shuffle(options)
            # Find the index of correct answer after shuffling
            correct_index = options.index(row['Correct Answer'])
            correct_option = chr(65 + correct_index)
            
            example = {
                'question': question,
                'options': options,
                'answer': correct_option
            }
            examples.append(example)
        
        # Shuffle and limit to 1000 examples like other handlers
        random.shuffle(examples)
        examples = examples[:1000]
        
        questions = []
        answers = []
        for example in examples:
            # Format question with options
            question = example['question'] + "\n\nOptions:\n"
            for i, opt in enumerate(example['options']):
                question += f"{chr(65+i)}. {opt}\n"
            
            questions.append(question)
            answers.append(example['answer'])
            
        return questions, answers


class Mydataset_DatasetHandler(DatasetHandler):
    def __init__(self, answer_pattern: str =  ANSWER_PATTERN_BOXED, name: str = "qwen3_frequent_solver_v1"):
        super().__init__(answer_pattern)
        self.name = name
    def load_data(self):
        dataset = load_dataset(self.name)['train']
        examples = []
        
        for row in dataset:
            example = {
                'question': row['problem'],
                'answer': row['answer']
            }
            examples.append(example)
        
        # Shuffle and limit to 1000 examples like other handlers
        random.shuffle(examples)
        # examples = examples[:1000]
        
        questions = []
        answers = []
        for example in examples:

            questions.append(example['question'])
            answers.append(example['answer'])
            
        return questions, answers

def get_dataset_handler(dataset_name: str,name: str = None) -> DatasetHandler:
    if dataset_name == "math":
        return MathDatasetHandler()
    elif dataset_name == "gsm8k":
        return Gsm8kDatasetHandler()
    elif dataset_name == "amc":
        return AmcDatasetHandler()
    elif dataset_name == "minerva":
        return MinervaDatasetHandler()
    elif dataset_name == "olympiad":
        return OlympiadDatasetHandler()
    elif dataset_name == "aime2024":
        return Aime2024DatasetHandler()
    elif dataset_name == "aime2025":
        return Aime2025DatasetHandler()
    elif dataset_name == "mmlu_pro":
        return MmluProDatasetHandler()
    elif dataset_name == "bbeh":
        return bbehDatasetHandler()
    elif dataset_name == "super_gpqa":
        return SuperGPQADatasetHandler()
    elif dataset_name == "gpqa":
        return GPQA_DatasetHandler()
    elif dataset_name == "mydataset":
        return Mydataset_DatasetHandler(name=name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    

if __name__ == "__main__":
    print("mmlu_pro")
    for dataset_name in ["gpqa"]:
        print(f"Loading {dataset_name} dataset")
        handler = get_dataset_handler(dataset_name)
        questions, answers = handler.load_data()
        print(questions[0])
        print('-'*100)
        print(answers[0])