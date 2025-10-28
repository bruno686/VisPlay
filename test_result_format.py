#!/usr/bin/env python3
"""
Test script to verify the updated result format with question_type
"""
import sys
import os
sys.path.append('.')

from datasets import Dataset
import base64
from io import BytesIO
from PIL import Image

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

def test_result_format():
    """Test the updated result format"""
    print("Testing updated result format...")
    
    # Load a sample
    dataset = load_vqa_dataset('/data/hezhuangzhuang-p/datasets/Vision-SR1-47K', max_samples=1)
    sample = dataset[0]
    
    # Convert image to base64
    image_base64 = image_to_base64(sample['images'])
    
    # Test different result scenarios
    test_results = [
        {
            "question_type": "multiple choice",
            "question": "What is the main color of the object in the image?",
            "answer": "A",
            "image": image_base64
        },
        {
            "question_type": "numerical", 
            "question": "How many objects are visible in the image?",
            "answer": "5",
            "image": image_base64
        },
        {
            "question_type": "regression",
            "question": "What is the approximate area of the largest object?",
            "answer": "25.5",
            "image": image_base64
        },
        {
            "question_type": "unknown",
            "question": "Invalid response format",
            "answer": "",
            "image": image_base64
        },
        {
            "question_type": "error",
            "question": "Error occurred during processing",
            "answer": "",
            "image": image_base64
        }
    ]
    
    print("Test result formats:")
    for i, result in enumerate(test_results):
        print(f"\nResult {i+1}:")
        print(f"  Keys: {list(result.keys())}")
        print(f"  Question type: {result['question_type']}")
        print(f"  Question: {result['question']}")
        print(f"  Answer: {result['answer']}")
        print(f"  Image length: {len(result['image'])}")
    
    print("\nAll result format tests passed!")

if __name__ == "__main__":
    test_result_format()
