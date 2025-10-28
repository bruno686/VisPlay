import json
import huggingface_hub
from datasets import Dataset, DatasetDict, Image as HFImage, Features, Value
from huggingface_hub import login
from PIL import Image
import argparse
import json
import os
import base64
import io
import gc
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
print(STORAGE_PATH)
with open('tokens.json', 'r') as f:
    token = json.load(f)['huggingface']
login(token=token)
parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen2.5-VL-3B-Instruct_solver_v1")
parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples to process at once")
parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (None for all)")
args = parser.parse_args()

# Function to convert base64 string to PIL Image
def base64_to_pil(base64_str):
    """Convert base64 string to PIL Image"""
    try:
        # Decode base64 string
        img_data = base64.b64decode(base64_str)
        # Create PIL Image from bytes
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

datas = []
# Load data from all version files (v1_0 to v1_7)
print(f"Loading data from experiment: {args.experiment_name}")
print(f"Base path: {STORAGE_PATH}/generated_question/")
print("-" * 80)

datas= []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

scores = [data['score'] for data in datas]
print(f"  - Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"  - Average score: {sum(scores)/len(scores):.4f}")

#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')
print(f"\nScore distribution saved to scores_distribution.png")

# Process data for HuggingFace with memory optimization
if not args.repo_name == "":
    print(f"Processing data with batch size: {args.batch_size}")
    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
    
    # First pass: filter and count valid samples
    valid_indices = []
    for i, data in enumerate(datas):
        if (data['score'] >= args.min_score and 
            data['score'] <= args.max_score and 
            data.get('answer') not in ['', 'None'] and
            'image' in data):
            valid_indices.append(i)
            if args.max_samples and len(valid_indices) >= args.max_samples:
                break
    
    print(f"Found {len(valid_indices)} valid samples")
    
    # Define features with Image type
    features = Features({
        'problem': Value('string'),
        'answer': Value('string'),
        'score': Value('float32'),
        'question_type': Value('string'),
        'image': HFImage()
    })
    
    # Process in batches to reduce memory usage
    all_filtered_datas = []
    batch_count = 0
    
    for batch_start in range(0, len(valid_indices), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        
        print(f"Processing batch {batch_count + 1}: samples {batch_start + 1}-{batch_end}")
        
        batch_filtered_datas = []
        for idx in batch_indices:
            data = datas[idx]
            
            # Convert base64 image to PIL Image
            pil_image = base64_to_pil(data['image'])
            
            if pil_image is not None:
                filtered_data = {
                    'problem': data['question'],
                    'answer': data['answer'],
                    'score': data['score'],
                    'question_type': data.get('question_type', 'unknown'),
                    'image': pil_image
                }
                batch_filtered_datas.append(filtered_data)
        
        all_filtered_datas.extend(batch_filtered_datas)
        batch_count += 1
        
        # Force garbage collection after each batch
        gc.collect()
        print(f"Batch {batch_count} completed. Total samples so far: {len(all_filtered_datas)}")
    
    print(f"Total filtered samples: {len(all_filtered_datas)}")
    
    # Create dataset in smaller chunks to reduce memory usage
    print("Creating dataset in chunks to reduce memory usage...")
    chunk_size = 500  # Smaller chunk size
    dataset_chunks = []
    
    for chunk_start in range(0, len(all_filtered_datas), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_filtered_datas))
        chunk_data = all_filtered_datas[chunk_start:chunk_end]
        
        print(f"Creating dataset chunk: {chunk_start + 1}-{chunk_end}")
        chunk_dataset = Dataset.from_list(chunk_data, features=features)
        dataset_chunks.append(chunk_dataset)
        
        # Clear memory immediately
        del chunk_data
        gc.collect()
        print(f"Chunk {chunk_start//chunk_size + 1} completed")
    
    # Combine chunks
    print("Combining dataset chunks...")
    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets(dataset_chunks)
    del dataset_chunks
    gc.collect()
    print("Dataset creation completed")
    
    # Clear the large list from memory
    del all_filtered_datas
    gc.collect()
    
    dataset_dict = {"train": train_dataset}
    config_name = f"{args.experiment_name}"
    dataset = DatasetDict(dataset_dict)
    
    print(f"Uploading to {HUGGINGFACENAME}/{args.repo_name} with config {config_name}")
    try:
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}", private=True, config_name=config_name)
        print("Upload complete!")
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Trying to save dataset locally first...")
        # Save dataset locally as backup
        local_path = f"./dataset_backup_{config_name}"
        dataset.save_to_disk(local_path)
        print(f"Dataset saved locally to: {local_path}")
        print("You can manually upload this dataset later")