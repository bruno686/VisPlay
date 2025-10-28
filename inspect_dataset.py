import base64
import os
from io import BytesIO
from PIL import Image
from datasets import load_dataset

# Load the dataset - adjust the path as needed
# You'll need to specify the correct data_path here
data_path = input("Please enter the dataset path (or press Enter to use default): ").strip()
if not data_path:
    # Try to find the data path from common locations
    # You may need to adjust this
    data_path = "/data/hezhuangzhuang-p/vr-zero/storage/local_parquet/Qwen2.5-VL-3B-Instruct_solver_v1_train.parquet"  # Default path

if "@" in data_path:
    data_path, data_split = data_path.split("@")
else:
    data_split = "train"

print(f"Loading dataset from: {data_path}")

if os.path.isdir(data_path):
    dataset = load_dataset("parquet", data_dir=data_path, split="train")
elif os.path.isfile(data_path):
    dataset = load_dataset("parquet", data_files=data_path, split="train")
else:
    dataset = load_dataset(data_path, split=data_split)

print(f"Dataset loaded. Total samples: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

# Create output directory for images
output_dir = "dataset_inspection_output"
os.makedirs(output_dir, exist_ok=True)

# Function to decode and save base64 image
def save_base64_image(b64_str, output_path):
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(output_path)
        print(f"  ✓ Saved image to: {output_path}")
        return img.size  # Return image dimensions
    except Exception as e:
        print(f"  ✗ Error saving image: {e}")
        return None

# Inspect first 2 samples
for idx in [0, 1]:
    print(f"\n{'='*80}")
    print(f"Sample {idx}:")
    print(f"{'='*80}")
    
    sample = dataset[idx]
    
    # Print problem
    if 'problem' in sample:
        print(f"\nProblem:")
        print(f"  {sample['problem']}")
    
    # Print problem_type
    if 'problem_type' in sample:
        print(f"\nProblem Type:")
        print(f"  {sample['problem_type']}")
    
    # Print answer
    if 'answer' in sample:
        print(f"\nAnswer:")
        print(f"  {sample['answer']}")
    
    # Handle images
    if 'images' in sample:
        print(f"\nImages:")
        images = sample['images']
        
        # Handle both single image and list of images
        if not isinstance(images, list):
            images = [images]
        
        print(f"  Number of images: {len(images)}")
        
        for img_idx, img in enumerate(images):
            if isinstance(img, str):  # base64 string
                print(f"\n  Image {img_idx}:")
                print(f"    Format: base64 string")
                print(f"    Length: {len(img)} characters")
                
                # Save the image
                output_path = os.path.join(output_dir, f"sample_{idx}_image_{img_idx}.png")
                size = save_base64_image(img, output_path)
                if size:
                    print(f"    Dimensions: {size[0]}x{size[1]} pixels")
            else:
                print(f"\n  Image {img_idx}:")
                print(f"    Format: {type(img)}")
                print(f"    Content: {img}")
    
    # Print all available keys in the sample
    print(f"\nAll available keys in this sample:")
    for key in sample.keys():
        value = sample[key]
        if key == 'images':
            print(f"  - {key}: [already displayed above]")
        elif isinstance(value, str) and len(value) > 100:
            print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
        else:
            print(f"  - {key}: {value}")

print(f"\n{'='*80}")
print(f"Inspection complete! Images saved to: {output_dir}/")
print(f"{'='*80}")

