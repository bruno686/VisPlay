import json
import pandas as pd
import argparse
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
print(STORAGE_PATH)
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="", help="Output directory for parquet files")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for input and output files")
args = parser.parse_args()

datas= []
# Find all matching result files
import glob
result_files = glob.glob(f'{STORAGE_PATH}/generated_question/{args.save_name}_*_results.json')
print(f"Found {len(result_files)} result files: {result_files}")

for file_path in result_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            datas.extend(data)
        print(f"Loaded {len(data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue


# Clean up result files after processing
# for file_path in result_files:
#     try:
#         os.remove(file_path)
#         print(f"Removed {file_path}")
#     except Exception as e:
#         print(f"Error removing {file_path}: {e}")
#         continue

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')

# Filter and save data as parquet files
filtered_datas = [{'problem':data['question'],'answer':data['answer'],'score':data['score'],'images':data.get('image', ''),'problem_type':data.get('question_type', 'unknown')} for data in datas if data['score'] >= args.min_score and data['score'] <= args.max_score and data['answer'] != '' and data['answer']!= 'None']
print(f"Filtered {len(filtered_datas)} samples with score between {args.min_score} and {args.max_score}")

if filtered_datas:
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.save_name}_train.parquet")
    else:
        # Default to STORAGE_PATH/generated_question/
        output_dir = f"{STORAGE_PATH}/local_parquet"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.save_name}_train.parquet")
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(filtered_datas)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(filtered_datas)} samples to {output_path}")
    
    # Also save a summary file
    summary_path = output_path.replace('.parquet', '_summary.json')
    summary = {
        "total_samples": len(filtered_datas),
        "score_range": [args.min_score, args.max_score],
        "experiment_name": args.save_name,
        "output_file": output_path
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary to {summary_path}")
else:
    print("No data to save after filtering")







