import argparse
import os
import glob

import torch

from probing_utils import LIST_OF_MODELS, LIST_OF_DATASETS,LIST_OF_TEST_DATASETS, MODEL_FRIENDLY_NAMES

parser = argparse.ArgumentParser(
    description='Merge separate resampling runs')
parser.add_argument("--model", choices=LIST_OF_MODELS)
parser.add_argument("--dataset", choices=LIST_OF_DATASETS+LIST_OF_TEST_DATASETS,
                    required=True)
parser.add_argument("--n_resamples", type=int, default=10,
                    help="Number of resamples to merge into final file")

args = parser.parse_args()

names = ['textual_answers', 'input_output_ids', 'exact_answers']
model = MODEL_FRIENDLY_NAMES[args.model]
dataset = args.dataset

resampling_dir = "../output/resampling"
if not os.path.exists(resampling_dir):
    os.makedirs(resampling_dir)
    print(f"Created directory: {resampling_dir}")

# 打印当前工作目录和要查找的目录
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for resampling files in: {os.path.abspath(resampling_dir)}")

for name in names:
    # 使用通配符查找所有匹配的重采样文件
    pattern = f"{resampling_dir}/{model}_{dataset}_2_{name}*.pt"
    matching_files = glob.glob(pattern)
    print(f"Found {len(matching_files)} files matching pattern: {pattern}")
    
    if not matching_files:
        print(f"Warning: No files found for {name} with pattern {pattern}. Will try other patterns.")
        # 尝试另一个模式，可能有些文件用了不同的命名方式
        pattern = f"{resampling_dir}/{model}_{dataset}_*_{name}_*.pt"
        matching_files = glob.glob(pattern)
        print(f"Second attempt found {len(matching_files)} files with pattern: {pattern}")
        
        if not matching_files and name == 'exact_answers':
            print(f"Skipping {name} as no files found and it may be optional.")
            continue
    
    outputs = []
    
    # 如果找到了文件，按照顺序加载它们
    for file_path in sorted(matching_files):
        try:
            print(f"Loading: {file_path}")
            output = torch.load(file_path)
            if isinstance(output, list):
                outputs.extend(output)
            else:
                # 对于exact_answers这样的字典类型
                if name == 'exact_answers' and isinstance(output, dict):
                    if not outputs:
                        outputs = output
                    else:
                        # 合并字典中的列表
                        for key in output:
                            if key in outputs:
                                outputs[key].extend(output[key])
                            else:
                                outputs[key] = output[key]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not outputs:
        print(f"Warning: No data loaded for {name}, skipping.")
        continue
    
    # 在保存结果之前，确保exact_answers的结构正确
    if name == 'exact_answers' and isinstance(outputs, dict):
        # 检查exact_answers结构是否需要调整
        print("Checking exact_answers structure...")
        try:
            if ('exact_answer' in outputs and 'valid_exact_answer' in outputs and 
                    len(outputs['exact_answer']) > 0):
                # 检查数据结构是否一致
                if not isinstance(outputs['exact_answer'][0], list):
                    print("Converting exact_answers structure to match expected format...")
                    # 将一维数组转换为二维结构
                    n_samples = len(outputs['exact_answer'])
                    outputs['exact_answer'] = [[outputs['exact_answer'][i] for i in range(n_samples)]]
                    outputs['valid_exact_answer'] = [[outputs['valid_exact_answer'][i] for i in range(n_samples)]]
                
                # 如果是IMDB数据集，确保exact_answer格式为数字字符串
                if dataset == 'imdb' or dataset == 'imdb_test':
                    print("Ensuring IMDB exact_answers are in numeric format...")
                    for i in range(len(outputs['exact_answer'])):
                        for j in range(len(outputs['exact_answer'][i])):
                            # 确保使用正确的映射: positive->1, negative->0
                            if outputs['exact_answer'][i][j] == 'positive':
                                outputs['exact_answer'][i][j] = '1'
                            elif outputs['exact_answer'][i][j] == 'negative':
                                outputs['exact_answer'][i][j] = '0'
        except Exception as e:
            print(f"Error checking exact_answers structure: {e}")

    n_total_resamples = args.n_resamples
    output_path = f"{resampling_dir}/{model}_{dataset}_{n_total_resamples}_{name}.pt"
    print(f"Saving {len(outputs) if isinstance(outputs, list) else 'dict'} items to: {output_path}")
    torch.save(outputs, output_path)
    print(f"Successfully saved: {output_path}")