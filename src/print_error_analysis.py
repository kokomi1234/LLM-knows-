import pandas as pd
import torch
from compute_correctness import compute_correctness_triviaqa
from resamples_utils import get_error_stats, get_types_of_mistakes

def load_data_safely(filepath):
    """安全加载数据文件"""
    try:
        return torch.load(filepath, weights_only=True, map_location='cpu')
    except Exception as e:
        print(f"使用 weights_only=True 加载失败，尝试使用默认设置: {e}")
        return torch.load(filepath, map_location='cpu')

def print_error_analysis(model_name="mistral-7b-instruct", dataset="triviaqa", n_resamples=1):
    """打印错误分析结果"""
    print(f"===== {model_name} 在 {dataset} 数据集上的错误分析 =====")
    
    try:
        # 加载数据
        csv_path = f'../output/{model_name}-answers-{dataset}.csv'
        model_output = pd.read_csv(csv_path)
        print("\n1. 基本统计:")
        print(f"总样本数: {len(model_output)}")
        print(f"正确答案数: {model_output['automatic_correctness'].sum()}")
        print(f"准确率: {model_output['automatic_correctness'].mean():.2%}")
        
        # 安全加载重采样结果
        textual_answers = load_data_safely(
            f"../output/resampling/{model_name}_{dataset}_{n_resamples}_textual_answers.pt"
        )
        exact_answers = load_data_safely(
            f"../output/resampling/{model_name}_{dataset}_{n_resamples}_exact_answers.pt"
        )
        
        # 获取错误统计
        error_stats = get_error_stats(textual_answers, exact_answers, model_output, compute_correctness_triviaqa)
        error_stats = pd.DataFrame.from_dict(error_stats)
        
        print("\n2. 错误类型分布:")
        TYPES_OF_MISTAKES = get_types_of_mistakes(error_stats, n_resamples=n_resamples)
        
        total_errors = 0
        for error_type, mask in TYPES_OF_MISTAKES.items():
            if error_type != 'all':
                count = mask.sum()
                total_errors += count
                percent = count / len(mask) * 100
                print(f"{error_type:25s}: {count:4d} 样本 ({percent:5.1f}%)")
        print("-" * 50)
        print(f"总错误数: {total_errors}")
        
        print("\n3. 错误示例分析:")
        errors = model_output[model_output['automatic_correctness'] == 0].head(20)
        for idx, row in errors.iterrows():
            print("\n" + "="*80)
            print(f"错误示例 #{idx+1}")
            print(f"评论摘要: {row['raw_question'][:150]}...")
            print(f"模型答案: {row['model_answer']}")
            print(f"正确答案: {row['correct_answer']}")
            print(f"模型标签: {row['exact_answer']}")
            print(f"问题类型: {'分类错误' if row['exact_answer'] != str(row['correct_answer']) else '其他错误'}")
            
    except FileNotFoundError as e:
        print(f"找不到必要的文件: {e}")
        print(f"请确保以下文件存在:\n- {csv_path}")
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='错误分析工具')
    parser.add_argument('--model', default='mistral-7b-instruct', help='模型名称')
    parser.add_argument('--dataset', default='triviaqa', help='数据集名称')
    parser.add_argument('--n_resamples', type=int, default=1, help='重采样次数')
    args = parser.parse_args()
    
    print_error_analysis(args.model, args.dataset, args.n_resamples)