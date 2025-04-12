from collections import defaultdict
import numpy as np

def get_error_stats(textual_answers, exact_answers, model_output_greedy, correctness_fn):
    """
    处理错误统计，同时处理边缘情况和索引错误
    """
    stats = defaultdict(list)
    
    # 首先检查数据结构的有效性
    has_exact_answers = (
        isinstance(exact_answers, dict) and 
        'exact_answer' in exact_answers and 
        'valid_exact_answer' in exact_answers and 
        len(exact_answers['exact_answer']) > 0
    )
    
    # 处理原始正确性，处理可能的缺失值或索引问题
    if 'automatic_correctness' not in model_output_greedy:
        print("Warning: automatic_correctness not found in model_output_greedy.")
        automatic_correctness = np.zeros(len(model_output_greedy))
    else:
        automatic_correctness = model_output_greedy['automatic_correctness'].values
    
    # 对于每个问题处理
    for q_idx in range(len(model_output_greedy)):
        stats['original_correctness'].append(automatic_correctness[q_idx] if q_idx < len(automatic_correctness) else 0)
        answer_sizes = []
        answer_accuracies = []
        exact_match_is_valid = []
        
        # 对每次重试处理
        for retry_idx in range(len(textual_answers)):
            try:
                if retry_idx < len(textual_answers) and q_idx < len(textual_answers[retry_idx]):
                    # 获取答案大小（字符数）
                    answer_sizes.append(len(textual_answers[retry_idx][q_idx]))
                    
                    # 尝试使用exact_answers，如果可用
                    if has_exact_answers:
                        # 验证索引是否有效
                        if (q_idx < len(exact_answers['exact_answer']) and
                            q_idx < len(exact_answers['valid_exact_answer'])):
                            # 检查exact_answer的结构，可能是二维或一维的
                            if isinstance(exact_answers['exact_answer'][q_idx], list):
                                if retry_idx < len(exact_answers['exact_answer'][q_idx]):
                                    exact_match_is_valid.append(1 if exact_answers['valid_exact_answer'][q_idx][retry_idx] else 0)
                                else:
                                    exact_match_is_valid.append(0)  # 默认不可用
                            else:
                                # 单个值，假设对所有重试都有效
                                exact_match_is_valid.append(1 if exact_answers['valid_exact_answer'][q_idx] else 0)
                        else:
                            exact_match_is_valid.append(0)  # 索引无效
                    else:
                        exact_match_is_valid.append(0)  # 没有exact_answers
                    
                    # 计算正确性
                    try:
                        if correctness_fn is not None:
                            # 安全调用correctness_fn，处理异常
                            try:
                                # 为了避免可能的列表或字符串解析问题，直接使用单元素列表
                                model_answer = textual_answers[retry_idx][q_idx]
                                correct_answer = model_output_greedy['correct_answer'].iloc[q_idx]
                                
                                # 确保正确答案格式兼容，避免eval错误
                                if isinstance(correct_answer, str) and ('[' in correct_answer or '{' in correct_answer):
                                    try:
                                        # 尝试安全解析，如果失败则使用原始字符串
                                        import ast
                                        eval_result = ast.literal_eval(correct_answer)
                                        if isinstance(eval_result, (list, dict)):
                                            correct_answer = eval_result
                                    except (SyntaxError, ValueError) as parse_err:
                                        print(f"安全解析错误，将使用原始字符串: {parse_err}")
                                
                                # 使用单元素列表调用函数
                                # result = correctness_fn([model_answer], [correct_answer])
                                # 根据函数名称确定需要传递的参数
                                if correctness_fn.__name__ == 'compute_correctness_winobias':
                                    # 检查是否有incorrect_answer列
                                    if 'incorrect_answer' in model_output_greedy.columns:
                                        wrong_label = model_output_greedy["incorrect_answer"].iloc[q_idx]
                                    elif 'wrong_label' in model_output_greedy.columns:
                                        wrong_label = model_output_greedy["wrong_label"].iloc[q_idx]
                                    else:
                                        print(f"警告: 找不到winobias所需的incorrect_answer列")
                                        wrong_label = ""
                                    # 使用3个参数调用
                                    result = correctness_fn([model_answer], [correct_answer], [wrong_label])
                                elif correctness_fn.__name__ == 'compute_correctness_winogrande':
                                    # 获取错误标签
                                    if 'incorrect_answer' in model_output_greedy.columns:
                                        wrong_label = model_output_greedy["incorrect_answer"].iloc[q_idx]
                                    elif 'wrong_label' in model_output_greedy.columns:
                                        wrong_label = model_output_greedy["wrong_label"].iloc[q_idx]
                                    else:
                                        print(f"警告: 找不到winogrande所需的incorrect_answer列")
                                        wrong_label = ""
                                    # 获取模型名称，如果无法获取则使用默认值
                                    model_name = model_output_greedy.get("model_name", ["unknown"])[0] if "model_name" in model_output_greedy else "unknown"
                                    # 使用4个参数调用
                                    result = correctness_fn([model_answer], [correct_answer], [wrong_label], model_name)
                                elif correctness_fn.__name__ == 'compute_correctness_natual_questions':
                                    # 尝试获取问题文本
                                    if 'question' in model_output_greedy.columns:
                                        question = model_output_greedy["question"].iloc[q_idx]
                                    else:
                                        # 如果找不到问题文本，使用空字符串
                                        question = ""
                                    # 使用3个基本参数调用
                                    result = correctness_fn([question], [model_answer], [correct_answer])
                                else:
                                    # 默认情况，使用2个参数调用
                                    result = correctness_fn([model_answer], [correct_answer])


                                # 检查result是否是字典并包含correctness键
                                if isinstance(result, dict) and 'correctness' in result:
                                    is_accurate = result['correctness'][0]
                                else:
                                    print(f"意外的correctness_fn返回格式: {result}")
                                    is_accurate = 0
                            except Exception as fn_error:
                                print(f"调用correctness_fn时出错: {fn_error}")
                                is_accurate = 0
                        else:
                            is_accurate = 0  # 默认不正确
                        
                        answer_accuracies.append(int(is_accurate))
                    except Exception as e:
                        print(f"计算correctness时出错，q_idx={q_idx}, retry_idx={retry_idx}: {e}")
                        answer_accuracies.append(0)  # 假设不正确
                else:
                    answer_sizes.append(0)
                    answer_accuracies.append(0)
                    exact_match_is_valid.append(0)
            except Exception as outer_e:
                print(f"外层循环异常，q_idx={q_idx}, retry_idx={retry_idx}: {outer_e}")
                answer_sizes.append(0)
                answer_accuracies.append(0)
                exact_match_is_valid.append(0)
        
        stats['answer_sizes'].append(answer_sizes)
        stats['answer_accuracies'].append(answer_accuracies)
        stats['exact_match_is_valid'].append(exact_match_is_valid)
    
    # 计算其他所需的统计数据
    stats = calculate_additional_stats(stats)
    
    return stats

def calculate_additional_stats(stats):
    """计算额外的统计信息，为get_types_of_mistakes函数准备数据"""
    # 计算各种聚合指标
    num_samples = len(stats['answer_accuracies'])
    
    # 生成wrong_answers列
    stats['wrong_answers'] = []
    stats['correct_answer_size'] = []
    stats['largest_incorrect_answer_size'] = []
    stats['n_wrong_answers'] = []
    
    for i in range(num_samples):
        # 收集此问题的答案准确性（处理可能的空列表情况）
        accuracies = stats['answer_accuracies'][i] if i < len(stats['answer_accuracies']) else []
        
        # 确保accuracies是有效的列表
        if not accuracies:
            # 如果accuracies为空，提供默认值
            stats['correct_answer_size'].append(0)
            stats['n_wrong_answers'].append(0)
            stats['wrong_answers'].append({"NO ANSWER": 0})
            stats['largest_incorrect_answer_size'].append(0)
            continue
        
        # 计算正确答案的数量
        correct_size = sum(a for a in accuracies if isinstance(a, (int, float)) and a > 0)
        stats['correct_answer_size'].append(correct_size)
        
        # 计算错误答案的数量
        n_wrong = sum(1 for a in accuracies if not (isinstance(a, (int, float)) and a > 0))
        stats['n_wrong_answers'].append(n_wrong)
        
        # 生成一个wrong_answers字典
        wrong_answers_dict = {"NO ANSWER": n_wrong} if n_wrong > 0 else {"NO ANSWER": 0}
        stats['wrong_answers'].append(wrong_answers_dict)
        
        # 计算最大不正确答案大小
        largest_incorrect = n_wrong  # 简化的模型，可以根据需要扩展
        stats['largest_incorrect_answer_size'].append(largest_incorrect)
    
    return stats

def get_types_of_mistakes(results, n_resamples):
    is_largest_no_answer = get_is_largest_no_answer(results)

    half_of_samples = n_resamples // 2
    third_of_samples = n_resamples // 3
    sixth_of_samples = n_resamples // 6

    TYPES_OF_MISTAKES = {
        "no_answer_is_largest": is_largest_no_answer,
        "wrong_is_largest_1": ((results.largest_incorrect_answer_size >= half_of_samples) & (is_largest_no_answer == 0) & (
                results.correct_answer_size > 0)).to_numpy(),
        "wrong_is_largest_2": ((results.largest_incorrect_answer_size >= half_of_samples) & (is_largest_no_answer == 0) & (
                results.correct_answer_size == 0)).to_numpy(),
        "right_is_largest_1": ((results.correct_answer_size >= half_of_samples) & (results.n_wrong_answers > 0)).to_numpy(),
        "right_is_largest_2": (results.correct_answer_size == n_resamples).to_numpy(),
        "many_different_answers_1": ((results.n_wrong_answers >= (third_of_samples - 1)) & (results.correct_answer_size > 0)).to_numpy(),
        "many_different_answers_2": ((results.n_wrong_answers >= third_of_samples) & (results.correct_answer_size == 0)).to_numpy(),
        "closely_competing_answers": (
                ((results['correct_answer_size'] - results['largest_incorrect_answer_size']).abs() <= sixth_of_samples) & (
                results['correct_answer_size'] > sixth_of_samples) & (
                        results['largest_incorrect_answer_size'] > sixth_of_samples)).to_numpy(),
        "all": np.array([True] * len(results))
    }
    return TYPES_OF_MISTAKES

def get_is_largest_no_answer(results):
    """
    确定"NO ANSWER"是否是最大的错误答案类别
    修复以处理可能缺失的列，并添加更多的错误处理
    """
    is_largest_no_answer = []
    
    # 检查结果是否包含必要的列
    if 'wrong_answers' not in results or 'correct_answer_size' not in results:
        print("Warning: results missing required columns for get_is_largest_no_answer")
        # 生成默认值
        return np.zeros(len(results), dtype=bool)
    
    try:
        for idx, (wrong_answers, correct_size) in enumerate(zip(results['wrong_answers'], results['correct_answer_size'])):
            try:
                # 处理字符串格式的wrong_answers（通常来自CSV文件）
                if isinstance(wrong_answers, str):
                    wa_str = wrong_answers.strip()
                    try:
                        wrong_answers_ = eval(wa_str)
                    except Exception as parse_error:
                        # 尝试修复不完整的JSON字符串
                        print(f"Error parsing wrong_answers at index {idx}: {parse_error}")
                        print(f"Raw data: '{wrong_answers}'")
                        wrong_answers_ = {"NO ANSWER": 0}  # 默认值
                else:
                    wrong_answers_ = wrong_answers
                
                # 验证wrong_answers是否为字典
                if not isinstance(wrong_answers_, dict):
                    print(f"Warning: wrong_answers at index {idx} is not a dictionary, type: {type(wrong_answers_)}")
                    is_largest_no_answer.append(False)
                    continue
                
                # 检查"NO ANSWER"键是否存在且是最大值
                if ('NO ANSWER' in wrong_answers_ and 
                    len(wrong_answers_) > 0 and
                    (len(wrong_answers_) == 1 or wrong_answers_['NO ANSWER'] == max(wrong_answers_.values())) and 
                    (wrong_answers_['NO ANSWER'] > correct_size)):
                    is_largest_no_answer.append(True)
                else:
                    is_largest_no_answer.append(False)
            except Exception as e:
                print(f"Error processing wrong_answers at index {idx}: {e}")
                is_largest_no_answer.append(False)
    except Exception as e:
        print(f"Error in get_is_largest_no_answer: {e}")
        # 生成默认值
        return np.zeros(len(results), dtype=bool)
    
    return np.array(is_largest_no_answer)