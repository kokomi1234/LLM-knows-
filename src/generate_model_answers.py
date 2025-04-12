import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed

from compute_correctness import compute_correctness
from probing_utils import load_model_and_validate_gpu, tokenize, generate, LIST_OF_DATASETS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_MODELS,LIST_OF_TEST_DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="A script for generating model answers and outputting to csv")
    parser.add_argument("--model",
                        choices=LIST_OF_MODELS,
                        required=True)
    parser.add_argument("--dataset",
                        choices=LIST_OF_DATASETS + LIST_OF_TEST_DATASETS)
    parser.add_argument("--verbose", action='store_true', help='print more information')
    parser.add_argument("--n_samples", type=int, help='number of examples to use', default=None)
    parser.add_argument("--model_dir", type=str, 
                        help='模型下载和加载的目录',
                        default=os.path.join("..", "model"))

    return parser.parse_args()


def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'../data/{file_name}_test.csv'
    else: # train
        file_path = f'../data/{file_name}_train.csv'
    if not os.path.exists(file_path):
        # split into train and test
        data = pd.read_csv(f"../data/{file_name}.csv")
        # spit into test and train - 50% each
        train, test = train_test_split(data, train_size=10000, random_state=42)
        train.to_csv(f"../data/{file_name}_train.csv", index=False)
        test.to_csv(f"../data/{file_name}_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    return questions, answers


def load_data_nli(split, data_file_names):
    data_folder = '../data'
    if split == 'train':
        file_path = f"{data_folder}/{data_file_names['train']}.csv"
    elif split == 'test':
        file_path = f"{data_folder}/{data_file_names['test']}.csv"
    else:
        raise ValueError("split should be either train or test")

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    origin = data['Origin']

    return questions, answers, origin

def load_data_snli(split):
    data_file_names = {
        'train': 'snli_train',
        'test': 'snli_validation'
    }
    return load_data_nli(split, data_file_names)

def load_data_mnli(split):
    data_file_names = {
        'train': 'mnli_train',
        'test': 'mnli_validation'
    }
    return load_data_nli(split, data_file_names)

def load_data_nq(split, with_context=False):
    raw_data_folder = '../data'
    data_folder = '../data'
    file_name = 'nq_wc' # don't need special file for no context, simply don't use it
    if split == 'train':
        file_path = f'{data_folder}/{file_name}_dataset_train.csv'
    elif split == 'test':
        file_path = f'{data_folder}/{file_name}_dataset_test.csv'
    else:
        raise ValueError("split should be either train or test")
    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"{raw_data_folder}/{file_name}_dataset.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"{data_folder}/{file_name}_dataset_train.csv", index=False)
        test.to_csv(f"{data_folder}/{file_name}_dataset_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']

    if with_context:
        context = data['Context']
    else:
        context = None

    return questions, answers, context


def load_data_winogrande(split):
    data_folder = '../data'
    if split == 'train':
        file_path = f"{data_folder}/winogrande_train.csv"
    elif split == 'test':
        file_path = f"{data_folder}/winogrande_test.csv"
    else:
        raise ValueError("split should be either train or test")

    if not os.path.exists(file_path):
        all_data = pd.read_csv(f"{data_folder}/winogrande.csv")
        train, test = train_test_split(all_data, train_size=10000, random_state=42)
        train.to_csv(f"{data_folder}/winogrande_train.csv", index=False)
        test.to_csv(f"{data_folder}/winogrande_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    wrong_answers = data['Wrong_Answer']

    return questions, answers, wrong_answers


def load_data_triviaqa(test=False, legacy=False):
    if legacy:
        with open('../data/verified-web-dev.json', 'r', encoding='utf-8') as f:
            data_verified = json.load(f)
            data_verified = data_verified['Data']
        with open('../data/web-dev.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['Data']
        questions_from_verified = [x['Question'] for x in data_verified]
        data_not_verified = []
        for x in data:
            if x['Question'] in questions_from_verified:
                pass
            else:
                data_not_verified.append(x)

        print("Length of not verified data: ", len(data_not_verified))
        print("Length of verified data: ", len(data_verified))

        if test:
            return [ex['Question'] for ex in data_verified], [ex['Answer']['Aliases'] for ex in data_verified]
        else:
            return [ex['Question'] for ex in data_not_verified], [ex['Answer']['Aliases'] for ex in data_not_verified]
    else:
        if test:
            file_path = '../data/triviaqa-unfiltered/unfiltered-web-dev.json'
        else:
            file_path = '../data/triviaqa-unfiltered/unfiltered-web-train.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['Data']
        data, _ = train_test_split(data, train_size=10000, random_state=42)
        return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def load_data_math(test=False):
    if test:
        data = pd.read_csv("../data/AnswerableMath_test.csv")
    else:
        data = pd.read_csv("../data/AnswerableMath.csv")


    questions = data['question']
    answers = data['answer'].map(lambda x: eval(x)[0])
    return questions, answers

def math_preprocess(model_name, all_questions, labels):
    prompts = []

    if 'instruct' in model_name.lower():
        for q in all_questions:
            prompts.append(q + " Answer shortly.")
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
            A:''')
    return prompts

def prepare_winogrande(model_name, all_questions, labels):
    prompts = []
    for q in all_questions:
        prompts.append(f'''Question: {q}\nAnswer:''')
    return prompts

def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False, output_scores=False,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):

    all_textual_answers = []
    all_scores = []
    all_input_output_ids = []
    all_output_ids = []
    counter = 0
    
    print(f"开始生成模型回答，共有 {len(data)} 条数据...")
    
    for prompt in tqdm(data):
        print(f"步骤1: 处理第 {counter+1}/{len(data)} 条数据...")
        
        model_input = tokenize(prompt, tokenizer, model_name).to(device)
        print(f"步骤2: 输入已tokenize，长度为 {len(model_input[0])}")

        with torch.no_grad():
            print(f"步骤3: 开始生成回答...")
            model_output = generate(model_input, model, model_name, do_sample, output_scores, max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer)
            print(f"步骤4: 回答生成完毕，输出序列长度为 {len(model_output['sequences'][0])}")

        print(f"步骤5: 解码生成的token...")
        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
        print(f"步骤5完成: 解码后的回答长度为 {len(answer)} 字符")
        
        if output_scores:
            print("步骤6: 处理输出分数...")
            scores = torch.concatenate(model_output['scores']).cpu()  # shape = (new_tokens, len(vocab))
            all_scores.append(scores)
            output_ids = model_output['sequences'][0][len(model_input[0]):].cpu()
            all_output_ids.append(output_ids)
            print(f"步骤6完成: 分数处理完毕，形状为 {scores.shape}")

        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())
        print(f"步骤7: 已添加到结果列表")

        if verbose:
            if counter % 100 == 0:
                print(f"Counter: {counter}")
                print(f"Prompt: {prompt}")
                print(f"Answer: {answer}")
        counter += 1
        
        if counter <= 2 or counter % 50 == 0:  # 只打印前两个和每50个的详细信息
            print(f"示例输出 #{counter}:")
            print(f"问题: {prompt[:100]}...")
            print(f"回答: {answer[:100]}...")
            print("-" * 40)
    
    print(f"所有 {len(data)} 条数据处理完毕!")
    return all_textual_answers, all_input_output_ids, all_scores, all_output_ids


def init_wandb(args):
    cfg = vars(args)
    cfg['dataset'] = args.dataset
    wandb.init(
        project="generate_answers",
        config=cfg
    )


def load_data_imdb(split):
    dataset = load_dataset("imdb")


    indices = np.arange(0, len(dataset[split]))
    np.random.shuffle(indices)

    reviews = dataset[split][indices[:10000]]['text']
    labels = dataset[split][indices[:10000]]['label']
    return reviews, labels

def imdb_preprocess(model_name, reviews, labels):

    prompts = []
    labels_to_name = ['negative', 'positive']

    review1 = None
    label1 = None

    if 'phi' in model_name.lower():
        for review, label in zip(reviews, labels):
            prompt = f"""
            Review: I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst kind are the ones that are unwatchable but you are suppose to like them because they are supposed to be good for you! The sex sequences, so shocking in its day, couldn't even arouse a rabbit. The so called controversial politics is strictly high school sophomore amateur night Marxism. The film is self-consciously arty in the worst sense of the term. The photography is in a harsh grainy black and white. Some scenes are out of focus or taken from the wrong angle. Even the sound is bad! And some people call this art?<br /><br />
            Label: negative
            Review: Zentropa is the most original movie I've seen in years. If you like unique thrillers that are influenced by film noir, then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days. Von Trier's follow-ups like Breaking the Waves have gotten more acclaim, but this is really his best work. It is flashy without being distracting and offers the perfect combination of suspense and dark humor. It's too bad he decided handheld cameras were the wave of the future. It's hard to say who talked him away from the style he exhibits here, but it's everyone's loss that he went into his heavily theoretical dogma direction instead.
            Label: positive
            Review: {review}
            Label:"""
            prompts.append(prompt)
    else:
        for review, label in zip(reviews, labels):
            if review1 is None:
                review1 = review
                label1 = labels_to_name[label]
            answer_first_prompt = "Start with either 'positive' or 'negative' first, then elaborate."
            example_prompt = ''
            if 'llama-3' not in model_name.lower():
                answer_first_prompt = ''
                example_prompt = f"""
                Example:
                Review: {review1}
                Label: {label1}
                Review: {review}
                Label: """


            prompt = f"""Classify the following movie reviews as either "positive" or "negative". {answer_first_prompt}
            {example_prompt}
            Review: {review}
            Label:"""
            prompts.append(prompt)

    return prompts

def triviqa_preprocess(model_name, all_questions, labels):
    prompts = []
    if 'instruct' in model_name.lower():
        prompts = all_questions
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
        A:''')
    return prompts

def nq_preprocess(model_name, all_questions, labels, with_context, context):
    prompts = []
    if with_context:
        if 'instruct' in model_name.lower():
            for q, context in zip(all_questions, context):
                prompts.append(f'{context}\n{q}')
        else:
            for q, context in zip(all_questions, context):
                prompts.append(f'''Context:{context}
                Q: {q}
A:''')
    else:
        if 'instruct' in model_name.lower():
            prompts = [f'{q}' for q in all_questions]
        else:
            for q in all_questions:
                prompts.append(f'''Q: {q}
                A:''')
    print('Prompt:', prompts[-1])
    return prompts
def triviaqa_postprocess(model_name, raw_answers):
    model_answers = []
    if 'instruct' in model_name.lower():
        model_answers = raw_answers
    else:
        for ans in raw_answers:
            model_answer = ans.strip().split('\n')[0]
            model_answers.append(model_answer)
    return raw_answers, model_answers


def load_winobias(dev_or_test):
    data = pd.read_csv(f'../data/winobias_{dev_or_test}.csv')
    return (data['sentence'], data['q'], data['q_instruct']), data['answer'], data['incorrect_answer'], data['stereotype'], data['type']

def winobias_preprocess(model_name, all_questions, labels):
    sentences, q, q_instruct = all_questions
    if 'instruct' in model_name.lower():
        prompts = [x + ' ' + y for x, y in zip(sentences, q_instruct)]
    else:
        prompts = [x + ' ' + y for x, y in zip(sentences, q)]

    return prompts

def load_hotpotqa(split, with_context):

    dataset = load_dataset("hotpot_qa", 'distractor', trust_remote_code=True)
    subset_indices = np.random.randint(0, len(dataset[split]), 10000)
    all_questions = [dataset[split][int(x)]['question'] for x in subset_indices]
    labels = [dataset[split][int(x)]['answer'] for x in subset_indices]
    if with_context:
        all_questions = []

        for idx in subset_indices:
            prompt = ""
            for evidence in dataset[split][int(idx)]['context']['sentences']:
                for sentence in evidence:
                    prompt += sentence + '\n'
            prompt += dataset[split][int(idx)]['question']
            all_questions.append(prompt)

    return all_questions, labels



def winogrande_preprocess(model_name, all_questions, labels):
    if 'instruct' not in model_name.lower():
        new_questions = []
        q1 = None
        label1 = None
        for q, label in zip(all_questions,labels):
            if q1 is None:
                q1 = q.split("Who does the blank refer to in the sentence?")[0].split("What does the blank refer to in the sentence?")[0]
                label1 = label
            q_ = q.split("Who does the blank refer to in the sentence?")[0].split("What does the blank refer to in the sentence?")[0]
            q_with_ex = \
                f"""{q1} The blank refers to: {label1}
{q_} The blank refers to:"""
            new_questions.append(q_with_ex)

        return new_questions
    return all_questions

def load_data(dataset_name):
    max_new_tokens = 100
    n_data = 100
    max_length = 600  # 添加最大长度限制
    context, origin, stereotype, type_, wrong_labels = None, None, None, None, None
    
    def filter_by_length(questions, labels, length_limit=max_length, required_count=n_data):
        filtered_questions = []
        filtered_labels = []
        for q, l in zip(questions, labels):
            if len(q) <= length_limit:
                filtered_questions.append(q)
                filtered_labels.append(l)
                if len(filtered_questions) >= required_count:
                    break
        return filtered_questions, filtered_labels
    
    if dataset_name == 'triviaqa':
        all_questions, labels = load_data_triviaqa(False)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'triviaqa_test':
        all_questions, labels = load_data_triviaqa(True)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'imdb':
        all_questions, labels = load_data_imdb('train')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = imdb_preprocess
    elif dataset_name == 'imdb_test':
        all_questions, labels = load_data_imdb('test')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = imdb_preprocess
    elif dataset_name == 'winobias':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('dev')
        # 对第一个问题组件进行长度过滤,其他数据相应调整
        filtered_indices = [i for i, q in enumerate(all_questions[0]) if len(q) <= max_length][:n_data]
        all_questions = tuple(q[filtered_indices] for q in all_questions)
        labels = labels[filtered_indices]
        wrong_labels = wrong_labels[filtered_indices]
        stereotype = stereotype[filtered_indices]
        type_ = type_[filtered_indices]
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'winobias_test':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('test')
        # 对第一个问题组件进行长度过滤,其他数据相应调整
        filtered_indices = [i for i, q in enumerate(all_questions[0]) if len(q) <= max_length][:n_data]
        all_questions = tuple(q[filtered_indices] for q in all_questions)
        labels = labels[filtered_indices]
        wrong_labels = wrong_labels[filtered_indices]
        stereotype = stereotype[filtered_indices]
        type_ = type_[filtered_indices]
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'hotpotqa':
        all_questions, labels = load_hotpotqa('train', with_context=False)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_test':
        all_questions, labels = load_hotpotqa('validation', with_context=False)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_with_context':
        all_questions, labels = load_hotpotqa('train', with_context=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_with_context_test':
        all_questions, labels = load_hotpotqa('validation', with_context=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'math':
        all_questions, labels = load_data_math(test=False)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = math_preprocess
        max_new_tokens = 200
    elif dataset_name == 'math_test':
        all_questions, labels = load_data_math(test=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = math_preprocess
        max_new_tokens = 200
    elif dataset_name == 'movies':
        all_questions, labels = load_data_movies(test=False)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'movies_test':
        all_questions, labels = load_data_movies(test=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'mnli':
        all_questions, labels, origin = load_data_mnli('train')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'mnli_test':
        all_questions, labels, origin = load_data_mnli('test')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'natural_questions':
        all_questions, labels, context = load_data_nq('train')
        all_questions, labels = filter_by_length(all_questions, labels)
        if context is not None:
            context = context[:n_data]
        preprocess_fn = nq_preprocess
    elif dataset_name == 'natural_questions_test':
        all_questions, labels, context = load_data_nq('test')
        all_questions, labels = filter_by_length(all_questions, labels)
        if context is not None:
            context = context[:n_data]
        preprocess_fn = nq_preprocess
    elif dataset_name == 'natural_questions_with_context':
        all_questions, labels, context = load_data_nq('train', with_context=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        if context is not None:
            context = context[:n_data]
        preprocess_fn = nq_preprocess
    elif dataset_name == 'natural_questions_with_context_test':
        all_questions, labels, context = load_data_nq('test', with_context=True)
        all_questions, labels = filter_by_length(all_questions, labels)
        if context is not None:
            context = context[:n_data]
        preprocess_fn = nq_preprocess
    elif dataset_name == 'winogrande':
        all_questions, labels, wrong_labels = load_data_winogrande('train')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = winogrande_preprocess
    elif dataset_name == 'winogrande_test':
        all_questions, labels, wrong_labels = load_data_winogrande('test')
        all_questions, labels = filter_by_length(all_questions, labels)
        preprocess_fn = winogrande_preprocess
    else:
        raise TypeError("data type is not supported")
    
    print(f"已限制{dataset_name}数据集为前{n_data}个长度小于{max_length}的样本")
    return all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels

def main():
    args = parse_args()
    init_wandb(args)
    set_seed(0)
    dataset_size = args.n_samples

    # 加载模型和tokenizer，指定模型存储目录
    print(f"准备加载模型: {args.model}")
    print(f"模型下载/加载目录: {args.model_dir}")
    model, tokenizer = load_model_and_validate_gpu(args.model, local_models_dir=args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args.dataset)

    if not os.path.exists('../output'):
        os.makedirs('../output')
        print("创建输出目录: ../output")

    file_path_output_ids = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt"
    file_path_scores = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-scores-{args.dataset}.pt"
    file_path_answers = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    print(f"输出文件路径设置完成:")
    print(f"- 输出ID文件: {file_path_output_ids}")
    print(f"- 分数文件: {file_path_scores}")
    print(f"- 回答文件: {file_path_answers}")

    if dataset_size:
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        print(f"数据集大小已限制为: {dataset_size}")
        if 'mnli' in args.dataset:
            origin = origin[:dataset_size]
        if 'winogrande' in args.dataset:
            wrong_labels = wrong_labels[:dataset_size]

    output_csv = {}
    if preprocess_fn:
        print(f"开始预处理数据...")
        if 'winobias' in args.dataset:
            output_csv['raw_question'] = all_questions[0]
        else:
            output_csv['raw_question'] = all_questions
        if 'natural_questions' in args.dataset:
            with_context = True if 'with_context' in args.dataset else False
            print('preprocessing nq')
            all_questions = preprocess_fn(args.model, all_questions, labels, with_context, context)
        else:
            all_questions = preprocess_fn(args.model, all_questions, labels)
        print(f"预处理完成，处理后的问题数量: {len(all_questions)}")

    print("\n" + "="*50)
    print("开始生成模型回答...")
    print("="*50)
    model_answers, input_output_ids, all_scores, all_output_ids = generate_model_answers(all_questions, model,
                                                                                         tokenizer, device, args.model,
                                                                                         output_scores=True, max_new_tokens=max_new_tokens,
                                                                                         stop_token_id=stop_token_id,
                                                                                         verbose=args.verbose)
    print("="*50)
    print("模型回答生成完成!")
    print("="*50 + "\n")

    print("计算正确率...")
    res = compute_correctness(all_questions, args.dataset, args.model, labels, model, model_answers, tokenizer, wrong_labels)
    correctness = res['correctness']

    acc = np.mean(correctness)
    wandb.summary[f'acc'] = acc
    print(f"准确率: {acc:.4f}")

    output_csv['question'] = all_questions
    output_csv['model_answer'] = model_answers
    output_csv['correct_answer'] = labels
    output_csv['automatic_correctness'] = correctness

    if 'exact_answer' in res:
        output_csv['exact_answer'] = res['exact_answer']
        # 创建与其他列表等长的值为1的列表
        output_csv['valid_exact_answer'] = [1] * len(res['exact_answer'])
        # output_csv['valid_exact_answer'] = 1
    if 'incorrect_answer' in res:
        output_csv['incorrect_answer'] = res['incorrect_answer']
    if 'winobias' in args.dataset:
        output_csv['stereotype'] = stereotype
        output_csv['type'] = type_
    if 'nli' in args.dataset:
        output_csv['origin'] = origin

    # 确保所有值都是列表或可迭代对象
    for key, value in list(output_csv.items()):
        if not hasattr(value, '__len__') or isinstance(value, (str, dict)):
            print(f"警告: 字段 '{key}' 不是列表类型 (类型: {type(value)})")
            if isinstance(value, (int, float, bool, str)):
                # 将单一值转换为列表
                reference_length = len(next((v for v in output_csv.values() if hasattr(v, '__len__') and not isinstance(v, (str, dict))), []))
                print(f"将单一值转换为长度为 {reference_length} 的列表")
                output_csv[key] = [value] * reference_length

    print("Saving answers to ", file_path_answers)

    lengths = {key: len(value) for key, value in output_csv.items()}
    print("输出字段长度检查:", lengths)
    if len(set(lengths.values())) > 1:
        print("警告: 输出字段长度不一致!")
        # 找出最小长度并截断所有字段
        min_length = min(lengths.values())
        for key in output_csv:
            if len(output_csv[key]) > min_length:
                print(f"截断字段 {key}: {len(output_csv[key])} -> {min_length}")
                output_csv[key] = output_csv[key][:min_length]

    pd.DataFrame.from_dict(output_csv).to_csv(file_path_answers)

    print("Saving input output ids to ", file_path_output_ids)
    torch.save(input_output_ids, file_path_output_ids)

    print("Saving input output ids to ", file_path_scores)
    torch.save({"all_scores": all_scores,
                "all_output_ids": all_output_ids}, file_path_scores)

if __name__ == "__main__":
    main()