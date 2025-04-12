# config.ps1 - 基础配置
# 在运行其他脚本前，先运行: . .\config.ps1

# 基本配置
$MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
$MODEL_DIR = "../model"
$OUTPUT_DIR = "../output/test_workflow"
$SMALL_SAMPLES = 100  # 用于内存密集型操作的样本数量
$ALL_SAMPLES = 100    # 其他操作的样本数量

# 模型名称映射
$MODEL_FRIENDLY_NAMES = @{
    "mistralai/Mistral-7B-Instruct-v0.2" = "mistral-7b-instruct"
    # "mistralai/Mistral-7B-v0.3" = "mistral-7b"
    # "meta-llama/Meta-Llama-3-8B" = "llama-3-8b"
    # "meta-llama/Meta-Llama-3-8B-Instruct" = "llama-3-8b-instruct"
}

# 数据集
$DATASETS = @(
    # TriviaQA
    "triviaqa", "triviaqa_test"
    # # Movies
    # "movies", "movies_test",
    # # HotpotQA - 普通版本
    # "hotpotqa", "hotpotqa_test",
    # # HotpotQA - 带上下文版本
    # "hotpotqa_with_context", "hotpotqa_with_context_test",
    # # Winobias
    # "winobias", "winobias_test",
    # # Winogrande
    # "winogrande", "winogrande_test",
    # # NLI (mnli) - 注意验证集名称应使用mnli_validation而非mnli_test
    # # "mnli",
    # "mnli_test", 
    # # IMDB
    # "imdb", "imdb_test",
    # # Math
    # "math", "math_test"
    # # Natural Questions - 带上下文
    # "natural_questions_with_context", "natural_questions_with_context_test"
)

# 需要提取精确答案的数据集
$EXTRACT_DATASETS = @(
    # 需要提取精确答案的数据集
    "triviaqa", "triviaqa_test",
    "movies", "movies_test",
    "hotpotqa", "hotpotqa_test",
    "hotpotqa_with_context", "hotpotqa_with_context_test",
    "math", "math_test"
    # "natural_questions_with_context", "natural_questions_with_context_test"
)

# 确保输出目录存在
if (!(Test-Path -Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
    Write-Host "创建输出目录: $OUTPUT_DIR" -ForegroundColor Blue
}

# GPU内存清理函数
function Clear-GPUMemory {
    python -c "import torch; torch.cuda.empty_cache(); print('GPU 内存已清理')"
}

Write-Host "配置已加载！请运行各步骤脚本..." -ForegroundColor Green