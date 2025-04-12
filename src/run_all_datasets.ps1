# run_all_datasets.ps1

# 指定模型名称
$MODEL="mistralai/Mistral-7B-Instruct-v0.2"
$MODEL_DIR="../model"
$SAMPLES=100
$BASE_OUTPUT_DIR="../output/new"

# 添加模型友好名称映射
$MODEL_FRIENDLY_NAMES = @{
    "mistralai/Mistral-7B-Instruct-v0.2" = "mistral-7b-instruct"
    "mistralai/Mistral-7B-v0.3" = "mistral-7b"
    "meta-llama/Meta-Llama-3-8B" = "llama-3-8b"
    "meta-llama/Meta-Llama-3-8B-Instruct" = "llama-3-8b-instruct"
}

# 创建主输出目录
if (!(Test-Path -Path $BASE_OUTPUT_DIR)) {
    Write-Host "Creating output directory: $BASE_OUTPUT_DIR" -ForegroundColor Blue
    New-Item -ItemType Directory -Path $BASE_OUTPUT_DIR -Force | Out-Null
}
# 创建数据集列表
$DATASETS = @(
    "natural_questions_with_context", "natural_questions_with_context_test"
)

# 需要提取精确答案的数据集
$EXTRACT_DATASETS = @(
    "natural_questions_with_context", "natural_questions_with_context_test"
)

# 为每个数据集执行生成和提取
foreach ($DATASET in $DATASETS) {
    Write-Host "========================================================" 
    Write-Host "Processing dataset: $DATASET" -ForegroundColor Green
    
    # 创建数据集专用目录
    $DATASET_DIR = "$BASE_OUTPUT_DIR/$DATASET"
    if (!(Test-Path -Path $DATASET_DIR)) {
        Write-Host "Creating dataset directory: $DATASET_DIR" -ForegroundColor Blue
        New-Item -ItemType Directory -Path $DATASET_DIR -Force | Out-Null
    }
    
    Write-Host "Step 1: Generating model answers" -ForegroundColor Yellow
    
    # 生成模型答案 - 移除不支持的参数
    python generate_model_answers.py --model $MODEL --dataset $DATASET --n_samples $SAMPLES --model_dir $MODEL_DIR
    
    # 在生成完成后，将文件移动到数据集专用目录
    $SOURCE_FILE_ANSWERS = "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-answers-$DATASET.csv"
    $SOURCE_FILE_IDS = "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-input_output_ids-$DATASET.pt"
    $SOURCE_FILE_SCORES = "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-scores-$DATASET.pt"
    
    if (Test-Path $SOURCE_FILE_ANSWERS) {
        Copy-Item $SOURCE_FILE_ANSWERS -Destination "$DATASET_DIR/" -Force
    }
    if (Test-Path $SOURCE_FILE_IDS) {
        Copy-Item $SOURCE_FILE_IDS -Destination "$DATASET_DIR/" -Force
    }
    if (Test-Path $SOURCE_FILE_SCORES) {
        Copy-Item $SOURCE_FILE_SCORES -Destination "$DATASET_DIR/" -Force
    }
    
    # 检查是否需要提取精确答案
    if ($EXTRACT_DATASETS -contains $DATASET) {
        Write-Host "Step 2: Extracting exact answers" -ForegroundColor Yellow
        # 不使用不支持的参数
        python extract_exact_answer.py --model $MODEL --dataset $DATASET
        
        # 提取完成后再次复制文件
        if (Test-Path $SOURCE_FILE_ANSWERS) {
            Copy-Item $SOURCE_FILE_ANSWERS -Destination "$DATASET_DIR/" -Force
        }
    } 
    else {
        Write-Host "This dataset does not need exact answer extraction" -ForegroundColor Yellow
    }
    
    Write-Host "Completed processing dataset: $DATASET" -ForegroundColor Green
    Write-Host "========================================================" 
}

Write-Host "All datasets processing complete!" -ForegroundColor Green
Write-Host "Results saved in: $BASE_OUTPUT_DIR" -ForegroundColor Green