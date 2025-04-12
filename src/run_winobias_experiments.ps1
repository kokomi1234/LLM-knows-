# PowerShell script for running Winobias experiments
# Setup variables
$MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
$DATASET = "imdb"
$DATASET_TEST = "imdb_test"
# Fix seeds parameter, using array instead of space-separated string
$SEEDS_ARRAY = @(0)  # Array format for internal script use
$SEEDS = "0"             # Keep original variable for reference
$LAYER = 15  # Adjust best layer based on model
$TOKEN = "exact_answer_last_token"
$PROBE_AT = "mlp"
$N_RESAMPLES = 1  # Use 10 resamples to save time instead of 30 as in the paper

# Set PowerShell to use UTF-8 encoding to prevent character encoding issues
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "===== Starting Winobias Experiments =====" -ForegroundColor Green
Write-Host "Using model: $MODEL" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET" -ForegroundColor Cyan

# #Step 1: Generate model answers
Write-Host "===== Step 1: Generate Model Answers =====" -ForegroundColor Yellow
python generate_model_answers.py --model $MODEL --dataset $DATASET
python generate_model_answers.py --model $MODEL --dataset $DATASET_TEST

# Note: Winobias dataset doesn't need to extract exact answers as they can be extracted during generation

# Step 2: Probe all layers and tokens to create heatmap (Section 2 of paper)
Write-Host "===== Step 2: Probe All Layers and Tokens =====" -ForegroundColor Yellow
python probe_all_layers_and_tokens.py --model $MODEL --probe_at mlp_last_layer_only_input --seed 0 --n_samples 1000 --dataset $DATASET

# Step 3: Probe specific layer and token (Section 3 of paper)
Write-Host "===== Step 3: Probe Specific Layer and Token =====" -ForegroundColor Yellow
# Fix seeds parameter format, ensure each seed is passed as a separate argument
python probe.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --save_clf --dataset $DATASET --layer $LAYER --token $TOKEN

# Step 4: Generalization Test (Section 4 of paper) - Optional
# If you want to test generalization from Winobias to Winogrande, uncomment below
Write-Host "===== Step 4: Generalization Test =====" -ForegroundColor Yellow
python probe.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --save_clf --dataset $DATASET --test_dataset winogrande --layer $LAYER --token $TOKEN

# Step 5: Resampling (Prepare for subsequent experiments)
Write-Host "===== Step 5: Resampling =====" -ForegroundColor Yellow


# Ensure we're in the correct directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Fix job execution method, use absolute paths and correctly pass variables
# Training set
Write-Host "Processing training set resampling..." -ForegroundColor Cyan
foreach ($seed in $SEEDS_ARRAY) {
    $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
    $jobName = "resample_train_$tagIndex"
    # 修复变量引用，使用${varname}语法
    Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET}" -ForegroundColor Cyan
    
    # Execute Python script directly instead of using Start-Job
    $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET' --n_resamples 2 --tag $tagIndex"
    Write-Host "Executing command: $command" -ForegroundColor Cyan
    Invoke-Expression $command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Test set
Write-Host "Processing test set resampling..." -ForegroundColor Cyan
foreach ($seed in $SEEDS_ARRAY) {
    $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
    $jobName = "resample_test_$tagIndex"
    # 修复变量引用，使用${varname}语法
    Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET_TEST}" -ForegroundColor Cyan
    
    # Execute Python script directly instead of using Start-Job
    $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET_TEST' --n_resamples 2 --tag $tagIndex"
    Write-Host "Executing command: $command" -ForegroundColor Cyan
    Invoke-Expression $command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Alternative approach for parallel execution with correct variable passing (commented out)
<#
# Training set - parallel execution version
$trainJobs = @()
foreach ($seed in $SEEDS_ARRAY) {
    $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
    $scriptBlock = {
        param($model, $seed, $dataset, $tag, $scriptDir)
        Set-Location $scriptDir
        python resampling.py --model $model --seed $seed --dataset $dataset --n_resamples 2 --tag $tag
    }
    $trainJobs += Start-Job -ScriptBlock $scriptBlock -ArgumentList $MODEL, $seed, $DATASET, $tagIndex, $scriptDir
}

# Wait for all training set jobs to complete
Wait-Job -Job $trainJobs
foreach ($job in $trainJobs) {
    $result = Receive-Job -Job $job
    Write-Host "Job $($job.Id) output:" -ForegroundColor Gray
    $result
    if ($job.State -eq "Failed") {
        Write-Host "Job failed: $($job.Id)" -ForegroundColor Red
    }
}
Remove-Job -Job $trainJobs
#>

# Check if merge_resampling_files.py exists, if not create a simple version
$mergeScriptPath = "$scriptDir\resampling_merge_runs.py"
if (-not (Test-Path $mergeScriptPath)) {
    Write-Host "resampling_merge_runs.py not found at ${mergeScriptPath}, checking alternative path..." -ForegroundColor Yellow
    $mergeScriptPath = "e:\courses\3down\Hallucination_detection\2410.20707v3\LLMsKnow\src\resampling_merge_runs.py"
    if (-not (Test-Path $mergeScriptPath)) {
        Write-Host "resampling_merge_runs.py not found, cannot merge resampling results!" -ForegroundColor Red
        Write-Host "Please ensure the script exists before continuing to step 6." -ForegroundColor Red
        exit 1
    }
}

# Create output/resampling directory if it doesn't exist
$resamplingDir = "../output/resampling"
if (-not (Test-Path $resamplingDir)) {
    Write-Host "Creating directory: $resamplingDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $resamplingDir -Force | Out-Null
}

# Verify we have resampling files before trying to merge
$resamplingFiles = Get-ChildItem -Path $resamplingDir -Filter "*${DATASET}*" | Measure-Object
if ($resamplingFiles.Count -eq 0) {
    Write-Host "Warning: No resampling files found in $resamplingDir for dataset $DATASET." -ForegroundColor Yellow
    Write-Host "Make sure the resampling step completed successfully." -ForegroundColor Yellow
}

# Merge resampling results
Write-Host "Merging resampling results..." -ForegroundColor Cyan
# 确保目录正确且命令正确执行
python $mergeScriptPath --model $MODEL --dataset $DATASET --n_resamples $N_RESAMPLES
python $mergeScriptPath --model $MODEL --dataset $DATASET_TEST --n_resamples $N_RESAMPLES

# 明确显示执行的命令
Write-Host "Executed: python $mergeScriptPath --model $MODEL --dataset $DATASET --n_resamples $N_RESAMPLES" -ForegroundColor Yellow
Write-Host "Executed: python $mergeScriptPath --model $MODEL --dataset $DATASET_TEST --n_resamples $N_RESAMPLES" -ForegroundColor Yellow

# Verify merged files exist before continuing
# 使用PowerShell的插值字符串来确保变量正确展开
$trainFile = "../output/resampling/mistral-7b-instruct_${DATASET}_${N_RESAMPLES}_textual_answers.pt"
$testFile = "../output/resampling/mistral-7b-instruct_${DATASET_TEST}_${N_RESAMPLES}_textual_answers.pt"

Write-Host "Checking for merged files:" -ForegroundColor Cyan
Write-Host "  - Training file: $trainFile" -ForegroundColor Cyan  
Write-Host "  - Testing file: $testFile" -ForegroundColor Cyan

if (-not (Test-Path $trainFile)) {
    Write-Host "Warning: Training merged file not found: $trainFile" -ForegroundColor Red
    # 列出可能的文件以帮助诊断
    Write-Host "Available files in ../output/resampling/ directory:" -ForegroundColor Yellow
    Get-ChildItem -Path "../output/resampling/" -Filter "*${DATASET}*" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
}
if (-not (Test-Path $testFile)) {
    Write-Host "Warning: Testing merged file not found: $testFile" -ForegroundColor Red
    Write-Host "Step 6 may fail without this file!" -ForegroundColor Red
    # 列出可能的文件以帮助诊断
    Write-Host "Available files in ../output/resampling/ directory:" -ForegroundColor Yellow
    Get-ChildItem -Path "../output/resampling/" -Filter "*${DATASET_TEST}*" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
}

# 在Step 5前添加检查，确保文件存在
$trainFile = "../output/resampling/mistral-7b-instruct_${DATASET}_${N_RESAMPLES}_textual_answers.pt"
$testFile = "../output/resampling/mistral-7b-instruct_${DATASET_TEST}_${N_RESAMPLES}_textual_answers.pt"

# 如果文件已经存在，可以跳过Step 5和合并步骤
$skipResampling = (Test-Path $trainFile) -and (Test-Path $testFile)

if ($skipResampling) {
    Write-Host "Required resampling files already exist. Skipping Step 5 and merging." -ForegroundColor Green
}
else {
    # Step 5: Resampling (Prepare for subsequent experiments)
    Write-Host "===== Step 5: Resampling =====" -ForegroundColor Yellow
    
    # Ensure we're in the correct directory
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $scriptDir

    # Fix job execution method, use absolute paths and correctly pass variables
    # Training set
    Write-Host "Processing training set resampling..." -ForegroundColor Cyan
    foreach ($seed in $SEEDS_ARRAY) {
        $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
        $jobName = "resample_train_$tagIndex"
        # 修复变量引用，使用${varname}语法
        Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET}" -ForegroundColor Cyan
        
        # Execute Python script directly instead of using Start-Job
        $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET' --n_resamples 2 --tag $tagIndex"
        Write-Host "Executing command: $command" -ForegroundColor Cyan
        Invoke-Expression $command
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }

    # Test set
    Write-Host "Processing test set resampling..." -ForegroundColor Cyan
    foreach ($seed in $SEEDS_ARRAY) {
        $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
        $jobName = "resample_test_$tagIndex"
        # 修复变量引用，使用${varname}语法
        Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET_TEST}" -ForegroundColor Cyan
        
        # Execute Python script directly instead of using Start-Job
        $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET_TEST' --n_resamples 2 --tag $tagIndex"
        Write-Host "Executing command: $command" -ForegroundColor Cyan
        Invoke-Expression $command
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }

    # Check if merge_resampling_files.py exists, if not create a simple version
    $mergeScriptPath = "$scriptDir\resampling_merge_runs.py"
    if (-not (Test-Path $mergeScriptPath)) {
        Write-Host "resampling_merge_runs.py not found at ${mergeScriptPath}, checking alternative path..." -ForegroundColor Yellow
        $mergeScriptPath = "e:\courses\3down\Hallucination_detection\2410.20707v3\LLMsKnow\src\resampling_merge_runs.py"
        if (-not (Test-Path $mergeScriptPath)) {
            Write-Host "resampling_merge_runs.py not found, cannot merge resampling results!" -ForegroundColor Red
            Write-Host "Please ensure the script exists before continuing to step 6." -ForegroundColor Red
            exit 1
        }
    }

    # Create output/resampling directory if it doesn't exist
    $resamplingDir = "../output/resampling"
    if (-not (Test-Path $resamplingDir)) {
        Write-Host "Creating directory: $resamplingDir" -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $resamplingDir -Force | Out-Null
    }

    # Verify we have resampling files before trying to merge
    $resamplingFiles = Get-ChildItem -Path $resamplingDir -Filter "*${DATASET}*" | Measure-Object
    if ($resamplingFiles.Count -eq 0) {
        Write-Host "Warning: No resampling files found in $resamplingDir for dataset $DATASET." -ForegroundColor Yellow
        Write-Host "Make sure the resampling step completed successfully." -ForegroundColor Yellow
    }

    # Merge resampling results
    Write-Host "Merging resampling results..." -ForegroundColor Cyan
    # 确保目录正确且命令正确执行
    python $mergeScriptPath --model $MODEL --dataset $DATASET --n_resamples $N_RESAMPLES
    python $mergeScriptPath --model $MODEL --dataset $DATASET_TEST --n_resamples $N_RESAMPLES

    # 明确显示执行的命令
    Write-Host "Executed: python $mergeScriptPath --model $MODEL --dataset $DATASET --n_resamples $N_RESAMPLES" -ForegroundColor Yellow
    Write-Host "Executed: python $mergeScriptPath --model $MODEL --dataset $DATASET_TEST --n_resamples $N_RESAMPLES" -ForegroundColor Yellow

    # Verify merged files exist before continuing
    # 使用PowerShell的插值字符串来确保变量正确展开
    $trainFile = "../output/resampling/mistral-7b-instruct_${DATASET}_${N_RESAMPLES}_textual_answers.pt"
    $testFile = "../output/resampling/mistral-7b-instruct_${DATASET_TEST}_${N_RESAMPLES}_textual_answers.pt"

    Write-Host "Checking for merged files:" -ForegroundColor Cyan
    Write-Host "  - Training file: $trainFile" -ForegroundColor Cyan  
    Write-Host "  - Testing file: $testFile" -ForegroundColor Cyan

    if (-not (Test-Path $trainFile)) {
        Write-Host "Warning: Training merged file not found: $trainFile" -ForegroundColor Red
        # 列出可能的文件以帮助诊断
        Write-Host "Available files in ../output/resampling/ directory:" -ForegroundColor Yellow
        Get-ChildItem -Path "../output/resampling/" -Filter "*${DATASET}*" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
    }
    if (-not (Test-Path $testFile)) {
        Write-Host "Warning: Testing merged file not found: $testFile" -ForegroundColor Red
        Write-Host "Step 6 may fail without this file!" -ForegroundColor Red
        # 列出可能的文件以帮助诊断
        Write-Host "Available files in ../output/resampling/ directory:" -ForegroundColor Yellow
        Get-ChildItem -Path "../output/resampling/" -Filter "*${DATASET_TEST}*" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
    }
}

# Step 6: Error Type Detection (Section 5 of paper)
Write-Host "===== Step 6: Error Type Detection =====" -ForegroundColor Yellow
# 首先确保文件存在
if (-not (Test-Path $trainFile) -or -not (Test-Path $testFile)) {
    Write-Host "Warning: Required resampling files not found! Step 6 may fail." -ForegroundColor Red
    Write-Host "Missing files:" -ForegroundColor Yellow
    if (-not (Test-Path $trainFile)) {
        Write-Host "  - $trainFile" -ForegroundColor Yellow
    }
    if (-not (Test-Path $testFile)) {
        Write-Host "  - $testFile" -ForegroundColor Yellow
    }
    
    # 显示可用的文件
    Write-Host "Available files in resampling directory:" -ForegroundColor Cyan
    Get-ChildItem -Path "../output/resampling/" | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
    
    # 询问是否继续
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        Write-Host "Aborting..." -ForegroundColor Red
        exit 1
    }
}

# Fix seeds parameter format
python probe_type_of_error.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --n_resamples $N_RESAMPLES --token $TOKEN --dataset $DATASET_TEST --layer $LAYER --merge_types

# Step 7: Answer Selection Experiment (Section 6 of paper)
Write-Host "===== Step 7: Answer Selection Experiment =====" -ForegroundColor Yellow
# Fix seeds parameter format
python probe_choose_answer.py --model $MODEL --probe_at $PROBE_AT --layer $LAYER --token $TOKEN --dataset $DATASET_TEST --n_resamples $N_RESAMPLES --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4]

# Step 8: Other Baseline Experiments
Write-Host "===== Step 8: Other Baseline Experiments =====" -ForegroundColor Yellow
# Check if detection_by_logprob module required files exist
try {
    # logprob detection, fix seeds parameter format
    python logprob_detection.py --model $MODEL --dataset $DATASET --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --use_exact_answer
    
    # p_true detection, fix seeds parameter format
#     python p_true_detection.py --model $MODEL --dataset $DATASET --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --use_exact_answer
}
catch {
    Write-Host "Error executing baseline experiments: $_" -ForegroundColor Red
    Write-Host "Necessary modules might be missing, check if detection_by_logprob.py exists" -ForegroundColor Yellow
}

Write-Host "===== Experiments Completed! =====" -ForegroundColor Green