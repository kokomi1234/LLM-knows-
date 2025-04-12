# PowerShell script for running TriviaQA experiments
# Setup variables
$MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
$DATASET = "triviaqa"
$DATASET_TEST = "triviaqa_test"
# Fix seeds parameter, using array instead of space-separated string
$SEEDS_ARRAY = @(0)  # Array format for multiple seeds
$SEEDS = "0"             # Keep original variable for reference
$LAYER = 15  # Adjust best layer based on model or heatmap results
$TOKEN = "exact_answer_last_token"
$PROBE_AT = "mlp"
$N_RESAMPLES = 1  # Use 30 resamples as per paper (adjust to 1 or 10 for testing)

# Set PowerShell to use UTF-8 encoding to prevent character encoding issues
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "===== Starting TriviaQA Experiments =====" -ForegroundColor Green
Write-Host "Using model: $MODEL" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET" -ForegroundColor Cyan

# Step 1: Generate model answers
Write-Host "===== Step 1: Generate Model Answers =====" -ForegroundColor Yellow
python generate_model_answers.py --model $MODEL --dataset $DATASET
python generate_model_answers.py --model $MODEL --dataset $DATASET_TEST

# Step 1b: Extract exact answers (required for TriviaQA)
Write-Host "===== Step 1b: Extract Exact Answers =====" -ForegroundColor Yellow
python extract_exact_answer.py --model $MODEL --dataset $DATASET
python extract_exact_answer.py --model $MODEL --dataset $DATASET_TEST

# Step 2: Probe all layers and tokens to create heatmap (Section 2 of paper)
Write-Host "===== Step 2: Probe All Layers and Tokens =====" -ForegroundColor Yellow
python probe_all_layers_and_tokens.py --model $MODEL --probe_at mlp_last_layer_only_input --seed 0 --n_samples 1000 --dataset $DATASET

# Step 3: Probe specific layer and token (Section 3 of paper)
Write-Host "===== Step 3: Probe Specific Layer and Token =====" -ForegroundColor Yellow
python probe.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --save_clf --dataset $DATASET --layer $LAYER --token $TOKEN

# Step 4: Generalization Test (Section 4 of paper) - Optional
Write-Host "===== Step 4: Generalization Test =====" -ForegroundColor Yellow
# Test generalization from TriviaQA to HotpotQA (example)
python probe.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --save_clf --dataset $DATASET --test_dataset hotpotqa --layer $LAYER --token $TOKEN

# Step 5: Resampling (Prepare for subsequent experiments)
Write-Host "===== Step 5: Resampling =====" -ForegroundColor Yellow

# Ensure we're in the correct directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Training set resampling
Write-Host "Processing training set resampling..." -ForegroundColor Cyan
foreach ($seed in $SEEDS_ARRAY) {
    $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
    $jobName = "resample_train_$tagIndex"
    Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET}" -ForegroundColor Cyan
    
    $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET' --n_resamples $N_RESAMPLES --tag $tagIndex"
    Write-Host "Executing command: $command" -ForegroundColor Cyan
    Invoke-Expression $command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Test set resampling
Write-Host "Processing test set resampling..." -ForegroundColor Cyan
foreach ($seed in $SEEDS_ARRAY) {
    $tagIndex = [array]::IndexOf($SEEDS_ARRAY, $seed)
    $jobName = "resample_test_$tagIndex"
    Write-Host "Starting job ${jobName} - Using seed ${seed}, tag ${tagIndex}, dataset ${DATASET_TEST}" -ForegroundColor Cyan
    
    $command = "python $scriptDir\resampling.py --model '$MODEL' --seed $seed --dataset '$DATASET_TEST' --n_resamples $N_RESAMPLES --tag $tagIndex"
    Write-Host "Executing command: $command" -ForegroundColor Cyan
    Invoke-Expression $command
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Resampling job failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Merge resampling results
Write-Host "Merging resampling results..." -ForegroundColor Cyan
$mergeScriptPath = "$scriptDir\resampling_merge_runs.py"
if (-not (Test-Path $mergeScriptPath)) {
    Write-Host "resampling_merge_runs.py not found at ${mergeScriptPath}, checking alternative path..." -ForegroundColor Yellow
    $mergeScriptPath = "e:\courses\3down\Hallucination_detection\2410.20707v3\LLMsKnow\src\resampling_merge_runs.py"
    if (-not (Test-Path $mergeScriptPath)) {
        Write-Host "resampling_merge_runs.py not found, cannot merge resampling results!" -ForegroundColor Red
        exit 1
    }
}

python $mergeScriptPath --model $MODEL --dataset $DATASET --n_resamples $N_RESAMPLES
python $mergeScriptPath --model $MODEL --dataset $DATASET_TEST --n_resamples $N_RESAMPLES

# Step 5b: Extract exact answers from resampled data (required for TriviaQA)
Write-Host "===== Step 5b: Extract Exact Answers from Resampled Data =====" -ForegroundColor Yellow
python extract_exact_answer.py --dataset $DATASET --do_resampling $N_RESAMPLES --model $MODEL --extraction_model $MODEL
python extract_exact_answer.py --dataset $DATASET_TEST --do_resampling $N_RESAMPLES --model $MODEL --extraction_model $MODEL

# Verify merged files
$trainFile = "../output/resampling/mistral-7b-instruct_${DATASET}_${N_RESAMPLES}_textual_answers.pt"
$testFile = "../output/resampling/mistral-7b-instruct_${DATASET_TEST}_${N_RESAMPLES}_textual_answers.pt"
Write-Host "Checking for merged files:" -ForegroundColor Cyan
Write-Host "  - Training file: $trainFile" -ForegroundColor Cyan  
Write-Host "  - Testing file: $testFile" -ForegroundColor Cyan

if (-not (Test-Path $trainFile) -or -not (Test-Path $testFile)) {
    Write-Host "Warning: Required resampling files not found!" -ForegroundColor Red
    if (-not (Test-Path $trainFile)) { Write-Host "  - $trainFile" -ForegroundColor Yellow }
    if (-not (Test-Path $testFile)) { Write-Host "  - $testFile" -ForegroundColor Yellow }
}

# Step 6: Error Type Detection (Section 5 of paper)
Write-Host "===== Step 6: Error Type Detection =====" -ForegroundColor Yellow
python probe_type_of_error.py --model $MODEL --probe_at $PROBE_AT --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --n_samples all --n_resamples $N_RESAMPLES --token $TOKEN --dataset $DATASET_TEST --layer $LAYER --merge_types

# Step 7: Answer Selection Experiment (Section 6 of paper)
Write-Host "===== Step 7: Answer Selection Experiment =====" -ForegroundColor Yellow
python probe_choose_answer.py --model $MODEL --probe_at $PROBE_AT --layer $LAYER --token $TOKEN --dataset $DATASET_TEST --n_resamples $N_RESAMPLES --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4]

# Step 8: Other Baseline Experiments
Write-Host "===== Step 8: Other Baseline Experiments =====" -ForegroundColor Yellow
try {
    python logprob_detection.py --model $MODEL --dataset $DATASET --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --use_exact_answer
    # python p_true_detection.py --model $MODEL --dataset $DATASET --seeds $SEEDS_ARRAY[0] $SEEDS_ARRAY[1] $SEEDS_ARRAY[2] $SEEDS_ARRAY[3] $SEEDS_ARRAY[4] --use_exact_answer
}
catch {
    Write-Host "Error executing baseline experiments: $_" -ForegroundColor Red
}

Write-Host "===== Experiments Completed! =====" -ForegroundColor Green