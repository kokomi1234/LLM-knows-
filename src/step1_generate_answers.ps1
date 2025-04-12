# step1_generate_answers.ps1
# 运行前请先加载配置: . .\config.ps1
# 验证配置已正确加载


Write-Host "数据集: $($DATASETS -join ', ')" -ForegroundColor Cyan
Write-Host "模型: $MODEL" -ForegroundColor Cyan
Write-Host "样本数: $ALL_SAMPLES" -ForegroundColor Cyan

if ($null -eq $DATASETS -or $DATASETS.Count -eq 0) {
    Write-Host "错误: 数据集列表为空！请确保已正确运行 '. .\config.ps1'" -ForegroundColor Red
    exit
}


foreach ($DATASET in $DATASETS) {
    Write-Host "===== 处理数据集: $DATASET - 生成答案 =====" -ForegroundColor Green
    
    # GPU内存清理
    Clear-GPUMemory
    
    # 生成模型答案
    python ./generate_model_answers.py --model $MODEL --dataset $DATASET --n_samples $ALL_SAMPLES --model_dir $MODEL_DIR
    
    # 复制文件到期望位置以便后续步骤使用
    if (Test-Path "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-answers-$DATASET.csv") {
        Copy-Item "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-answers-$DATASET.csv" -Destination "$OUTPUT_DIR/" -Force
        Write-Host "已复制答案文件到: $OUTPUT_DIR" -ForegroundColor Blue
    }
    
    # 对需要的数据集提取精确答案
    if ($EXTRACT_DATASETS -contains $DATASET) {
        Write-Host "提取精确答案: $DATASET" -ForegroundColor Yellow
        python extract_exact_answer.py --model $MODEL --dataset $DATASET
        # 添加这段：提取精确答案后复制更新的文件
        if (Test-Path "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-answers-$DATASET.csv") {
            Copy-Item "../output/$($MODEL_FRIENDLY_NAMES[$MODEL])-answers-$DATASET.csv" -Destination "$OUTPUT_DIR/" -Force
            Write-Host "已复制带精确答案的文件到: $OUTPUT_DIR" -ForegroundColor Blue
        } else {
            Write-Host "警告: 提取精确答案后的文件未找到" -ForegroundColor Yellow
        }
    }
}

Write-Host "步骤1完成: 所有数据集的模型答案已生成" -ForegroundColor Green