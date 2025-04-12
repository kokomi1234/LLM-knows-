# step5_resample_mnli.ps1
# 运行前请先加载配置: . .\config.ps1

# 只处理mnli_test
$DATASET = "mnli_test"  

Write-Host "===== 处理数据集: ${DATASET} - 重采样 =====" -ForegroundColor Green

# GPU内存清理
Clear-GPUMemory

# 执行重采样
python ./resampling.py --model $MODEL --seed 0 --dataset $DATASET --n_resamples 5

# 处理文件命名 (复制并删除tag)
$SOURCE = "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_textual_answers_0.pt"
$TARGET = "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_textual_answers.pt"

if (Test-Path $SOURCE) {
    Copy-Item -Path $SOURCE -Destination $TARGET -Force
    Write-Host "复制文件: $SOURCE -> $TARGET" -ForegroundColor Blue
    
    # 复制其他相关文件
    Copy-Item -Path "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_exact_answers_0.pt" -Destination "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_exact_answers.pt" -Force
    Copy-Item -Path "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_input_output_ids_0.pt" -Destination "../output/resampling/$($MODEL_FRIENDLY_NAMES[$MODEL])_${DATASET}_5_input_output_ids.pt" -Force
} else {
    Write-Host "警告: 源文件不存在: $SOURCE" -ForegroundColor Yellow
}

Write-Host "完成: ${DATASET} 重采样已完成" -ForegroundColor Green