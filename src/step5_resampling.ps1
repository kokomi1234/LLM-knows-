# step5_resampling.ps1
# 运行前请先加载配置: . .\config.ps1

foreach ($DATASET in $DATASETS) {
    Write-Host "===== 处理数据集: $DATASET - 重采样 =====" -ForegroundColor Green
    
    # GPU内存清理
    Clear-GPUMemory
    
    # 执行少量重采样(仅用于测试)
    python ./resampling.py --model $MODEL --seed 0 --dataset $DATASET --n_resamples 5 --tag 0
    
    # 统一重命名所有带标签的文件 (从带_0的文件创建无标签的标准文件)
    $friendlyModelName = $MODEL_FRIENDLY_NAMES[$MODEL]
    $baseOutputDir = "../output/resampling"
    
    # 处理_2_文件
    Write-Host "重命名_2_标签文件: ${DATASET}" -ForegroundColor Cyan
    $files = @(
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_textual_answers_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_textual_answers.pt"},
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_input_output_ids_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_input_output_ids.pt"},
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_exact_answers_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_2_exact_answers.pt"}
    )
    
    foreach ($file in $files) {
        if (Test-Path $file.Source) {
            Copy-Item -Path $file.Source -Destination $file.Target -Force
            Write-Host "  复制: $($file.Source) -> $($file.Target)" -ForegroundColor DarkGray
        }
    }
    
    # 处理_5_文件
    Write-Host "重命名_5_标签文件: ${DATASET}" -ForegroundColor Cyan
    $files = @(
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_textual_answers_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_textual_answers.pt"},
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_input_output_ids_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_input_output_ids.pt"},
        @{Source = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_exact_answers_0.pt"; Target = "${baseOutputDir}/${friendlyModelName}_${DATASET}_5_exact_answers.pt"}
    )
    
    foreach ($file in $files) {
        if (Test-Path $file.Source) {
            Copy-Item -Path $file.Source -Destination $file.Target -Force
            Write-Host "  复制: $($file.Source) -> $($file.Target)" -ForegroundColor DarkGray
        }
    }

    # 对需要的数据集提取重采样后的精确答案
    if ($EXTRACT_DATASETS -contains $DATASET) {
        # 提取精确答案
        Write-Host "提取重采样数据的精确答案: $DATASET" -ForegroundColor Yellow
        python ./extract_exact_answer.py --dataset $DATASET --do_resampling 5 --model $MODEL
    }
}

Write-Host "步骤5完成: 所有数据集的重采样已完成" -ForegroundColor Green