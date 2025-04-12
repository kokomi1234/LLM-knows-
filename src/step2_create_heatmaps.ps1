# step2_create_heatmaps.ps1
# 运行前请先加载配置: . .\config.ps1

# 获取训练数据集列表(排除_test和_validation后缀的数据集)
$TRAIN_DATASETS = $DATASETS | Where-Object { 
    $_ -notlike "*_test" -and 
    $_ -notlike "*_validation" 
}
Write-Host "创建热力图的数据集: $($TRAIN_DATASETS -join ', ')" -ForegroundColor Cyan

foreach ($DATASET in $TRAIN_DATASETS) {
    Write-Host "===== 处理数据集: $DATASET - 创建热力图 =====" -ForegroundColor Green
    
    # GPU内存清理
    Clear-GPUMemory
    
    # 创建热力图(使用较少样本以节省内存)
    python ./probe_all_layers_and_tokens.py --model $MODEL --probe_at mlp_last_layer_only_input --seed 0 --n_samples $SMALL_SAMPLES --dataset $DATASET
    
    Write-Host "已完成 $DATASET 的热力图生成，请在wandb上查看结果" -ForegroundColor Yellow
}

Write-Host "步骤2完成: 热力图已创建" -ForegroundColor Green