# step6_error_type_probing.ps1
# 运行前请先加载配置: . .\config.ps1

# 完整的数据集到token类型的映射
$TOKEN_MAPPING = @{
    # 默认使用exact_answer_last_token
    "default" = "exact_answer_last_token"
    
    # 特定配置
    "winobias" = "last_q_token"
    "winogrande" = "last_q_token"
    "mnli" = "last_q_token"
    "imdb" = "last_q_token"
    
    # 测试数据集
    "winobias_test" = "last_q_token"
    "winogrande_test" = "last_q_token"
    "mnli_validation" = "last_q_token"
    "imdb_test" = "last_q_token"
}

# 获取训练数据集列表(排除_test和_validation后缀的数据集)
$TRAIN_DATASETS = $DATASETS | Where-Object { 
    $_ -notlike "*_test" -and 
    $_ -notlike "*_validation" 
}

Write-Host "错误类型探测的数据集: $($DATASETS -join ', ')" -ForegroundColor Cyan

foreach ($DATASET in $TRAIN_DATASETS) {
    Write-Host "===== 处理数据集: $DATASET - 错误类型探测 =====" -ForegroundColor Green
    
    # GPU内存清理
    Clear-GPUMemory
    
    # 选择适当的token
    $TOKEN = if ($TOKEN_MAPPING.ContainsKey($DATASET)) { 
        $TOKEN_MAPPING[$DATASET] 
    } else { 
        $TOKEN_MAPPING["default"] 
    }
    
    # 错误类型探测
    python ./probe_type_of_error.py --model $MODEL --probe_at mlp --seeds 0 --n_samples $ALL_SAMPLES --n_resamples 5 --token $TOKEN --dataset $DATASET --layer 15 --merge_types
    
    Write-Host "已完成 $DATASET 的错误类型探测" -ForegroundColor Yellow
}

Write-Host "步骤6完成: 错误类型探测已完成" -ForegroundColor Green