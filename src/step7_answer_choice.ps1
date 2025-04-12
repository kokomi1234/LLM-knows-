# step7_answer_choice.ps1
# 运行前请先加载配置: . .\config.ps1

# 完整的数据集到token类型的映射
$TOKEN_MAPPING = @{
    # 默认使用exact_answer_last_token
    "default" = "exact_answer_last_token"
    
    # 特定配置 - 测试数据集
    "winobias_test" = "last_q_token"
    "winogrande_test" = "last_q_token"
    "mnli_validation" = "last_q_token"
    "imdb_test" = "last_q_token"
}

# 获取测试数据集列表
$TEST_DATASETS = $DATASETS | Where-Object { 
    $_ -like "*_test" -or 
    $_ -like "*_validation" 
}

Write-Host "答案选择的数据集: $($TEST_DATASETS -join ', ')" -ForegroundColor Cyan


foreach ($DATASET in $TEST_DATASETS) {
    Write-Host "===== 处理数据集: $DATASET - 答案选择 =====" -ForegroundColor Green
    
    # GPU内存清理
    Clear-GPUMemory
    
    # 选择适当的token
    $TOKEN = if ($TOKEN_MAPPING.ContainsKey($DATASET)) { 
        $TOKEN_MAPPING[$DATASET] 
    } else { 
        $TOKEN_MAPPING["default"] 
    }
    
    # 答案选择实验
    python ./probe_choose_answer.py --model $MODEL --probe_at mlp --layer 15 --token $TOKEN --dataset $DATASET --n_resamples 5 --seeds 0
    
    Write-Host "已完成 $DATASET 的答案选择实验" -ForegroundColor Yellow
}


Write-Host "步骤7完成: 答案选择实验已完成" -ForegroundColor Green