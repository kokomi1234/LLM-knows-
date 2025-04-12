# step4_generalization.ps1
# 运行前请先加载配置: . .\config.ps1

Write-Host "===== 执行泛化实验 =====" -ForegroundColor Green

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

# GPU内存清理
Clear-GPUMemory

Write-Host "泛化实验的数据集: $($TRAIN_DATASETS -join ', ')" -ForegroundColor Cyan

# # winobias到movies的泛化
# Write-Host "测试从winobias到movies的泛化" -ForegroundColor Yellow
# python probe.py --model $MODEL --probe_at mlp --seeds 0 --dataset winobias --layer 15 --token last_q_token --test_dataset movies

# # GPU内存清理
# Clear-GPUMemory

# # movies到winobias的泛化
# Write-Host "测试从movies到winobias的泛化" -ForegroundColor Yellow
# python probe.py --model $MODEL --probe_at mlp --seeds 0 --dataset movies --layer 15 --token exact_answer_last_token --test_dataset winobias

# 生成所有可能的数据集对并执行泛化实验
# 生成所有可能的数据集对并执行泛化实验
for ($i = 0; $i -lt $TRAIN_DATASETS.Count; $i++) {
    $sourceDataset = $TRAIN_DATASETS[$i]
    
    for ($j = 0; $j -lt $TRAIN_DATASETS.Count; $j++) {
        # 跳过相同的数据集
        if ($i -eq $j) { continue }
        
        $targetDataset = $TRAIN_DATASETS[$j]
        
        # GPU内存清理
        Clear-GPUMemory
        
        # 获取源数据集的正确token类型
        $TOKEN_SOURCE = if ($TOKEN_MAPPING.ContainsKey($sourceDataset)) { 
            $TOKEN_MAPPING[$sourceDataset] 
        } else { 
            $TOKEN_MAPPING["default"] 
        }
        
        Write-Host "测试从 $sourceDataset 到 $targetDataset 的泛化" -ForegroundColor Yellow
        python probe.py --model $MODEL --probe_at mlp --seeds 0 --dataset $sourceDataset --layer 15 --token $TOKEN_SOURCE --test_dataset $targetDataset
    }
}

Write-Host "步骤4完成: 泛化实验已完成" -ForegroundColor Green