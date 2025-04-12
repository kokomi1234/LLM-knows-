# run_workflow.ps1 - 按步骤执行实验流程
# 使用方法: .\run_workflow.ps1 [-Steps 1,2,3] [-SkipExisting] [-ContinueOnError]

param (
    [Parameter(Mandatory=$false)]
    [int[]]$Steps = (1, 2, 3, 4, 5, 6, 7),
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipExisting,
    
    [Parameter(Mandatory=$false)]
    [switch]$ContinueOnError
)

# 加载配置
. .\config.ps1

# 验证配置已正确加载
if ($null -eq $DATASETS -or $DATASETS.Count -eq 0) {
    Write-Host "错误: 数据集列表为空！请确保已正确运行 '. .\config.ps1'" -ForegroundColor Red
    exit
}

# 记录开始时间
$startTime = Get-Date

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  LLMs Know More Than They Show - 实验工作流" -ForegroundColor Cyan
Write-Host "  开始时间: $startTime" -ForegroundColor Cyan
Write-Host "  模型: $MODEL" -ForegroundColor Cyan
Write-Host "  数据集: $($DATASETS -join ', ')" -ForegroundColor Cyan
Write-Host "  运行步骤: $($Steps -join ', ')" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

function Run-Step {
    param (
        [int]$StepNumber,
        [string]$Description,
        [string]$ScriptPath
    )
    
    if ($Steps -contains $StepNumber) {
        $stepStart = Get-Date
        # 修改第44行
        Write-Host "`n===== 步骤 ${StepNumber}: ${Description} =====" -ForegroundColor Green
        
        try {
            # 运行步骤脚本
            & $ScriptPath
            $success = $?
            
            if (-not $success) {
                throw "脚本执行返回了错误状态"
            }
            
            $stepEnd = Get-Date
            $duration = ($stepEnd - $stepStart).TotalMinutes
            Write-Host "步骤 $StepNumber 成功完成！用时: $($duration.ToString('0.0'))分钟" -ForegroundColor Green
        }
        catch {
            Write-Host "步骤 $StepNumber 执行出错: $_" -ForegroundColor Red
            
            if (-not $ContinueOnError) {
                Write-Host "终止工作流。使用 -ContinueOnError 参数可以在出错时继续执行。" -ForegroundColor Red
                exit 1
            } else {
                Write-Host "继续执行下一步..." -ForegroundColor Yellow
            }
        }
    } else {
        # 修改第70行
        Write-Host "`n跳过步骤 ${StepNumber}: ${Description}" -ForegroundColor DarkGray
    }
}

# 运行各步骤
Run-Step -StepNumber 1 -Description "生成模型答案" -ScriptPath ".\step1_generate_answers.ps1"
Run-Step -StepNumber 2 -Description "创建热力图" -ScriptPath ".\step2_create_heatmaps.ps1"
Run-Step -StepNumber 3 -Description "探测特定层和token" -ScriptPath ".\step3_probe_specific_layers.ps1"
Run-Step -StepNumber 4 -Description "泛化实验" -ScriptPath ".\step4_generalization.ps1"
Run-Step -StepNumber 5 -Description "重采样" -ScriptPath ".\step5_resampling.ps1"
Run-Step -StepNumber 6 -Description "错误类型探测" -ScriptPath ".\step6_error_type_probing.ps1"
Run-Step -StepNumber 7 -Description "答案选择" -ScriptPath ".\step7_answer_choice.ps1"

# 计算总时间
$endTime = Get-Date
$totalTime = ($endTime - $startTime).TotalMinutes

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  LLMs Know More Than They Show - 实验完成!" -ForegroundColor Cyan
Write-Host "  开始时间: $startTime" -ForegroundColor Cyan
Write-Host "  结束时间: $endTime" -ForegroundColor Cyan
Write-Host "  总用时: $($totalTime.ToString('0.0'))分钟" -ForegroundColor Cyan
Write-Host "  输出保存在: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan