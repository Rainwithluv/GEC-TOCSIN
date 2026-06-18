@echo off
REM Attention Mode 训练流程完整脚本
REM 包括: 数据准备 → 质量检查 → 模型训练

echo ====================================
echo Attention Mode 训练流程
echo ====================================
echo.
echo 此脚本将:
echo 1. 准备训练数据 (计算CoEdIT和TOCSIN分数)
echo 2. 检查数据质量
echo 3. 训练Attention模型
echo.
echo 预计耗时: 10-30分钟 (取决于样本数量)
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause >nul

cd /d d:\code_VScode\GEC-TOCSIN

echo.
echo ====================================
echo 步骤 1: 准备训练数据
echo ====================================
echo.
echo 配置:
echo   数据集: xsum
echo   模型: gpt-4
echo   每类别样本数: 250
echo   总样本数: 500
echo.

call demo\venv\Scripts\activate.bat

echo 正在计算分数 (这可能需要几分钟)...
python demo\prepare_training_data.py --dataset xsum --model gpt-4 --n-samples 250 --output demo\data\training_data.json

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 错误: 数据准备失败!
    pause
    exit /b 1
)

echo.
echo ====================================
echo 步骤 2: 检查数据质量
echo ====================================
echo.

python demo\check_training_data.py demo\data\training_data.json

echo.
echo ====================================
echo 步骤 3: 训练Attention模型
echo ====================================
echo.
echo 训练配置:
echo   批次大小: 32
echo   训练轮数: 50
echo   学习率: 0.001
echo   隐藏维度: 16
echo   注意力头数: 2
echo.

python demo\train_attention.py --data demo\data\training_data.json --output demo\models\attention_model.pth --batch-size 32 --epochs 50

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 错误: 训练失败!
    pause
    exit /b 1
)

echo.
echo ====================================
echo 训练完成!
echo ====================================
echo.
echo 训练好的模型已保存到: demo\models\attention_model.pth
echo.
echo 使用模型进行评估:
echo   python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion dynamic --dynamic-mode attention --n-samples 50
echo.
pause
