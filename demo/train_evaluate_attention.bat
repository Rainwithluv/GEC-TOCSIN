@echo off
REM Attention Mode 训练和评估完整流程
REM 使用现有data目录中的JSON文件

echo ====================================
echo Attention Mode 训练和评估流程
echo ====================================
echo.
echo 此脚本将:
echo 1. 训练Attention模型
echo 2. 评估训练好的模型
echo 3. 与固定权重方法对比
echo.
echo 可用数据集:
echo   - xsum.GPT-4o.normal.test_data.json
echo   - writing.GPT-4o.normal.test_data.json
echo   - xsum.Claude-3.5-Sonnet.normal.test_data.json
echo   - writing.Claude-3.5-Sonnet.normal.test_data.json
echo   - 以及更多...
echo.
echo 请选择要使用的数据集编号:
echo   1. xsum.GPT-4o (推荐，效果较好)
echo   2. writing.GPT-4o
echo   3. xsum.Claude-3.5-Sonnet
echo   4. writing.Claude-3.5-Sonnet
echo.

set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" set DATA_FILE=data\xsum.GPT-4o.normal.test_data.json
if "%choice%"=="2" set DATA_FILE=data\writing.GPT-4o.normal.test_data.json
if "%choice%"=="3" set DATA_FILE=data\xsum.Claude-3.5-Sonnet.normal.test_data.json
if "%choice%"=="4" set DATA_FILE=data\writing.Claude-3.5-Sonnet.normal.test_data.json

if "%DATA_FILE%"=="" (
    echo 无效的选择!
    pause
    exit /b 1
)

echo.
echo 选择的数据集: %DATA_FILE%
echo.

REM 检查文件是否存在
if not exist "%DATA_FILE%" (
    echo 错误: 文件不存在: %DATA_FILE%
    pause
    exit /b 1
)

cd /d d:\code_VScode\GEC-TOCSIN

echo.
echo 请选择模式:
echo   1. 快速测试 (使用100样本，约5分钟)
echo   2. 完整训练 (使用全部样本，约20-40分钟)
echo.

set /p mode="请输入选择 (1-2): "

if "%mode%"=="1" (
    set N_SAMPLES=100
    set EPOCHS=30
    echo 模式: 快速测试 (100样本, 30轮)
) else (
    set N_SAMPLES=
    set EPOCHS=50
    echo 模式: 完整训练 (全部样本, 50轮)
)

echo.
echo Press Ctrl+C to cancel or any key to continue...
pause >nul

echo.
echo Activating virtual environment...
call demo\venv\Scripts\activate.bat

REM 设置输出路径
set MODEL_PATH=models\attention_model_%random%.pth
set RESULTS_PATH=results\attention_eval_%random%.json

REM 创建输出目录
if not exist models mkdir models
if not exist results mkdir results

echo.
echo ====================================
echo 步骤 1: 训练Attention模型
echo ====================================
echo.
echo 配置:
echo   数据: %DATA_FILE%
echo   样本数: %N_SAMPLES% (未指定=全部)
echo   训练轮数: %EPOCHS%
echo   模型保存: %MODEL_PATH%
echo.

if defined N_SAMPLES (
    python demo\train_attention_simple.py --data %DATA_FILE% --output %MODEL_PATH% --n-samples %N_SAMPLES% --epochs %EPOCHS%
) else (
    python demo\train_attention_simple.py --data %DATA_FILE% --output %MODEL_PATH% --epochs %EPOCHS%
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 错误: 训练失败!
    pause
    exit /b 1
)

echo.
echo ====================================
echo 步骤 2: 评估模型
echo ====================================
echo.

python demo\evaluate_attention.py --model %MODEL_PATH% --data %DATA_FILE% --compare --output %RESULTS_PATH%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 错误: 评估失败!
    pause
    exit /b 1
)

echo.
echo ====================================
echo 训练和评估完成!
echo ====================================
echo.
echo 训练好的模型: %MODEL_PATH%
echo 评估结果: %RESULTS_PATH%
echo.
echo 下一步:
echo   1. 查看评估结果JSON文件
echo   2. 在其他数据集上测试模型泛化能力
echo   3. 调整超参数重新训练
echo.
pause
