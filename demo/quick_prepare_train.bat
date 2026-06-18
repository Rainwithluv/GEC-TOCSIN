@echo off
REM 快速准备Attention训练数据 (使用少量样本快速测试)

echo ====================================
echo 快速准备训练数据 (测试模式)
echo ====================================
echo.
echo 使用少量样本 (50个) 快速测试训练流程
echo 预计耗时: 2-5分钟
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause >nul

cd /d d:\code_VScode\GEC-TOCSIN

echo.
echo Activating virtual environment...
call demo\venv\Scripts\activate.bat

echo.
echo Preparing training data (50 samples)...
python demo\prepare_training_data.py --dataset xsum --model gpt-4 --n-samples 50 --output demo\data\training_data_small.json

echo.
echo Checking data quality...
python demo\check_training_data.py demo\data\training_data_small.json

echo.
echo Training Attention model (quick test)...
python demo\train_attention.py --data demo\data\training_data_small.json --output demo\models\attention_model_small.pth --batch-size 16 --epochs 20

echo.
echo ====================================
echo Quick Test Complete!
echo ====================================
echo.
echo Small model saved to: demo\models\attention_model_small.pth
echo.
pause
