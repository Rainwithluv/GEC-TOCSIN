@echo off
REM Test Dynamic Attention Fusion strategy

echo ====================================
echo Dynamic Attention Fusion Test
echo ====================================
echo.
echo This will test the NEW dynamic fusion strategy and compare with weighted fusion.
echo.
echo Dynamic fusion adapts weights for each sample based on:
echo   - Channel confidence
echo   - Information entropy
echo   - Hybrid of both (recommended)
echo.
echo Expected improvements:
echo   - Writing dataset: Higher Recall (currently 0.56 with fixed weights)
echo   - XSum dataset: Maintain or improve performance
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause >nul

cd /d d:\code_VScode\GEC-TOCSIN

echo.
echo Activating virtual environment...
call demo\venv\Scripts\activate.bat

echo.
echo ========================================
echo Test 1: Weighted Fusion (baseline)
echo ========================================
echo.
echo Testing on XSum dataset...
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion weighted --n-samples 50

echo.
echo ========================================
echo Test 2: Dynamic Fusion (Confidence)
echo ========================================
echo.
echo Testing on XSum dataset...
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion dynamic --dynamic-mode confidence --n-samples 50

echo.
echo ========================================
echo Test 3: Dynamic Fusion (Hybrid - Recommended)
echo ========================================
echo.
echo Testing on XSum dataset...
python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion dynamic --dynamic-mode hybrid --n-samples 50

echo.
echo ========================================
echo Test 4: Dynamic Fusion on Writing Dataset
echo ========================================
echo.
echo Testing on Writing dataset (where dynamic fusion should help most)...
python -m demo.src.detector --mode evaluate --dataset writing --model gpt-4 --fusion dynamic --dynamic-mode hybrid --n-samples 50

echo.
echo ====================================
echo Dynamic Fusion Test Complete!
echo ====================================
echo.
echo Key improvements to look for:
echo   - Writing dataset Recall should be higher than 0.56
echo   - XSum dataset performance should be maintained
echo   - ROC AUC should be competitive or better
echo.
pause
