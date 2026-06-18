"""
抑制Hugging Face警告的脚本
在使用detector前运行此脚本
"""

import os
import warnings

# 抑制特定警告
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 抑制UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

print("✓ Hugging Face警告已抑制")
print("\n现在运行detector：")
print("  python -m demo.src.detector --mode evaluate --dataset xsum --model gpt-4 --fusion voting --n-samples 50")
