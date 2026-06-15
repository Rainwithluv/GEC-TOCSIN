# Data Directory

This directory contains or links to datasets for training and evaluation.

## Datasets

### TOCSIN Datasets

Located at: `../TOCSIN-main/TOCSIN-main/exp_API-based_model/data/`

Available datasets:
- `xsum_gpt-4.raw_data.json` - XSum dataset with GPT-4 generated texts
- `xsum_gpt-3.5-turbo.raw_data.json` - XSum dataset with GPT-3.5 generated texts
- `xsum_gemini.raw_data.json` - XSum dataset with Gemini generated texts
- `writing_gpt-4.raw_data.json` - Writing dataset with GPT-4 generated texts
- `writing_gpt-3.5-turbo.raw_data.json` - Writing dataset with GPT-3.5 generated texts
- `pubmed_gpt-4.raw_data.json` - PubMed dataset with GPT-4 generated texts
- And more...

Format:
```json
{
  "original": ["human text 1", "human text 2", ...],
  "sampled": ["llm text 1", "llm text 2", ...]
}
```

### GECSore Datasets

Located at: `../GECScore-main/GECScore-main/data/`

Available datasets:
- `xsum.raw_data.json` - XSum raw dataset
- `writing.raw_data.json` - Writing raw dataset

Format:
```json
{
  "id": "sample_id",
  "text": "text content",
  ...
}
```

## Usage

The `DataLoader` class in `src/utils/data_loader.py` handles loading from both formats:

```python
from src.utils.data_loader import DataLoader

loader = DataLoader(base_dir="../")

# Load TOCSIN format
data = loader.load_tocsin_data('xsum', 'gpt-4')

# Load GECSore format
data = loader.load_gecscore_raw_data('xsum')

# Auto-detect format
data = loader.load_combined_data('xsum', 'gpt-4')
```

## Data Linking

To avoid duplicating large files, consider creating symbolic links:

```bash
# On Linux/Mac
ln -s ../TOCSIN-main/TOCSIN-main/exp_API-based_model/data ./data/tocsin
ln -s ../GECScore-main/GECScore-main/data ./data/gecscore

# On Windows (as Administrator)
mklink /D data\tocsin ..\TOCSIN-main\TOCSIN-main\exp_API-based_model\data
mklink /D data\gecscore ..\GECScore-main\GECScore-main\data
```
