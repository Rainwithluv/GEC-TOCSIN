"""
Data loading utilities for TOCSIN and GECSore datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


class DataLoader:
    """Load and process data from TOCSIN and GECSore formats."""

    def __init__(self, base_dir: str = "../"):
        """
        Initialize data loader.

        Args:
            base_dir: Base directory containing TOCSIN and GECSore projects
        """
        self.base_dir = Path(base_dir)

        # Try multiple data directory locations
        self.tocsin_data_dirs = [
            Path("demo/data"),  # Local demo/data directory
            Path("data"),  # Current directory data
            self.base_dir / "TOCSIN-main/TOCSIN-main/exp_API-based_model/data",
            Path("../TOCSIN-main/TOCSIN-main/exp_API-based_model/data"),
        ]

        self.gecscore_data_dirs = [
            Path("demo/data"),  # Local demo/data directory
            Path("data"),  # Current directory data
            self.base_dir / "GECScore-main/GECScore-main/data",
            Path("../GECScore-main/GECScore-main/data"),
        ]

    def load_tocsin_data(self, dataset: str, model: str) -> Dict[str, List[str]]:
        """
        Load TOCSIN format data.

        Args:
            dataset: Dataset name (e.g., 'xsum', 'writing', 'pubmed')
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'gemini')

        Returns:
            Dict with 'original' and 'sampled' text lists
        """
        filename = f"{dataset}_{model}.raw_data.json"

        # Try each directory
        for data_dir in self.tocsin_data_dirs:
            file_path = data_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    'original': data.get('original', []),
                    'sampled': data.get('sampled', [])
                }

        raise FileNotFoundError(f"Data file not found: {filename}. Tried directories: {self.tocsin_data_dirs}")

    def load_gecscore_raw_data(self, dataset: str) -> List[Dict]:
        """
        Load GECSore raw data format.

        Args:
            dataset: Dataset name ('xsum' or 'writing')

        Returns:
            List of data entries with 'id', 'text', etc.
        """
        file_path = self.gecscore_data_dir / f"{dataset}.raw_data.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def load_gecscore_test_data(self, dataset: str, model: str = "GPT-4o") -> List[Dict]:
        """
        Load GECSore test data format.

        Args:
            dataset: Dataset name ('xsum' or 'writing')
            model: Model name (default 'GPT-4o')

        Returns:
            List of test data entries
        """
        file_path = self.gecscore_data_dir / "normal_data" / f"{dataset}.{model}.normal.test_data.json"

        if not file_path.exists():
            # Try raw data as fallback
            return self.load_gecscore_raw_data(dataset)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def create_pair_data(self, data: List[Dict], text_key: str = 'text',
                        label_key: str = 'label') -> Tuple[List[str], List[str]]:
        """
        Create paired data for evaluation.

        Args:
            data: List of data entries
            text_key: Key for text content
            label_key: Key for label

        Returns:
            Tuple of (human_texts, llm_texts)
        """
        human_texts = []
        llm_texts = []

        for entry in data:
            label = entry.get(label_key, '').lower()
            text = entry.get(text_key, '')

            if label == 'human':
                human_texts.append(text)
            elif label in ['llm', 'machine', 'ai']:
                llm_texts.append(text)

        return human_texts, llm_texts

    def load_combined_data(self, dataset: str, model: str = 'gpt-4') -> Dict:
        """
        Load data from available sources (TOCSIN or GECSore).

        Args:
            dataset: Dataset name
            model: Model name

        Returns:
            Combined data dictionary
        """
        # Try TOCSIN format first
        try:
            return self.load_tocsin_data(dataset, model)
        except FileNotFoundError:
            pass

        # Try GECSore format
        try:
            data = self.load_gecscore_test_data(dataset, model.upper().replace('-', '.'))
            human_texts, llm_texts = self.create_pair_data(data)
            return {'original': human_texts, 'sampled': llm_texts}
        except FileNotFoundError:
            pass

        raise FileNotFoundError(f"No data found for dataset={dataset}, model={model}")


def load_test_data(file_path: str) -> Dict:
    """
    Load test data from a JSON file.

    Args:
        file_path: Path to the test data file

    Returns:
        Dictionary with test data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
