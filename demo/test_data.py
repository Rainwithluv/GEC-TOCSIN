"""
Quick test to verify data loading is working.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from demo.src.utils.data_loader import DataLoader
except ImportError:
    from demo.src.utils.data_loader import DataLoader


def test_data_loading():
    """Test data loading from different locations."""
    print("=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)

    # Test with different base_dir values
    base_dirs = ["../", "./", "../../"]

    for base_dir in base_dirs:
        print(f"\nTrying base_dir: '{base_dir}'")
        loader = DataLoader(base_dir=base_dir)

        try:
            data = loader.load_combined_data('xsum', 'gpt-4')
            print(f"  ✓ SUCCESS!")
            print(f"    Original: {len(data['original'])} samples")
            print(f"    Sampled: {len(data['sampled'])} samples")

            # Show sample
            print(f"\n  Sample original text: {data['original'][0][:80]}...")
            print(f"  Sample sampled text: {data['sampled'][0][:80]}...")

            return loader, base_dir

        except FileNotFoundError as e:
            print(f"  ✗ Failed: {e}")

    print("\n❌ All base_dir attempts failed!")
    print("\nChecking if data files exist...")
    import os
    if os.path.exists("demo/data/xsum_gpt-4.raw_data.json"):
        print("  ✓ File exists at: demo/data/xsum_gpt-4.raw_data.json")
    else:
        print("  ✗ File NOT found at: demo/data/xsum_gpt-4.raw_data.json")

    if os.path.exists("data/xsum_gpt-4.raw_data.json"):
        print("  ✓ File exists at: data/xsum_gpt-4.raw_data.json")
    else:
        print("  ✗ File NOT found at: data/xsum_gpt-4.raw_data.json")

    return None, None


if __name__ == '__main__':
    loader, base_dir = test_data_loading()

    if loader:
        print(f"\n✅ Use base_dir='{base_dir}' for the detector")
