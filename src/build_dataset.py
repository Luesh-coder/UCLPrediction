"""Build data/processed (reference tables) and data/model (model-ready train/predict files).

Usage:  python src/build_dataset.py
"""
import sys

from ucl_data.build import build_all

if __name__ == "__main__":
    result = build_all()
    for name, df in result.items():
        if not name.startswith("_"):
            print(f"{name:26s} {len(df):7d} rows  {df.shape[1]:3d} cols")
    print("duplicates removed:", result["_duplicates_removed"])
    issues = result["_issues"]
    for issue in issues:
        print("VALIDATION:", issue)
    print("validation passed" if not issues else f"{len(issues)} validation issue(s)")
    sys.exit(1 if issues else 0)
