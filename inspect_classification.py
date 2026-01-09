import polars as pl
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect Classified Questions Parquet")
    parser.add_argument("input_file", nargs='?', default="classified_questions.parquet", help="Path to the .parquet file")
    
    args = parser.parse_args()
    
    print(f"--- Loading {args.input_file} ---")
    try:
        df = pl.read_parquet(args.input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # 1. General Stats
    total_rows = len(df)
    print(f"Total Rows: {total_rows:,}\n")

    # 2. Distribution
    print("--- Distribution ---")
    distribution = df.group_by("label").len().sort("label")
    print(distribution)
    
    # Calculate percentages
    valid_count = df.filter(pl.col("label") == "Valid").height
    print(f"\nValid Percentage:   {(valid_count/total_rows):.2%}")
    print(f"Invalid Percentage: {1 - (valid_count/total_rows):.2%}\n")

    # 3. Examples
    pl.Config.set_fmt_str_lengths(200) # Ensure we see the full text
    pl.Config.set_tbl_rows(40)

    print("--- Examples: VALID (Subjective/Evaluative) ---")
    valid_examples = df.filter(pl.col("label") == "Valid").sample(n=min(20, valid_count), seed=42)
    print(valid_examples.select(["title", "label"]))

    print("\n--- Examples: INVALID (Narrative/Factual) ---")
    invalid_count = total_rows - valid_count
    invalid_examples = df.filter(pl.col("label") == "Invalid").sample(n=min(20, invalid_count), seed=42)
    print(invalid_examples.select(["title", "label"]))

if __name__ == "__main__":
    main()
