import polars as pl
import argparse
import sys
import time
from datetime import datetime

def write_stat(f, label, value):
    """Helper to write formatted lines to file and print to console"""
    line = f"{label:<40} : {value}"
    print(line)
    f.write(line + "\n")

def write_section(f, title):
    """Helper for section headers"""
    line = f"\n{'='*20} {title} {'='*20}"
    print(line)
    f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze Reddit Arrow Dataset.")
    parser.add_argument("input_file", help="Path to the .arrow output file")
    parser.add_argument("--output", default="statistics.txt", help="Output text file")
    
    args = parser.parse_args()

    print(f"Loading data from: {args.input_file}")
    start_time = time.time()

    # Lazy load the Arrow file
    try:
        lf = pl.scan_ipc(args.input_file)
    except Exception:
        lf = pl.scan_arrow(args.input_file)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Statistics Report for {args.input_file}\n")
        f.write(f"Generated on: {datetime.now()}\n")
        
        # ---------------------------------------------------------
        # 1. GENERAL METADATA
        # ---------------------------------------------------------
        write_section(f, "General Metadata")
        
        # Total Submissions
        total_subs = lf.select(pl.len()).collect().item()
        write_stat(f, "Total Submissions", f"{total_subs:,}")

        # Date Range
        print("Calculating Date Range...")
        date_stats = lf.select([
            pl.col("created_utc").min().alias("min_ts"),
            pl.col("created_utc").max().alias("max_ts"),
            pl.col("over_18").sum().alias("nsfw_count")
        ]).collect()
        
        min_date = datetime.fromtimestamp(date_stats["min_ts"][0])
        max_date = datetime.fromtimestamp(date_stats["max_ts"][0])
        nsfw_count = date_stats["nsfw_count"][0]
        
        write_stat(f, "Date Range Start", min_date)
        write_stat(f, "Date Range End", max_date)
        write_stat(f, "NSFW Submissions", f"{nsfw_count:,} ({nsfw_count/total_subs:.2%})")

        # ---------------------------------------------------------
        # 2. COMMENT COUNTS (per submission)
        # ---------------------------------------------------------
        write_section(f, "Comments per Submission (Top-Level)")
        print("Calculating Comment Count Distribution...")
        
        # FIX 1: fill_null(0) ensures posts with no comments are counted as 0, not skipped
        list_len_stats = lf.select(
            pl.col("top_level_comments").list.len().fill_null(0).alias("count")
        ).select([
            # FIX 2: Explicit Aliases to prevent DuplicateError
            pl.mean("count").alias("count_mean"),
            pl.std("count").alias("count_std"),
            pl.col("count").quantile(0.01).alias("p01"),
            pl.col("count").median().alias("p50"),
            pl.col("count").quantile(0.99).alias("p99"),
            pl.max("count").alias("count_max")
        ]).collect()

        write_stat(f, "Mean Comments per Sub", f"{list_len_stats['count_mean'][0]:.4f}")
        write_stat(f, "Std Dev Comments", f"{list_len_stats['count_std'][0]:.4f}")
        write_stat(f, "1st Percentile (p1)", f"{list_len_stats['p01'][0]:.0f}")
        write_stat(f, "Median (p50)", f"{list_len_stats['p50'][0]:.0f}")
        write_stat(f, "99th Percentile (p99)", f"{list_len_stats['p99'][0]:.0f}")
        write_stat(f, "Max Comments in one Sub", f"{list_len_stats['count_max'][0]:.0f}")

        # ---------------------------------------------------------
        # 3. TEXT LENGTHS & SCORES
        # ---------------------------------------------------------
        write_section(f, "Content Analysis (Lengths & Scores)")
        print("Calculating Submission Text/Score Stats...")

        sub_stats = lf.select([
            pl.col("title").str.len_chars().mean().alias("title_len_mean"),
            pl.col("score").mean().alias("score_mean"),
            pl.col("score").median().alias("score_median")
        ]).collect()

        write_stat(f, "Avg Submission Title Length", f"{sub_stats['title_len_mean'][0]:.2f} chars")
        write_stat(f, "Avg Submission Score", f"{sub_stats['score_mean'][0]:.2f}")
        write_stat(f, "Median Submission Score", f"{sub_stats['score_median'][0]:.0f}")

        print("Calculating Comment Body/Ups Stats (This scans exploded lists)...")
        
        com_stats = lf.select(
            pl.col("top_level_comments").list.explode()
        ).select([
            pl.col("top_level_comments").struct.field("body").str.len_chars().mean().alias("body_len_mean"),
            pl.col("top_level_comments").struct.field("ups").mean().alias("ups_mean"),
            pl.col("top_level_comments").struct.field("downs").mean().alias("downs_mean"),
        ]).collect()

        write_stat(f, "Avg Comment Body Length", f"{com_stats['body_len_mean'][0]:.2f} chars")
        write_stat(f, "Avg Comment Upvotes", f"{com_stats['ups_mean'][0]:.2f}")
        write_stat(f, "Avg Comment Downvotes", f"{com_stats['downs_mean'][0]:.2f}")

        # ---------------------------------------------------------
        # 4. AUTHOR ANALYSIS
        # ---------------------------------------------------------
        write_section(f, "Author Analysis")
        print("Calculating Unique Authors (This is memory intensive)...")

        # 1. Submission Authors
        print(" -> Fetching Submission Authors...")
        sub_authors_df = lf.select("author").unique().collect()
        unique_sub_authors = set(sub_authors_df["author"].to_list())
        del sub_authors_df
        
        # 2. Comment Authors
        print(" -> Fetching Comment Authors...")
        com_authors_df = lf.select(
            pl.col("top_level_comments").list.explode().struct.field("author")
        ).unique().collect()
        
        unique_com_authors = set(com_authors_df["author"].to_list())
        del com_authors_df

        # 3. Set Operations
        print(" -> Calculating Sets...")
        total_unique = len(unique_sub_authors.union(unique_com_authors))
        intersection = len(unique_sub_authors.intersection(unique_com_authors))

        write_stat(f, "Unique Submission Authors", f"{len(unique_sub_authors):,}")
        write_stat(f, "Unique Comment Authors", f"{len(unique_com_authors):,}")
        write_stat(f, "Total Unique Authors (Union)", f"{total_unique:,}")
        write_stat(f, "Authors in Both (Intersection)", f"{intersection:,}")
        
        if len(unique_sub_authors) > 0:
            pct_intersect = intersection/len(unique_sub_authors)
        else:
            pct_intersect = 0
            
        write_stat(f, "% Sub Authors who also Commented", f"{pct_intersect:.2%}")

    print(f"\nAnalysis Complete! Results saved to {args.output}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
