import polars as pl
import pyarrow as pa
import pyarrow.ipc
import argparse
import sys
import json
import gc
from tqdm import tqdm
import time

# List of known bot names to filter out
BOT_LIST = [
    "AutoModerator",
    "RemindMeBot",
    "WikiTextBot",
    "ClickableLinkBot",
    "haiku_bot",
    "image_linker_bot",
    "sneakpeekbot"
]

def read_jsonl_chunk(file_path, chunk_size):
    """Generator that yields chunks of lines from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    parser = argparse.ArgumentParser(description="Batched Reddit Data Processor (OOM Safe).")
    parser.add_argument("submissions_file", help="Path to the submissions .json/.jsonl file")
    parser.add_argument("comments_file", help="Path to the comments .json/.jsonl file")
    parser.add_argument("--output", default="output.arrow", help="Path for the output Arrow IPC file")
    parser.add_argument("--batch-size", type=int, default=34180339, help="Number of submissions to process at once")
    
    args = parser.parse_args()

    # Columns definitions
    sub_cols = ["id", "title", "author", "created_utc", "over_18", "score", "num_comments"]
    com_cols = ["id", "body", "ups", "downs", "created_utc", "author", "parent_id", "link_id"]

    print(f"--- Configuration ---")
    print(f"Submissions: {args.submissions_file}")
    print(f"Comments:    {args.comments_file}")
    print(f"Output:      {args.output}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"---------------------")

    # 1. Setup PyArrow Writer
    writer = None
    output_sink = None
    
    # We need a schema for the writer. We will define it explicitly to avoid inference issues 
    # if the first batch is empty or has nulls.
    # We use Polars to generate the Arrow schema from a dummy dataframe.
    dummy_struct = pl.DataFrame({
        "id": ["dummy"], "body": ["dummy"], "ups": [1], "downs": [1], 
        "created_utc": [1], "author": ["dummy"]
    }).to_struct("top_level_comments")

    dummy_schema_df = pl.DataFrame({
        "id": ["dummy"], "title": ["dummy"], "author": ["dummy"], 
        "created_utc": [1], "over_18": [True], "score": [1], "num_comments": [1]
    }).with_columns(
        pl.Series([ [dummy_struct[0]] ]).alias("top_level_comments")
    )
    
    # Cast to ensure types match our processing logic
    dummy_schema_df = dummy_schema_df.with_columns([
        pl.col("created_utc").cast(pl.Int64),
        pl.col("score").cast(pl.Int64),
        pl.col("num_comments").cast(pl.Int64),
    ])
    
    arrow_schema = dummy_schema_df.to_arrow().schema

    # 2. Count lines for progress bar (optional, but helpful)
    print("Counting total submissions (scanning file)...")
    total_lines = sum(1 for _ in open(args.submissions_file, 'r'))
    print(f"Total Submissions: {total_lines}")

    start_time = time.time()

    # Open the output file stream with ZSTD compression
    with pa.OSFile(args.output, 'wb') as sink:
        with pa.ipc.new_file(sink, arrow_schema, options=pa.ipc.IpcWriteOptions(compression='zstd')) as writer:
            
            # 3. Iterate through chunks
            with tqdm(total=total_lines, desc="Processing Submissions") as pbar:
                
                for chunk_lines in read_jsonl_chunk(args.submissions_file, args.batch_size):
                    
                    # --- A. Process Submissions Chunk ---
                    # We convert the list of JSON strings to a Polars DataFrame
                    # Using read_ndjson on a memory buffer of the chunk
                    chunk_str = "".join(chunk_lines)
                    
                    # UPDATE: Removed "edited" from schema_overrides as it was filtered out
                    df_subs_chunk = pl.read_ndjson(chunk_str.encode('utf-8'), schema_overrides={
                        "created_utc": pl.String, 
                        "score": pl.String, 
                        "num_comments": pl.String, 
                        "over_18": pl.String
                    }, ignore_errors=True)

                    # Clean Types
                    df_subs_chunk = df_subs_chunk.select(sub_cols).with_columns([
                        pl.col("created_utc").cast(pl.Float64, strict=False).cast(pl.Int64),
                        pl.col("score").cast(pl.Float64, strict=False).cast(pl.Int64),
                        pl.col("num_comments").cast(pl.Float64, strict=False).cast(pl.Int64),
                        # FIX: Manual boolean check for over_18 string
                        (pl.col("over_18") == "true").fill_null(False).alias("over_18"),
                    ]).filter(
                        (pl.col("author") != "[deleted]") &
                        (~pl.col("author").is_in(BOT_LIST))
                    )

                    # Get IDs for filtering comments
                    # Optimization: Prepend 't3_' here so we can filter match strictly in the DB
                    current_batch_ids = df_subs_chunk["id"].to_list()
                    link_ids_to_find = ["t3_" + x for x in current_batch_ids]

                    if not current_batch_ids:
                        pbar.update(len(chunk_lines))
                        continue

                    # --- B. Find Matches in Comments File ---
                    # We lazy scan the FULL comments file, but we filter strictly by the current IDs.
                    # This is I/O heavy (scanning comments file multiple times), but Memory safe.
                    try:
                        lf_coms = pl.scan_ndjson(
                            args.comments_file,
                            schema_overrides={"created_utc": pl.String, "ups": pl.String, "downs": pl.String}
                        )
                        
                        # Filter: link_id is in our current batch list
                        # We also apply the standard cleanup filters
                        matched_comments = lf_coms.select(com_cols).filter(
                            (pl.col("link_id").is_in(link_ids_to_find)) & 
                            (pl.col("parent_id").str.starts_with("t3_")) &
                            (pl.col("body") != "[deleted]") &
                            (pl.col("body") != "[removed]") &
                            (pl.col("author") != "[deleted]") &
                            (~pl.col("author").is_in(BOT_LIST))
                        ).collect() # Materialize ONLY matching comments

                    except Exception as e:
                        print(f"Error scanning comments: {e}")
                        continue

                    # Clean matched comments types
                    matched_comments = matched_comments.with_columns([
                        pl.col("created_utc").cast(pl.Float64, strict=False).cast(pl.Int64),
                        pl.col("ups").cast(pl.Float64, strict=False).cast(pl.Int64),
                        pl.col("downs").cast(pl.Float64, strict=False).cast(pl.Int64),
                        # Strip t3_ for joining back to submissions
                        pl.col("link_id").str.strip_prefix("t3_").alias("link_id_clean")
                    ])

                    # --- C. Aggregate Comments ---
                    comment_struct = pl.struct([
                        pl.col("id"), pl.col("body"), pl.col("ups"), 
                        pl.col("downs"), pl.col("created_utc"), pl.col("author")
                    ])

                    # Group by ID
                    df_coms_agg = matched_comments.group_by("link_id_clean").agg(
                        comment_struct.alias("top_level_comments")
                    )

                    # --- D. Join ---
                    df_batch_final = df_subs_chunk.join(
                        df_coms_agg,
                        left_on="id",
                        right_on="link_id_clean",
                        how="left"
                    )

                    # --- E. Write Batch ---
                    # Convert to Arrow Table
                    pa_table = df_batch_final.to_arrow()
                    
                    # Ensure Schema match (cast if necessary, though Polars usually handles it)
                    # We cast to the schema we defined at the start to ensure consistency across batches
                    pa_table = pa_table.cast(arrow_schema)
                    
                    writer.write_table(pa_table)
                    
                    # --- F. Cleanup ---
                    del df_subs_chunk
                    del matched_comments
                    del df_coms_agg
                    del df_batch_final
                    del pa_table
                    gc.collect()

                    pbar.update(len(chunk_lines))

    end_time = time.time()
    print(f"\nDone! processed in {end_time - start_time:.2f} seconds.")

    # Verification
    print("\n--- Verifying Output (First 2 Records) ---")
    df_verify = pl.read_ipc(args.output)
    pl.Config.set_fmt_str_lengths(100)
    pl.Config.set_tbl_rows(5)
    print(df_verify.head(2))

if __name__ == "__main__":
    main()
