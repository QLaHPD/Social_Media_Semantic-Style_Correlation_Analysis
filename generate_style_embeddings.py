import torch
import torch.nn as nn
import torch.multiprocessing as mp
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import argparse
import sys
import os
import queue
from tqdm import tqdm
import time

# Set start method to 'spawn' is required for CUDA multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# -----------------------------------------------------------------------------
# 1. WORKER PROCESS (Runs on specific GPU)
# -----------------------------------------------------------------------------
def gpu_worker(gpu_id, task_queue, result_queue, model_name, batch_size):
    """
    Independent process that sits on a specific GPU.
    It loads its own model replica and consumes tasks from the queue.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Initializing model...")

        # Load Tokenizer (Fast, Rust-based)
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        
        # Load Model directly to the specific GPU
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        
        print(f"[GPU {gpu_id}] Ready to process.")

        while True:
            try:
                # Get task with a timeout to allow checking for exit conditions
                task = task_queue.get(timeout=5)
            except queue.Empty:
                # If queue is empty and we are told to stop (sentinel checked later), or just wait
                continue

            # Sentinel to kill the process
            if task is None:
                break

            author, texts = task
            
            # --- INFERENCE LOGIC ---
            total_sum = None
            count = 0
            
            # Manual batching to avoid DataLoader overhead for simple lists
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                with torch.no_grad():
                    # Mixed precision for speed
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask=attention_mask)
                        # Pooler output is usually [Batch, Hidden]
                        embeddings = outputs.pooler_output
                
                # Sum results on GPU
                curr_sum = torch.sum(embeddings, dim=0)
                
                if total_sum is None:
                    total_sum = curr_sum
                else:
                    total_sum += curr_sum
                
                count += len(batch_texts)

            # Calculate Mean and move to CPU
            if total_sum is not None:
                mean_embedding = (total_sum / count).float().cpu().numpy().tolist()
                result_queue.put((author, mean_embedding, count))
            
            # -----------------------

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        # Send error signal or None to ensure writer doesn't hang
        result_queue.put(None)

# -----------------------------------------------------------------------------
# 2. MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Independent Instance Style Embeddings")
    parser.add_argument("input_file", help="Path to the .arrow file")
    parser.add_argument("--output", default="author_style_embeddings.parquet", help="Output Parquet file")
    parser.add_argument("--min-comments", type=int, default=50, help="Min comments per author")
    parser.add_argument("--min-chars", type=int, default=255, help="Min chars per comment")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    
    args = parser.parse_args()
    
    model_name = 'AIDA-UPM/star'

    if not torch.cuda.is_available():
        print("Error: CUDA is required for this script.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"--- Configuration ---")
    print(f"GPUs Available: {num_gpus}")
    print(f"Input: {args.input_file}")
    
    # ---------------------------------------------------------
    # STEP 1: LOAD & GROUP DATA
    # ---------------------------------------------------------
    print("Loading and grouping data with Polars...")
    
    try:
        lf = pl.scan_ipc(args.input_file)
    except:
        lf = pl.scan_arrow(args.input_file)

    # Filtering Logic
    q = (
        lf.select(pl.col("top_level_comments").list.explode())
        .select([
            pl.col("top_level_comments").struct.field("author").alias("author"),
            pl.col("top_level_comments").struct.field("body").alias("body")
        ])
        .filter(pl.col("body").str.len_chars() >= args.min_chars)
        .filter(pl.col("author").count().over("author") >= args.min_comments)
        .group_by("author")
        .agg(pl.col("body"))
    )
    
    df_authors = q.collect()
    total_authors = len(df_authors)
    print(f"Found {total_authors:,} valid authors.")
    
    if total_authors == 0:
        return

    # ---------------------------------------------------------
    # STEP 2: SETUP QUEUES & WORKERS
    # ---------------------------------------------------------
    # Queue for sending (Author, [Texts]) to workers
    # We limit size to prevent RAM explosion if loading is faster than GPU
    task_queue = mp.Queue(maxsize=100) 
    
    # Queue for receiving (Author, Embedding) from workers
    result_queue = mp.Queue()

    workers = []
    for i in range(num_gpus):
        p = mp.Process(
            target=gpu_worker, 
            args=(i, task_queue, result_queue, model_name, args.batch_size)
        )
        p.start()
        workers.append(p)

    # ---------------------------------------------------------
    # STEP 3: FEEDER THREAD (Main Process) & WRITER LOOP
    # ---------------------------------------------------------
    
    # Parquet Schema
    schema = pa.schema([
        ('author', pa.string()),
        ('embedding', pa.list_(pa.float32())),
        ('num_comments_used', pa.int32())
    ])
    pq_writer = pq.ParquetWriter(args.output, schema)

    # We iterate manually to feed the queue
    author_iter = df_authors.iter_rows(named=True)
    
    # Progress Bar
    pbar = tqdm(total=total_authors, desc="Processing", unit="author")
    
    # State tracking
    sent_count = 0
    received_count = 0
    authors_exhausted = False

    while received_count < total_authors:
        
        # 1. Feed Tasks (Non-blocking attempt)
        if not authors_exhausted:
            try:
                # Try to fill queue until full
                while not task_queue.full():
                    row = next(author_iter)
                    task_queue.put((row['author'], row['body']))
                    sent_count += 1
            except StopIteration:
                authors_exhausted = True
                # Send kill signals (one None per worker)
                for _ in range(num_gpus):
                    task_queue.put(None)

        # 2. Collect Results
        # We fetch items from result queue. 
        # If queue is empty, we wait briefly, but we keep looping to ensure we feed tasks.
        try:
            while True:
                # Non-blocking get
                result = result_queue.get_nowait()
                
                # Check for error signal
                if result is None:
                    print("A worker encountered an error.")
                    continue # Or handle break logic

                author_res, emb_res, count_res = result
                
                # Write to Parquet immediately
                table = pa.Table.from_pylist([{
                    'author': author_res,
                    'embedding': emb_res,
                    'num_comments_used': count_res
                }], schema=schema)
                
                pq_writer.write_table(table)
                
                received_count += 1
                pbar.update(1)
                
        except queue.Empty:
            # If result queue is empty, sleep briefly to let GPUs work
            # This prevents the CPU loop from spinning 100%
            time.sleep(0.01)

    pbar.close()
    pq_writer.close()
    
    # Join workers
    for p in workers:
        p.join()

    print(f"\nDone! Saved to {args.output}")

if __name__ == "__main__":
    main()
