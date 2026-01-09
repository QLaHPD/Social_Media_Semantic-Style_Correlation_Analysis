import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
import argparse
import sys
import os
import queue
from tqdm import tqdm
import time
import gc

# Force spawn
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# -----------------------------------------------------------------------------
# 1. MODEL UTILS
# -----------------------------------------------------------------------------
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        # Ensure indices are on the correct device
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# -----------------------------------------------------------------------------
# 2. WORKER PROCESS
# -----------------------------------------------------------------------------
def gpu_worker(gpu_id, task_queue, result_queue, model_name, batch_size):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[GPU {gpu_id}] Initializing {model_name}...")

        # Tokenizer must be left-padded for the pooling logic to work
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        
        # Load Model
        # using "eager" to avoid Flash Attention import errors
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": gpu_id},
            torch_dtype=torch.float16,
            attn_implementation="eager" 
        )
        model.eval()
        
        print(f"[GPU {gpu_id}] Ready.")

        while True:
            # We fetch a BATCH of tasks to minimize queue overhead
            batch_tasks = []
            
            try:
                # Blocking get for first item
                item = task_queue.get(timeout=2)
                if item is None: break # Sentinel
                batch_tasks.append(item)
                
                # Non-blocking get for rest of batch
                for _ in range(batch_size - 1):
                    try:
                        item = task_queue.get_nowait()
                        if item is None: 
                            task_queue.put(None) # Re-queue sentinel
                            break
                        batch_tasks.append(item)
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if not batch_tasks:
                break

            # Unpack batch
            # task = (question_id, author, answer_text)
            q_ids = [t[0] for t in batch_tasks]
            authors = [t[1] for t in batch_tasks]
            texts = [t[2] for t in batch_tasks]

            # --- INFERENCE ---
            max_len = 2048 
            
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            
            batch_dict = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings_cpu = embeddings.float().cpu().numpy()

            # --- SEND RESULTS ---
            for i in range(len(batch_tasks)):
                result_queue.put({
                    'question_id': q_ids[i],
                    'author': authors[i],
                    'embedding': embeddings_cpu[i].tolist()
                })

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 3. MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings for Specific Answers")
    parser.add_argument("data_file", help="Path to output.arrow (submissions+comments)")
    parser.add_argument("questions_file", help="Path to classified_questions.parquet")
    parser.add_argument("style_file", help="Path to style_embeddings.parquet (to get valid authors)")
    parser.add_argument("--output", default="answer_embeddings.parquet", help="Output file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--min-chars", type=int, default=250, help="Minimum char length for answer to be included")
    
    args = parser.parse_args()
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    
    num_gpus = torch.cuda.device_count()

    # ---------------------------------------------------------
    # STEP 1: LOAD METADATA
    # ---------------------------------------------------------
    print("--- Loading Metadata ---")
    
    print(f"Loading Valid Authors from {args.style_file}...")
    df_style = pl.read_parquet(args.style_file, columns=["author"])
    valid_authors = set(df_style["author"].to_list())
    print(f"-> {len(valid_authors):,} authors loaded.")

    print(f"Loading Valid Questions from {args.questions_file}...")
    df_questions = pl.read_parquet(args.questions_file)
    valid_q_ids = set(df_questions.filter(pl.col("label") == "Valid")["id"].to_list())
    print(f"-> {len(valid_q_ids):,} valid questions loaded.")

    # ---------------------------------------------------------
    # STEP 2: PREPARE DATA (Polars)
    # ---------------------------------------------------------
    print("\n--- Scanning & Filtering Data ---")
    print(f"Minimum Answer Length: {args.min_chars} characters")
    
    try:
        lf = pl.scan_ipc(args.data_file)
    except:
        lf = pl.scan_arrow(args.data_file)

    print("Executing query... (this filters the massive dataset)")
    
    valid_q_ids_list = list(valid_q_ids)
    valid_authors_list = list(valid_authors)

    q = (
        lf.filter(pl.col("id").is_in(valid_q_ids_list))
        .select(["id", "top_level_comments"]) 
        .explode("top_level_comments")         
        .select([
            pl.col("id").alias("question_id"),
            pl.col("top_level_comments").struct.field("author").alias("author"),
            pl.col("top_level_comments").struct.field("body").alias("body"),
            pl.col("top_level_comments").struct.field("created_utc").alias("created_utc")
        ])
        .filter(pl.col("author").is_in(valid_authors_list))
        .filter(pl.col("body").is_not_null())
        # NEW: Filter by length
        .filter(pl.col("body").str.len_chars() >= args.min_chars)
    )
    
    df_answers = q.collect()
    
    print(f"Found {len(df_answers):,} candidate answers.")
    
    if len(df_answers) == 0:
        print("No matching answers found (try lowering --min-chars if this is unexpected).")
        return

    print("Sorting and deduplicating to find oldest answers...")
    # Sort by Question -> Author -> Date Ascending
    df_answers = df_answers.sort(["question_id", "author", "created_utc"])
    
    # Unique on (Question, Author) keeping 'first' (oldest)
    df_final_tasks = df_answers.unique(subset=["question_id", "author"], keep="first")
    
    total_tasks = len(df_final_tasks)
    print(f"Final Task Count: {total_tasks:,} answers to process.")

    # ---------------------------------------------------------
    # STEP 3: EXECUTION
    # ---------------------------------------------------------
    task_queue = mp.Queue(maxsize=10000)
    result_queue = mp.Queue()
    
    workers = []
    for i in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(i, task_queue, result_queue, model_name, args.batch_size)
        )
        p.start()
        workers.append(p)

    schema = pa.schema([
        ('question_id', pa.string()),
        ('author', pa.string()),
        ('embedding', pa.list_(pa.float32()))
    ])
    
    pq_writer = None
    pbar = None
    
    try:
        pq_writer = pq.ParquetWriter(args.output, schema)
        pbar = tqdm(total=total_tasks, desc="Embedding Answers")
        
        data_iter = df_final_tasks.select(["question_id", "author", "body"]).iter_rows()
        
        sent_count = 0
        received_count = 0
        exhausted = False
        
        while received_count < total_tasks:
            # Feed
            if not exhausted:
                try:
                    while not task_queue.full():
                        row = next(data_iter)
                        task_queue.put(row)
                        sent_count += 1
                except StopIteration:
                    exhausted = True
                    for _ in range(num_gpus):
                        task_queue.put(None)
            
            # Read
            try:
                while True:
                    res = result_queue.get_nowait()
                    
                    table = pa.Table.from_pylist([res], schema=schema)
                    pq_writer.write_table(table)
                    
                    received_count += 1
                    pbar.update(1)
            except queue.Empty:
                if exhausted and received_count >= sent_count:
                    break
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
    finally:
        if pq_writer: pq_writer.close()
        if pbar: pbar.close()
        for p in workers:
            if p.is_alive(): p.terminate()
            p.join()

    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
