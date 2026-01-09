import torch
import torch.multiprocessing as mp
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import os
import queue
from tqdm import tqdm
import time
import re

# Force spawn for CUDA multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# -----------------------------------------------------------------------------
# 1. THE PROMPT (System Instruction)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert linguist specializing in intent classification.
Your task is to analyze the Reddit question and categorize the TYPE OF ANSWER it solicits.

### Categories
1. VALID (Subjective/Evaluative):
   - The user asks for an opinion, stance, mindset, theory, or judgment.
   - Requires accessing Semantic Memory (beliefs, general knowledge).
   - Examples: "What do you think of...?", "Should X be legalized?", "Is it wrong to...?"

2. INVALID (Narrative/Episodic):
   - The user asks for a specific story, past event, or personal experience.
   - Requires accessing Episodic Memory (replaying a specific past event).
   - Examples: "What was your worst date?", "Tell me about a time...", "What happened when...?"

3. INVALID (Factual/Other):
   - Factual questions, advice requests, or "Does anyone else" checks.

### Output Format
Output ONLY the word "Valid" or "Invalid". Do not provide reasoning. Do not output punctuation.
"""

# -----------------------------------------------------------------------------
# 2. WORKER PROCESS
# -----------------------------------------------------------------------------
def gpu_worker(gpu_id, task_queue, result_queue, model_name, max_new_tokens):
    """
    Worker that loads Qwen Instruct on a specific GPU.
    """
    try:
        # Assign specific GPU
        device_str = f"cuda:{gpu_id}"
        device = torch.device(device_str)
        print(f"[GPU {gpu_id}] Initializing {model_name}...")

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load Model
        # using float16 for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map={"": gpu_id} # strict mapping
        )
        model.eval()
        
        print(f"[GPU {gpu_id}] Ready.")

        while True:
            try:
                task = task_queue.get(timeout=2)
            except queue.Empty:
                continue

            if task is None:
                break

            row_id, question_text = task

            # --- PREPARE INPUT ---
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Question: "{question_text}"'}
            ]
            
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

            # --- GENERATE ---
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens, # Can be small (e.g. 16) since we only want one word
                    temperature=0.01, # Almost deterministic for classification
                    do_sample=False   # Greedy decoding is usually better for strict classification
                )

            # --- PARSE OUTPUT ---
            # Decode only the new tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # --- CLEANUP & VALIDATE ---
            # Remove punctuation and whitespace
            clean_label = re.sub(r'[^\w\s]', '', content).strip().lower()

            final_label = None
            
            # Robust checking incase model says "Valid." or "It is Invalid"
            if "invalid" in clean_label:
                final_label = "Invalid"
            elif "valid" in clean_label:
                final_label = "Valid"
            
            if final_label:
                result_queue.put((row_id, question_text, final_label))
            else:
                # If model hallucinates something else, ignore or log
                # print(f"[GPU {gpu_id}] Unclear output: {content}")
                pass

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}")
        # Signal main process might hang if we don't put something, 
        # but usually main loop handles timeout.

# -----------------------------------------------------------------------------
# 3. MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify Reddit Questions (Qwen Instruct)")
    parser.add_argument("input_file", help="Path to the .arrow file (submissions)")
    parser.add_argument("--output", default="classified_questions.parquet", help="Output file")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of rows to sample")
    
    args = parser.parse_args()
    
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    if not torch.cuda.is_available():
        print("Error: CUDA required.")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f"--- Configuration ---")
    print(f"GPUs: {num_gpus}")
    print(f"Model: {model_name}")
    print(f"Sampling: {args.sample_size} rows from {args.input_file}")

    # ---------------------------------------------------------
    # 1. LOAD & SAMPLE DATA
    # ---------------------------------------------------------
    print("Loading data...")
    try:
        lf = pl.scan_ipc(args.input_file)
    except:
        lf = pl.scan_arrow(args.input_file)

    # Filter out empty titles and sample
    df_sample = (
        lf.filter(pl.col("title").is_not_null())
          .select(["id", "title"])
          .collect()
          .sample(n=args.sample_size, seed=42) 
    )
    
    total_tasks = len(df_sample)
    print(f"Sampled {total_tasks} questions.")

    # ---------------------------------------------------------
    # 2. SETUP QUEUES
    # ---------------------------------------------------------
    task_queue = mp.Queue(maxsize=1000)
    result_queue = mp.Queue()
    
    workers = []
    
    # Spawn workers
    for i in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            # max_new_tokens=32 is enough for "Valid"/"Invalid"
            args=(i, task_queue, result_queue, model_name, 32) 
        )
        p.start()
        workers.append(p)

    # ---------------------------------------------------------
    # 3. PROCESS LOOP (With Safe Closing)
    # ---------------------------------------------------------
    
    # Parquet Schema
    schema = pa.schema([
        ('id', pa.string()),
        ('title', pa.string()),
        ('label', pa.string())
    ])
    
    pq_writer = None
    pbar = None

    try:
        pq_writer = pq.ParquetWriter(args.output, schema)
        pbar = tqdm(total=total_tasks, desc="Classifying")
        
        data_iter = df_sample.iter_rows(named=True)
        
        sent_count = 0
        received_count = 0
        data_exhausted = False
        
        while received_count < total_tasks:
            
            # A. Feed Inputs
            if not data_exhausted:
                try:
                    while not task_queue.full():
                        row = next(data_iter)
                        task_queue.put((row['id'], row['title']))
                        sent_count += 1
                except StopIteration:
                    data_exhausted = True
                    # Send poison pills to stop workers when they are done
                    for _ in range(num_gpus):
                        task_queue.put(None)

            # B. Collect Results
            try:
                # Non-blocking get
                while True:
                    res = result_queue.get_nowait()
                    
                    row_id, title, label = res
                    
                    # Write immediately
                    table = pa.Table.from_pylist([
                        {'id': row_id, 'title': title, 'label': label}
                    ], schema=schema)
                    pq_writer.write_table(table)
                    
                    received_count += 1
                    pbar.update(1)
            except queue.Empty:
                # If we are done, break
                if data_exhausted and received_count >= sent_count:
                    break
                # Wait a bit to prevent CPU spinning
                time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\n--- INTERRUPTED BY USER ---")
        print("Closing Parquet writer safely...")
        # The finally block will handle the actual closing
        
    except Exception as e:
        print(f"\n\n--- UNEXPECTED ERROR: {e} ---")
        
    finally:
        # CRITICAL: Always close the writer, otherwise file is corrupt
        if pq_writer:
            pq_writer.close()
            print(f"Parquet file closed: {args.output}")
        
        if pbar:
            pbar.close()

        print("Terminating workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()
            p.join()

if __name__ == "__main__":
    main()
