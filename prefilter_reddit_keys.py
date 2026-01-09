import json
import argparse
import sys
import os
from tqdm import tqdm

def process_file(input_path, output_path, keys_to_keep, desc):
    """
    Reads a JSONL file line by line, keeps only selected keys, 
    and writes to a new file.
    """
    # Get file size for progress bar
    total_size = os.path.getsize(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for line in f_in:
                # Update progress bar with bytes read
                pbar.update(len(line))
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Create a new dictionary with only the keys we need
                # distinct check for existing keys to avoid inserting 'None'
                new_data = {k: data[k] for k in keys_to_keep if k in data}

                # Write back as valid JSONL
                f_out.write(json.dumps(new_data) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Filter Reddit JSONL files to keep only essential keys.")
    parser.add_argument("submissions_file", help="Input submissions .json/.jsonl file")
    parser.add_argument("comments_file", help="Input comments .json/.jsonl file")
    parser.add_argument("--suffix", default="_filtered", help="Suffix for output files (e.g., file.json -> file_filtered.json)")
    
    args = parser.parse_args()

    # Define keys
    sub_keys = {"id", "title", "author", "created_utc", "over_18", "score", "num_comments"}
    com_keys = {"id", "body", "ups", "downs", "created_utc", "author", "parent_id", "link_id"}

    # Determine output filenames
    sub_base, sub_ext = os.path.splitext(args.submissions_file)
    com_base, com_ext = os.path.splitext(args.comments_file)
    
    sub_out = f"{sub_base}{args.suffix}{sub_ext}"
    com_out = f"{com_base}{args.suffix}{com_ext}"

    print(f"--- Configuration ---")
    print(f"Submissions Input:  {args.submissions_file}")
    print(f"Submissions Output: {sub_out}")
    print(f"Keys: {sub_keys}")
    print(f"\nComments Input:     {args.comments_file}")
    print(f"Comments Output:    {com_out}")
    print(f"Keys: {com_keys}")
    print(f"---------------------")

    # Process Submissions
    process_file(args.submissions_file, sub_out, sub_keys, "Filtering Submissions")

    # Process Comments
    process_file(args.comments_file, com_out, com_keys, "Filtering Comments")

    print("\nDone! Now run the arrow converter on the new files.")

if __name__ == "__main__":
    main()
