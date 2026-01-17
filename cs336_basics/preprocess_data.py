import argparse
import os
import numpy as np
import pickle
import tqdm
from cs336_basics.bpe_tokenizer.train_bpe import train_bpe
from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess data for Transformer")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw text files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save binary files")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--sample_size", type=int, default=50 * 1024 * 1024, help="Size of data subset for BPE training (bytes)")
    return parser.parse_args()

def save_tokenizer(vocab, merges, path):
    with open(path, 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['vocab'], data['merges']

def process_file(tokenizer, input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    
    # Check file size for progress bar
    file_size = os.path.getsize(input_path)
    total_tokens = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'wb') as fout:
        # Read in chunks
        # Use line-based reading to avoid splitting tokens
        buffer = []
        buffer_size = 0
        target_chunk_size = 10 * 1024 * 1024 # 10MB chunks
        
        pbar = tqdm.tqdm(total=file_size, unit='B', unit_scale=True)
        
        for line in fin:
            buffer.append(line)
            buffer_size += len(line.encode('utf-8'))
            
            if buffer_size >= target_chunk_size:
                text_chunk = "".join(buffer)
                pbar.update(len(text_chunk.encode('utf-8')))
                
                # Encode
                ids = tokenizer.encode(text_chunk)
                
                # Write to file
                ids_np = np.array(ids, dtype=np.uint16)
                fout.write(ids_np.tobytes())
                total_tokens += len(ids)
                
                buffer = []
                buffer_size = 0
        
        if buffer:
            text_chunk = "".join(buffer)
            pbar.update(len(text_chunk.encode('utf-8')))
            ids = tokenizer.encode(text_chunk)
            
            # Write to file
            ids_np = np.array(ids, dtype=np.uint16)
            fout.write(ids_np.tobytes())
            total_tokens += len(ids)
            
        pbar.close()

    print(f"Saved {total_tokens} tokens to {output_path}...")
    print("Done.")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.input_dir, "TinyStoriesV2-GPT4-train.txt")
    val_path = os.path.join(args.input_dir, "TinyStoriesV2-GPT4-valid.txt")
    
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pkl")
    
    # 1. Train Tokenizer (or load if exists)
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        vocab, merges = load_tokenizer(tokenizer_path)
    else:
        print("Training BPE tokenizer...")
        # Create a sample file
        sample_path = os.path.join(args.output_dir, "train_sample.txt")
        with open(train_path, 'r', encoding='utf-8') as fin, open(sample_path, 'w', encoding='utf-8') as fout:
            fout.write(fin.read(args.sample_size))
            
        special_tokens = ["<|endoftext|>"]
        vocab, merges = train_bpe(sample_path, args.vocab_size, special_tokens)
        
        # Save tokenizer
        save_tokenizer(vocab, merges, tokenizer_path)
        
        # Cleanup sample
        os.remove(sample_path)
        print("Tokenizer trained and saved.")

    # 2. Initialize Tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    
    # 3. Process Train Data
    process_file(tokenizer, train_path, os.path.join(args.output_dir, "train.bin"))
    
    # 4. Process Val Data
    process_file(tokenizer, val_path, os.path.join(args.output_dir, "val.bin"))

if __name__ == "__main__":
    main()
