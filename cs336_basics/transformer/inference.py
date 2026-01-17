import argparse
import pickle
import torch
import torch.nn.functional as F
import os
import sys

# Add the parent directory to sys.path to allow imports from cs336_basics
# Assuming the script is run from .../cs336_basics/transformer/ or .../
# We want to add the folder containing cs336_basics to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is .../cs336_basics/transformer
# parent is .../cs336_basics
# grand_parent is .../ (the root of the repo)
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.bpe_tokenizer.tokenizer import BPETokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for TransformerLM")
    
    # Model configuration
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer (.pkl)")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling probability")
    
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda)")
    
    return parser.parse_args()

def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Apply top-p (nucleus) sampling to logits.
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # If temperature is 0, use argmax (effectively top-k=1 or just greedy)
        return torch.argmax(logits, dim=-1).item()

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create a mask for removing tokens with cumulative probability above the threshold
    # We want to keep the top-p mass.
    # Elements to remove are those where cumulative_probs > top_p, 
    # BUT we must keep the first element that crosses the threshold (so we shift right).
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensor to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    
    # Set logits of removed tokens to -inf
    logits[indices_to_remove] = float('-inf')
    
    # Re-normalize probabilities
    if temperature > 0:
        probs = F.softmax(logits, dim=-1)
        
    # Sample from the filtered distribution
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token.item()

def generate(model: TransformerLM, 
             tokenizer: BPETokenizer, 
             prompt: str, 
             max_tokens: int, 
             temperature: float = 1.0, 
             top_p: float = 0.9, 
             device: str = 'cpu') -> str:
    
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        print("Warning: Prompt encoded to empty sequence.")
        # Try to provide at least a BOS token if it exists, or just return empty
        return ""
        
    # Identify <|endoftext|> id if possible
    eot_id = None
    if tokenizer.special_tokens:
        try:
            # Try to encode the special token directly to find its ID
            # Assuming <|endoftext|> is in the special_tokens list
            if "<|endoftext|>" in tokenizer.special_tokens:
                 # We need to manually find the ID because encode might split it if logic differs
                 # But our tokenizer.encode handles special tokens logic
                 encoded_eot = tokenizer.encode("<|endoftext|>")
                 if len(encoded_eot) == 1:
                     eot_id = encoded_eot[0]
        except:
            pass

    print(f"Generating continuation for prompt id count: {len(input_ids)}")
    
    # Keep track of generated tokens
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_tokens):
            # Prepare current input: combine prompt + generated so far
            # Crop input to context length - 1 (to predict next)
            # Actually, the model takes context_len and outputs (batch, seq_len, vocab).
            # We just need the last logit.
            # Efficiently, we can just pass the last context_length tokens.
            
            full_sequence = input_ids + generated_ids
            input_tensor_seq = full_sequence[-model.tf_blocks[0].attention.rope.max_seq_len:]
            
            current_input = torch.tensor(input_tensor_seq, dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass
            logits = model(current_input)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            next_token = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)
            
            # Stop if we generated the end of text token
            if eot_id is not None and next_token == eot_id:
                break
                
            generated_ids.append(next_token)
            
    # Decode the full text (original prompt + generated)
    output_text = tokenizer.decode(input_ids + generated_ids)
    return output_text

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer_obj = pickle.load(f)
    
    # Handle both raw BPETokenizer object and dict wrapper
    if isinstance(tokenizer_obj, dict) and 'vocab' in tokenizer_obj and 'merges' in tokenizer_obj:
        tokenizer = BPETokenizer(tokenizer_obj['vocab'], tokenizer_obj['merges'], tokenizer_obj.get('special_tokens'))
    elif hasattr(tokenizer_obj, 'encode') and hasattr(tokenizer_obj, 'decode'):
        tokenizer = tokenizer_obj
    else:
        raise ValueError(f"Could not load valid tokenizer from {args.tokenizer_path}")

    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")

    # Initialize Model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    )

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle potential prefix issues (e.g., 'module.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    print("Model loaded successfully.")

    # Generate
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    main()
