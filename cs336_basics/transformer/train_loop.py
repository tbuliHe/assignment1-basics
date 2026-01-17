import argparse
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import wandb

from cs336_basics.transformer.train_utils import (
    AdamWOptimizer,
    clip_gradients,
    cosine_annealing_lr,
    cross_entropy,
    get_batch,
    save_checkpoint,
)
from cs336_basics.transformer.transformer import TransformerLM


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Data params
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data .bin file")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data .bin file")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    
    # Model params
    parser.add_argument("--context-length", type=int, default=256, help="Context length (sequence length)")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta")
    
    # Training params
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Min learning rate")
    parser.add_argument("--max-iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
    
    # Logging & Eval
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval-interval", type=int, default=200, help="Evaluation interval")
    parser.add_argument("--eval-iters", type=int, default=50, help="Number of iterations for evaluation")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--wandb-project", type=str, default="cs336-basics", help="WandB project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length)
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            # Flatten for cross_entropy
            # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch*seq_len)
            B, T, C = logits.shape
            loss = cross_entropy(logits.view(-1, C), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args):
    # Setup WandB
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=args)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data using memory mapping
    # Assuming data is stored as uint16 (standard for vocab size < 65536)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    print(f"Loaded train data: {len(train_data)} tokens")
    print(f"Loaded val data: {len(val_data)} tokens")
    
    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Initialize optimizer
    # train_utils.AdamWOptimizer expects a list of parameters
    optimizer = AdamWOptimizer(
        list(model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Training loop
    t0 = time.time()
    for iter in range(args.max_iters):
        # Calculate current learning rate
        lr = cosine_annealing_lr(
            initial_lr=args.lr,
            final_lr=args.min_lr,
            total_steps=args.max_iters,
            current_step=iter,
            warmup_steps=args.warmup_steps,
            cycle_steps=args.max_iters
        )
        
        # Update optimizer learning rate
        # Although AdamWOptimizer in train_utils takes lr in init, 
        # it doesn't seem to expose a way to update it easily per group via the simple class.
        # But looking at AdamWOptimizer in train_utils.py:
        # It inherits from torch.optim.Optimizer. 
        # So we can update param_groups.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation
        if iter % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data, 
                args.batch_size, args.context_length, args.eval_iters, device
            )
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if not args.no_wandb:
                wandb.log({
                    "iter": iter,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
        
        # Save checkpoint
        if iter > 0 and iter % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iter}.pt")
            save_checkpoint(model, optimizer, iter, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Training step
        X, Y = get_batch(train_data, args.batch_size, args.context_length)
        X, Y = X.to(device), Y.to(device)
        
        # Forward
        logits = model(X)
        
        # Loss
        B, T, C = logits.shape
        loss = cross_entropy(logits.view(-1, C), Y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        clip_gradients(list(model.parameters()), max_norm=args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iter % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if not args.no_wandb:
                wandb.log({"train/batch_loss": loss.item()})
            print(f"step {iter}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")

    # Final save
    final_path = os.path.join(args.output_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print("Training complete.")


if __name__ == "__main__":
    args = get_args()
    train(args)
