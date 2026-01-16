import math
import torch
from einops import rearrange, repeat, einsum
import os
import torch.nn as nn
from typing import IO,BinaryIO,Callable,Dict,List,Optional,Tuple,Union
import numpy.typing as npt
import numpy as np


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and targets.

    Args:
        logits (torch.Tensor): The predicted logits of shape (batch_size, num_classes).
        targets (torch.Tensor): The ground truth labels of shape (batch_size,).
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    logits_i_reshaped = rearrange(logits, 'b c -> b 1 c')
    targets_i_reshaped = rearrange(targets, 'b -> b 1')
    
    logits_i_normalized = logits_i_reshaped - logits_i_reshaped.max(dim=-1, keepdim=True).values
    target_logits = logits_i_normalized.gather(dim=-1, index=targets_i_reshaped.unsqueeze(1)).squeeze(1);
    log_sum_exp = torch.logsumexp(logits_i_normalized, dim=-1)
    loss = -target_logits + log_sum_exp
    return loss.mean()

class SGDOptimizer:
    def __init__(self, parameters: List[torch.Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.data.zero_()

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters: List[torch.Tensor], lr: float, betas: Tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWOptimizer, self).__init__(parameters, defaults)
    
    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        return loss

def cosine_annealing_lr(initial_lr: float, final_lr: float, total_steps: int, current_step: int, warmup_steps: int, cycle_steps: int) -> float:
    if current_step < warmup_steps:
        lr = initial_lr * (current_step / warmup_steps)
    elif current_step < cycle_steps:
        progress = (current_step - warmup_steps) / (cycle_steps - warmup_steps)
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * progress))
    else:
        lr = final_lr
    return lr

def clip_gradients(parameters: List[torch.Tensor], max_norm: float = 1.0, eps: float = 1e-6) -> float:
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    return total_norm

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Get a batch of input and target sequences from the dataset.
    Args:
        dataset (npt.NDArray): The dataset from which to sample sequences.
        batch_size (int): The number of sequences in the batch.
        context_length (int): The length of each input sequence.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    '''
    num_samples = dataset.shape[0] - context_length
    start_indices = np.random.randint(0, num_samples, size=batch_size)
    
    inputs = []
    targets = []
    for idx in start_indices:
        input_seq = dataset[idx:idx + context_length]
        target_seq = dataset[idx + 1:idx + context_length + 1]
        inputs.append(input_seq)
        targets.append(target_seq)
    
    inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.long)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.long)
    
    return inputs_tensor, targets_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str | os.PathLike | BinaryIO | IO[bytes]):
    '''
    Save the model and optimizer state to a checkpoint file.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        filepath (str | os.PathLike | BinaryIO | IO[bytes]): The path to the checkpoint file or a file-like object.
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str | os.PathLike | BinaryIO | IO[bytes]) -> int:
    '''
    Load the model and optimizer state from a checkpoint file.
    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filepath (str | os.PathLike | BinaryIO | IO[bytes]): The path to the checkpoint file or a file-like object.
    Returns:
        int: The epoch number from which to resume training.
    '''
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def evaluate_model(model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device) -> float:
    '''
    Evaluate the model on the given dataset and compute the average loss.
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (npt.NDArray): The dataset for evaluation.
        batch_size (int): The batch size for evaluation.
        context_length (int): The context length for input sequences.
        device (torch.device): The device to run the evaluation on.
    Returns:
        float: The average loss over the dataset.
    '''
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        num_samples = dataset.shape[0] - context_length
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            inputs = []
            targets = []
            for idx in range(start_idx, end_idx):
                input_seq = dataset[idx:idx + context_length]
                target_seq = dataset[idx + 1:idx + context_length + 1]
                inputs.append(input_seq)
            logit = model(torch.tensor(np.array(inputs), dtype=torch.long).to(device))
            loss = cross_entropy(logit.view(-1, logit.size(-1)), torch.tensor(np.array(targets), dtype=torch.long).to(device).view(-1))
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / total_batches