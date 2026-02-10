"""Perplexity evaluation for language models.

This module provides utilities to compute perplexity on standard datasets
using a sliding window approach for long contexts.
"""

from typing import Any

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    max_length: int = 2048,
    stride: int = 512,
    limit: int | None = None,
) -> float:
    """Evaluate perplexity on a text dataset using sliding window.

    Computes perplexity by processing the dataset with a sliding window approach,
    which allows evaluation on sequences longer than the model's context window.

    Args:
        model: HuggingFace causal language model
        tokenizer: HuggingFace tokenizer
        dataset_name: Name of dataset from HuggingFace datasets (default: "wikitext")
        dataset_config: Dataset configuration (default: "wikitext-103-raw-v1")
        split: Dataset split to use (default: "test")
        max_length: Maximum sequence length for model input (default: 2048)
        stride: Stride for sliding window (default: 512)
        limit: Limit number of examples to process (default: None, uses all)

    Returns:
        Perplexity score (lower is better)

    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> ppl = evaluate_perplexity(model, tokenizer, limit=100)
        >>> print(f"Perplexity: {ppl:.2f}")

    Notes:
        - Uses torch.no_grad() for memory efficiency
        - Implements strided sliding window to handle long sequences
        - Only computes loss on the non-overlapping stride tokens
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Apply limit if specified
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Concatenate all texts
    texts = dataset["text"]
    full_text = "\n\n".join(texts)

    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids

    # Move to model's device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Get sequence length
    seq_len = input_ids.size(1)

    # Initialize loss tracking
    nlls = []  # negative log-likelihoods
    loss_fct = CrossEntropyLoss(reduction="none")

    # Sliding window evaluation
    prev_end_loc = 0
    pbar = tqdm(range(0, seq_len, stride), desc="Evaluating perplexity")

    with torch.no_grad():
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Target length for this window

            # Extract window
            input_ids_window = input_ids[:, begin_loc:end_loc]

            # Forward pass
            outputs = model(input_ids_window, labels=input_ids_window)

            # Get logits (shift will be handled in loss computation)
            logits = outputs.logits

            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids_window[:, 1:].contiguous()

            # Compute loss only on non-overlapping part
            # We want to compute loss on the last `trg_len` tokens
            if trg_len > 0:
                # Take the last trg_len predictions
                target_logits = shift_logits[:, -trg_len:, :]
                target_labels = shift_labels[:, -trg_len:]

                # Flatten for loss computation
                target_logits_flat = target_logits.view(-1, target_logits.size(-1))
                target_labels_flat = target_labels.view(-1)

                # Compute token-level loss
                token_losses = loss_fct(target_logits_flat, target_labels_flat)

                # Store losses
                nlls.append(token_losses)

            prev_end_loc = end_loc

            # Update progress bar
            if len(nlls) > 0:
                current_nll = torch.cat(nlls).mean()
                current_ppl = torch.exp(current_nll).item()
                pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

            if end_loc == seq_len:
                break

    # Compute final perplexity
    if len(nlls) == 0:
        raise ValueError("No tokens processed for perplexity computation")

    nll = torch.cat(nlls).mean()
    perplexity = torch.exp(nll).item()

    return perplexity
