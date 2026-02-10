"""Evaluation functions for control models.

This module provides evaluation utilities for baseline control models including
logistic regression, CNN, and BERT-tiny.
"""

from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_logistic_regression(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """Evaluate logistic regression model on test data.

    Args:
        model: LogisticRegressionControl instance
        X_test: Test features (numpy array or similar)
        y_test: Test labels (numpy array or similar)

    Returns:
        Dictionary with 'accuracy' and 'f1' scores

    Examples:
        >>> from llm_radiation.models.control_models import LogisticRegressionControl
        >>> model = LogisticRegressionControl(input_dim=784, num_classes=10)
        >>> model.fit(X_train, y_train)
        >>> metrics = evaluate_logistic_regression(model, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)

    # For f1, use macro average for multiclass, binary for binary
    num_classes = len(set(y_test))
    if num_classes == 2:
        f1 = f1_score(y_test, y_pred, average="binary")
    else:
        f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def evaluate_cnn(
    model: torch.nn.Module,
    test_loader: DataLoader,
) -> dict[str, float]:
    """Evaluate CNN model on test data.

    Args:
        model: CNNControl instance or similar PyTorch CNN
        test_loader: DataLoader providing (images, labels) batches

    Returns:
        Dictionary with 'accuracy' and 'loss' scores

    Examples:
        >>> from llm_radiation.models.control_models import CNNControl
        >>> model = CNNControl(num_classes=10, input_channels=1)
        >>> metrics = evaluate_cnn(model, test_loader)
        >>> print(f"Test accuracy: {metrics['accuracy']:.3f}")
    """
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating CNN"):
            # Move to device and ensure float16
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss (convert to float32 for stable loss computation)
            loss = criterion(outputs.float(), labels)
            total_loss += loss.item() * images.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "loss": float(avg_loss),
    }


def evaluate_bert_tiny(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    batch_size: int = 32,
    max_length: int = 128,
) -> dict[str, float]:
    """Evaluate BERT-tiny model on a classification dataset.

    Args:
        model: BertTinyControl instance
        tokenizer: HuggingFace tokenizer
        dataset: Dataset with 'text' and 'label' fields (or similar)
        batch_size: Batch size for evaluation (default: 32)
        max_length: Maximum sequence length (default: 128)

    Returns:
        Dictionary with 'accuracy' score

    Examples:
        >>> from llm_radiation.models.control_models import BertTinyControl
        >>> model = BertTinyControl(num_labels=2)
        >>> metrics = evaluate_bert_tiny(model.get_model(), model.get_tokenizer(), test_dataset)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")

    Notes:
        - Expects dataset to have 'text' and 'label' fields
        - Handles tokenization internally
        - Uses model's device automatically
    """
    # Get underlying PyTorch model if it's a wrapper
    if hasattr(model, "get_model"):
        torch_model = model.get_model()
    else:
        torch_model = model

    device = next(torch_model.parameters()).device
    torch_model.eval()

    correct = 0
    total = 0

    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating BERT-tiny"):
        # Get batch
        batch_end = min(i + batch_size, len(dataset))
        batch = dataset[i:batch_end]

        # Handle both dict-like and list-like datasets
        if isinstance(batch, dict):
            texts = batch["text"] if "text" in batch else batch["sentence"]
            labels = batch["label"]
        else:
            # Assume list of examples
            texts = [ex["text"] if "text" in ex else ex["sentence"] for ex in batch]
            labels = [ex["label"] for ex in batch]

        # Tokenize
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        labels_tensor = torch.tensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            if hasattr(model, "forward"):
                outputs = model.forward(input_ids, attention_mask)
            else:
                outputs = torch_model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs.logits

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            total += labels_tensor.size(0)
            correct += (predicted == labels_tensor).sum().item()

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
    }
