"""Control models for bit-flip robustness experiments.

This module provides simpler baseline models (logistic regression, CNN, BERT-tiny)
for comparing bit-flip robustness against larger LLMs. All models use float16
to enable consistent bit-flip injection experiments.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogisticRegressionControl:
    """Logistic regression model with float16 weights for bit-flip experiments.

    Wraps scikit-learn's LogisticRegression but stores weights as torch float16
    tensors to enable bit-flip injection using the same tools as neural networks.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """Initialize logistic regression model.

        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sklearn_model = SKLearnLogisticRegression(
            max_iter=1000, random_state=42
        )

        # Initialize weight tensor (will be populated during fit)
        self.weight_tensor: torch.Tensor | None = None
        self.bias_tensor: torch.Tensor | None = None

    def fit(self, X: Any, y: Any) -> "LogisticRegressionControl":
        """Fit the logistic regression model.

        Args:
            X: Training features (numpy array or similar)
            y: Training labels (numpy array or similar)

        Returns:
            Self for method chaining
        """
        # Fit sklearn model
        self.sklearn_model.fit(X, y)

        # Convert weights to float16 torch tensors
        # coef_ shape: (num_classes, input_dim) for multiclass
        # For binary classification, sklearn returns (1, input_dim)
        coef = self.sklearn_model.coef_
        if coef.shape[0] == 1 and self.num_classes == 2:
            # Binary case - expand to 2 classes
            coef = torch.tensor(coef, dtype=torch.float16)
        else:
            coef = torch.tensor(coef, dtype=torch.float16)

        self.weight_tensor = coef
        self.bias_tensor = torch.tensor(
            self.sklearn_model.intercept_, dtype=torch.float16
        )

        return self

    def predict(self, X: Any) -> Any:
        """Predict class labels using current weights.

        If bit flips have been applied to weight_tensor, this will use the
        corrupted weights for prediction.

        Args:
            X: Input features (numpy array or similar)

        Returns:
            Predicted class labels
        """
        if self.weight_tensor is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Convert input to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Compute logits: X @ W.T + b
        logits = F.linear(
            X_tensor, self.weight_tensor.float(), self.bias_tensor.float()
        )

        # Get class predictions
        if self.num_classes == 2 and self.weight_tensor.shape[0] == 1:
            # Binary classification
            predictions = (logits.squeeze() > 0).long().numpy()
        else:
            # Multiclass
            predictions = torch.argmax(logits, dim=1).numpy()

        return predictions

    def get_weight_tensor(self) -> torch.Tensor:
        """Get the float16 weight tensor for bit-flip injection.

        Returns:
            Float16 tensor containing model weights

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.weight_tensor is None:
            raise RuntimeError("Model must be fitted before accessing weights")
        return self.weight_tensor

    def load_from_tensor(self, weight_tensor: torch.Tensor, bias_tensor: torch.Tensor) -> None:
        """Load weights from external tensors (e.g., after bit flips).

        Args:
            weight_tensor: Weight tensor to load
            bias_tensor: Bias tensor to load
        """
        self.weight_tensor = weight_tensor.to(dtype=torch.float16)
        self.bias_tensor = bias_tensor.to(dtype=torch.float16)


class CNNControl(nn.Module):
    """Simple CNN for MNIST/CIFAR-10 classification in float16.

    Architecture:
        - Conv1: 3/1 -> 32 channels, 3x3 kernel
        - Conv2: 32 -> 64 channels, 3x3 kernel
        - FC1: flattened -> 128 features
        - FC2: 128 -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        image_size: int = 28,
    ):
        """Initialize CNN model.

        Args:
            num_classes: Number of output classes (default: 10 for MNIST/CIFAR-10)
            input_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            image_size: Input image size (28 for MNIST, 32 for CIFAR-10)
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Calculate flattened size after conv and pooling
        # After 2 max pools with kernel=2: image_size -> image_size // 4
        flattened_size = 64 * (image_size // 4) * (image_size // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Convert all parameters to float16
        self.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Ensure input is float16
        x = x.half()

        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BertTinyControl:
    """BERT-tiny model wrapper for sequence classification in float16.

    Uses HuggingFace's prajjwal1/bert-tiny, a distilled BERT model with only
    2 layers and 128 hidden dimensions, suitable for fast experiments.
    """

    def __init__(self, num_labels: int = 2, model_name: str = "prajjwal1/bert-tiny"):
        """Initialize BERT-tiny model.

        Args:
            num_labels: Number of classification labels (default: 2 for binary)
            model_name: HuggingFace model name (default: prajjwal1/bert-tiny)
        """
        self.num_labels = num_labels
        self.model_name = model_name

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through BERT-tiny.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)

        Returns:
            Logits of shape (batch_size, num_labels)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model for bit-flip injection.

        Returns:
            The HuggingFace model
        """
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer.

        Returns:
            The HuggingFace tokenizer
        """
        return self.tokenizer


def load_control_model(name: str, **kwargs: Any) -> Any:
    """Factory function to load a control model by name.

    Args:
        name: Model name - one of "logistic", "cnn", "bert-tiny"
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instantiated control model

    Raises:
        ValueError: If model name is not recognized

    Examples:
        >>> model = load_control_model("logistic", input_dim=784, num_classes=10)
        >>> model = load_control_model("cnn", num_classes=10, input_channels=1)
        >>> model = load_control_model("bert-tiny", num_labels=2)
    """
    if name == "logistic":
        if "input_dim" not in kwargs or "num_classes" not in kwargs:
            raise ValueError(
                "LogisticRegressionControl requires 'input_dim' and 'num_classes'"
            )
        return LogisticRegressionControl(**kwargs)

    elif name == "cnn":
        return CNNControl(**kwargs)

    elif name == "bert-tiny":
        return BertTinyControl(**kwargs)

    else:
        raise ValueError(
            f"Unknown control model: {name}. Choose from: logistic, cnn, bert-tiny"
        )
