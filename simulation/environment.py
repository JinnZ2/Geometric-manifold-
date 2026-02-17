"""
Synthetic simulation environment.
Provides a base model, a drifted fine-tuned model, and prompt sets.
No real transformer required - runs on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyLLM(nn.Module):
    """Minimal MLP standing in for a language model."""
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def model_fn(inputs: torch.Tensor, theta_flat: torch.Tensor,
             input_dim=32, hidden_dim=64, output_dim=16) -> torch.Tensor:
    """Functional forward pass using flattened parameters."""
    split1 = input_dim * hidden_dim
    split2 = split1 + hidden_dim
    split3 = split2 + hidden_dim * output_dim

    w1 = theta_flat[:split1].reshape(hidden_dim, input_dim)
    b1 = theta_flat[split1:split2]
    w2 = theta_flat[split2:split3].reshape(output_dim, hidden_dim)
    b2 = theta_flat[split3:]

    h = F.relu(inputs @ w1.T + b1)
    return h @ w2.T + b2


class Environment:
    def __init__(self, config: dict):
        self.drift_strength = config.get('drift_strength', 0.3)
        self.seed = config.get('seed', 42)
        self.input_dim = 32
        self.hidden_dim = 64
        self.output_dim = 16
        self.batch_size = 32

        torch.manual_seed(self.seed)
        self._setup()

    def _setup(self):
        model = ToyLLM(self.input_dim, self.hidden_dim, self.output_dim)
        params = [p.data.flatten() for p in model.parameters()]
        self.theta_ref = torch.cat(params)

        # Drifted model: reference + noise
        self.theta_drifted = self.theta_ref + \
            self.drift_strength * torch.randn_like(self.theta_ref)

        # Synthetic prompt sets
        self.safety_inputs = torch.randn(self.batch_size, self.input_dim)
        self.task_inputs = torch.randn(self.batch_size, self.input_dim)
        self.task_labels = torch.randint(0, self.output_dim, (self.batch_size,))

        # Feature space data for GMR
        self.features = torch.randn(200, self.input_dim)
        self.labels = torch.randint(0, 2, (200,))
        # Intentionally imbalanced: 80% majority, 20% minority
        self.labels[:160] = 0
        self.labels[160:] = 1

    def get_model_fn(self):
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        def fn(inputs, theta):
            return model_fn(inputs, theta, input_dim, hidden_dim, output_dim)
        return fn
