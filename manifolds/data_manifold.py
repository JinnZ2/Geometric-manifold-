"""
Data Manifold Layer - GMR-style geometric rectification in feature space.

Asymmetric cleaning:
- Aggressively remove majority class intrusions into minority regions
- Conservatively preserve rare/safe samples
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist


class DataManifold:
    def __init__(self, config: dict):
        self.gamma_majority = config.get('confidence_threshold_majority', 0.7)
        self.gamma_minority = config.get('confidence_threshold_minority', 0.3)
        self.alpha = config.get('alpha', 0.5)   # majority weight
        self.beta = config.get('beta', 1.0)     # minority weight
        self.beta_prime = config.get('beta_prime', 0.1)  # low-confidence minority weight

    def geometric_confidence(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Estimate local geometric confidence for each sample.
        Uses k-NN density ratio as a proxy for manifold curvature confidence.
        """
        k = min(10, len(features) - 1)
        features_np = features.detach().numpy()
        dists = cdist(features_np, features_np, metric='euclidean')
        np.fill_diagonal(dists, np.inf)

        knn_indices = np.argsort(dists, axis=1)[:, :k]
        knn_labels = labels.numpy()[knn_indices]

        # Confidence = fraction of k-NN with same label
        same_label = (knn_labels == labels.numpy()[:, None]).mean(axis=1)
        return torch.tensor(same_label, dtype=torch.float32)

    def asymmetric_weights(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample weights using asymmetric cleaning logic.
        Majority class (label=0): aggressive removal of low-confidence samples
        Minority class (label=1): conservative - preserve even low-confidence samples
        """
        confidence = self.geometric_confidence(features, labels)
        weights = torch.zeros(len(labels))

        majority_mask = labels == 0
        minority_mask = labels == 1

        # Majority: only keep high-confidence samples
        weights[majority_mask & (confidence >= self.gamma_majority)] = self.alpha
        # Majority low-confidence: drop (weight=0)

        # Minority: keep high-confidence with full weight, low-confidence with small weight
        weights[minority_mask & (confidence >= self.gamma_minority)] = self.beta
        weights[minority_mask & (confidence < self.gamma_minority)] = self.beta_prime

        return weights

    def rectify(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Apply rectification: return cleaned features, labels, and weights.
        """
        weights = self.asymmetric_weights(features, labels)
        keep_mask = weights > 0
        return features[keep_mask], labels[keep_mask], weights[keep_mask]
