import numpy as np
import torch

class NormalizedMSELoss(torch.nn.Module):
    def __init__(self, feature_variance: list, lat_lons: list, device="cpu"):
        """
        Normalized MSE Loss as described in the paper

        Args:
            feature_variance: Variance for each of the physical features
            lat_lons: List of lat/lon pairs, used to generate weighting
            device: Device to compute the loss (default is CPU)
        """
        super().__init__()
        self.feature_variance = torch.tensor(feature_variance)
        assert not torch.isnan(self.feature_variance).any()

        # Compute weights based on cos(latitude)
        weights = []
        for lat, lon in lat_lons:
            weights.append(np.cos(lat * np.pi / 180.0))
        self.weights = torch.tensor(weights, dtype=torch.float)
        assert not torch.isnan(self.weights).any()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the loss

        Args:
            pred: Prediction tensor
            target: Target tensor

        Returns:
            MSE loss on the variance-normalized values weighted by latitude
        """
        self.feature_variance = self.feature_variance.to(pred.device)
        self.weights = self.weights.to(pred.device)

        out = (pred - target) ** 2
        assert not torch.isnan(out).any()

        out = out.mean(-1)  # Mean of the physical variables
        out = out * self.weights.expand_as(out)  # Weight by latitude

        assert not torch.isnan(out).any()
        return out.mean()
