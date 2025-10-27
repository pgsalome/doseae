import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from .clinical_metrics import ClinicalMetricsCalculator


class ClinicalLoss(nn.Module):
    """Loss function incorporating clinical metrics."""

    def __init__(self, config: dict):
        """Initialize clinical loss function."""
        super().__init__()
        self.config = config.get('loss_function', {})
        self.loss_type = self.config.get('type', 'mse')
        self.weights = self.config.get('weights', {})
        self.gamma_freq = self.config.get('gamma_calculation_frequency', 10)

        # Initialize clinical metrics calculator
        self.metrics_calculator = ClinicalMetricsCalculator(config)

        # MSE loss
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None, epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate loss with clinical metrics.

        Args:
            pred: Predicted dose distribution [B, C, D, H, W] or [B, C, H, W]
            target: Target dose distribution
            mask: Optional mask tensor
            epoch: Current epoch number (for epoch-based frequency calculation)

        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}

        # MSE loss (always calculated)
        if self.loss_type in ['mse', 'combined', 'clinical']:
            losses['mse'] = self.mse_loss(pred, target)

        # Clinical metrics (calculated based on epoch frequency)
        calculate_clinical = False
        if epoch is not None and self.loss_type in ['gamma', 'combined', 'clinical']:
            # epoch is 0-based, so calculate on epochs 0, 10, 20, 30...
            calculate_clinical = (epoch % self.gamma_freq == 0)

        if calculate_clinical:
            print(f"    Calculating gamma on epoch {epoch} (frequency: {self.gamma_freq})")
            # Convert to numpy for clinical calculations
            pred_np = pred[0, 0].detach().cpu().numpy()  # First sample in batch
            target_np = target[0, 0].detach().cpu().numpy()
            mask_np = mask[0].detach().cpu().numpy() if mask is not None else None

            # Calculate clinical metrics
            clinical_metrics = self.metrics_calculator.compare_dose_distributions(
                target_np, pred_np, mask_np
            )

            # Only gamma loss (1 - pass_rate) - DVH removed from loss calculation
            if 'gamma' in self.weights and self.weights['gamma'] > 0:
                gamma_pass_rate = clinical_metrics.get('gamma_pass_rate', 100.0)
                gamma_loss = 1.0 - gamma_pass_rate / 100.0
                losses['gamma'] = torch.tensor(gamma_loss, device=pred.device)
                print(f"    Gamma pass rate: {gamma_pass_rate:.2f}%, Gamma loss: {gamma_loss:.4f}")

        # Calculate total weighted loss (iterate over a copy to avoid dict-size change during iteration)
        total_loss = torch.tensor(0.0, device=pred.device)
        for loss_name, loss_value in list(losses.items()):
            weight = self.weights.get(loss_name, 1.0)
            weighted_loss = weight * loss_value
            losses[f'{loss_name}_weighted'] = weighted_loss
            total_loss += weighted_loss

        losses['total'] = total_loss

        return losses