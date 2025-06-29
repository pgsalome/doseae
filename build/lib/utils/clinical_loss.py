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

        # Counter for gamma calculation frequency
        self.batch_counter = 0

        # MSE loss
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate loss with clinical metrics.

        Args:
            pred: Predicted dose distribution [B, C, D, H, W] or [B, C, H, W]
            target: Target dose distribution
            mask: Optional mask tensor

        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}

        # MSE loss (always calculated)
        if self.loss_type in ['mse', 'combined', 'clinical']:
            losses['mse'] = self.mse_loss(pred, target)

        # Clinical metrics (calculated based on frequency)
        self.batch_counter += 1
        calculate_clinical = (self.batch_counter % self.gamma_freq == 0)

        if calculate_clinical and self.loss_type in ['gamma', 'dvh', 'combined', 'clinical']:
            # Convert to numpy for clinical calculations
            pred_np = pred[0, 0].detach().cpu().numpy()  # First sample in batch
            target_np = target[0, 0].detach().cpu().numpy()
            mask_np = mask[0].detach().cpu().numpy() if mask is not None else None

            # Calculate clinical metrics
            clinical_metrics = self.metrics_calculator.compare_dose_distributions(
                target_np, pred_np, mask_np
            )

            # Gamma loss (1 - pass_rate)
            if 'gamma' in self.weights and self.weights['gamma'] > 0:
                gamma_pass_rate = clinical_metrics.get('gamma_pass_rate', 100.0)
                losses['gamma'] = torch.tensor(1.0 - gamma_pass_rate / 100.0,
                                               device=pred.device)

            # DVH losses
            if 'dvh' in self.weights and self.weights['dvh'] > 0:
                dvh_loss = 0.0
                dvh_metrics = ['D95_diff', 'D50_diff', 'D2_diff']
                for metric in dvh_metrics:
                    if metric in clinical_metrics:
                        dvh_loss += clinical_metrics[metric]
                losses['dvh'] = torch.tensor(dvh_loss / len(dvh_metrics),
                                             device=pred.device)

            # Individual DVH metric losses
            for metric in ['d95', 'd50', 'd2']:
                if metric in self.weights and self.weights[metric] > 0:
                    metric_name = f'{metric.upper()}_diff'
                    if metric_name in clinical_metrics:
                        losses[metric] = torch.tensor(clinical_metrics[metric_name],
                                                      device=pred.device)

        # Calculate total weighted loss
        total_loss = torch.tensor(0.0, device=pred.device)
        for loss_name, loss_value in losses.items():
            weight = self.weights.get(loss_name, 1.0)
            weighted_loss = weight * loss_value
            losses[f'{loss_name}_weighted'] = weighted_loss
            total_loss += weighted_loss

        losses['total'] = total_loss

        return losses