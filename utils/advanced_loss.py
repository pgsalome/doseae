import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class AdvancedDoseLoss(nn.Module):
    """
    Advanced dose-aware loss function combining multiple clinically relevant terms.
    Based on ChatGPT's recommendation for dose distribution reconstruction.
    """

    def __init__(self, config: dict):
        """Initialize advanced dose loss function."""
        super().__init__()
        self.config = config.get('loss_function', {})
        
        # Loss weights (from ChatGPT recommendation)
        self.weights = {
            'dose_weighted_l1': 1.0,
            'gradient': 0.25,
            'frequency': 0.05,
            'ncc': 0.1
        }
        
        # Update with config weights if provided
        config_weights = self.config.get('weights', {})
        self.weights.update(config_weights)
        
        # Dose weighting parameters
        self.alpha = float(self.config.get('dose_alpha', 2.0))  # Dose weighting exponent
        self.epsilon = float(self.config.get('dose_epsilon', 1e-6))  # Small constant for numerical stability
        
        # Debug: Print types to ensure they're correct (commented out for MSE baseline)
        # print(f"AdvancedDoseLoss init: alpha={self.alpha} (type: {type(self.alpha)}), epsilon={self.epsilon} (type: {type(self.epsilon)})")
        
        # Gradient loss parameters
        self.gradient_mode = self.config.get('gradient_mode', '3d')  # '3d' or '2d'
        
        # Frequency loss parameters
        self.freq_weighting = self.config.get('freq_weighting', True)  # Apply frequency weighting
        
        # NCC parameters
        self.ncc_window_size = self.config.get('ncc_window_size', 9)  # Local NCC window size

    def dose_weighted_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Dose-weighted L1 loss that emphasizes high-dose regions.
        
        Args:
            pred: Predicted dose [B, C, D, H, W]
            target: Target dose [B, C, D, H, W]
            
        Returns:
            Dose-weighted L1 loss
        """
        # Normalize target to [0, 1] per batch
        target_max = target.max()
        if target_max > 0:
            target_norm = target / (target_max + self.epsilon)
        else:
            target_norm = target  # Avoid division by zero
        
        # Calculate dose weights: w_i = (d_i^* + ε)^α
        # Debug: Check types before operation
        if not isinstance(self.epsilon, (int, float)):
            print(f"ERROR: epsilon is not numeric: {self.epsilon} (type: {type(self.epsilon)})")
            self.epsilon = 1e-6
        if not isinstance(self.alpha, (int, float)):
            print(f"ERROR: alpha is not numeric: {self.alpha} (type: {type(self.alpha)})")
            self.alpha = 2.0
            
        dose_weights = torch.pow(target_norm + self.epsilon, self.alpha)
        
        # Calculate L1 loss with weights
        l1_diff = torch.abs(pred - target)
        weighted_l1 = dose_weights * l1_diff
        
        return weighted_l1.mean()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        3D gradient loss for edge preservation.
        
        Args:
            pred: Predicted dose [B, C, D, H, W]
            target: Target dose [B, C, D, H, W]
            
        Returns:
            Gradient loss
        """
        if self.gradient_mode == '3d':
            # 3D finite differences
            pred_grad_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
            pred_grad_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
            pred_grad_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
            
            target_grad_x = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
            target_grad_z = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
            
            grad_loss = (F.l1_loss(pred_grad_x, target_grad_x) + 
                        F.l1_loss(pred_grad_y, target_grad_y) + 
                        F.l1_loss(pred_grad_z, target_grad_z)) / 3.0
        else:
            # 2D gradient (for 2D models)
            pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            
            grad_loss = (F.l1_loss(pred_grad_x, target_grad_x) + 
                        F.l1_loss(pred_grad_y, target_grad_y)) / 2.0
        
        return grad_loss

    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Frequency domain loss using FFT magnitude.
        
        Args:
            pred: Predicted dose [B, C, D, H, W]
            target: Target dose [B, C, D, H, W]
            
        Returns:
            Frequency loss
        """
        try:
            # Convert to frequency domain
            pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
            target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
            
            # Get magnitude spectra
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            
            # Apply frequency weighting if enabled
            if self.freq_weighting:
                # Create frequency weighting (emphasize mid-high frequencies)
                freq_weights = self._create_frequency_weights(pred.shape[-3:], pred.device)
                pred_mag = pred_mag * freq_weights
                target_mag = target_mag * freq_weights
            
            # L1 loss on magnitude spectra
            freq_loss = F.l1_loss(pred_mag, target_mag)
            
            return freq_loss
        except Exception as e:
            print(f"Error in frequency_loss calculation: {e}")
            # Fallback to simple L1 loss
            return F.l1_loss(pred, target)

    def _create_frequency_weights(self, shape, device):
        """Create frequency weighting mask."""
        D, H, W = shape
        center_d, center_h, center_w = D // 2, H // 2, W // 2
        
        # Create distance from center
        d_coords = torch.arange(D, device=device).float() - center_d
        h_coords = torch.arange(H, device=device).float() - center_h
        w_coords = torch.arange(W, device=device).float() - center_w
        
        # Create 3D meshgrid
        dd, hh, ww = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Calculate distance from center
        distance = torch.sqrt(dd**2 + hh**2 + ww**2)
        max_distance = torch.sqrt(torch.tensor(center_d**2 + center_h**2 + center_w**2, dtype=torch.float32, device=device))
        
        # Normalize distance
        normalized_distance = distance / max_distance
        
        # Create weighting (emphasize mid-high frequencies)
        weights = 1.0 + 2.0 * normalized_distance  # Linear increase from 1 to 3
        
        return weights

    def ncc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Normalized Cross-Correlation loss.
        
        Args:
            pred: Predicted dose [B, C, D, H, W]
            target: Target dose [B, C, D, H, W]
            
        Returns:
            NCC loss (1 - NCC)
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        # Calculate means
        pred_mean = pred_flat.mean(dim=-1, keepdim=True)
        target_mean = target_flat.mean(dim=-1, keepdim=True)
        
        # Center the data
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        # Calculate NCC
        numerator = (pred_centered * target_centered).sum(dim=-1)
        pred_var = (pred_centered ** 2).sum(dim=-1)
        target_var = (target_centered ** 2).sum(dim=-1)
        # Debug: Check epsilon type in NCC
        if not isinstance(self.epsilon, (int, float)):
            print(f"ERROR in NCC: epsilon is not numeric: {self.epsilon} (type: {type(self.epsilon)})")
            self.epsilon = 1e-6
            
        denominator = torch.sqrt(pred_var * target_var + self.epsilon)
        
        ncc = numerator / denominator
        ncc_loss = 1.0 - ncc.mean()
        
        return ncc_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate advanced dose-aware loss.
        
        Args:
            pred: Predicted dose distribution [B, C, D, H, W]
            target: Target dose distribution
            mask: Optional mask tensor (not used in this implementation)
            
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        try:
            # Dose-weighted L1 loss
            if self.weights['dose_weighted_l1'] > 0:
                try:
                    losses['dose_weighted_l1'] = self.dose_weighted_l1_loss(pred, target)
                except Exception as e:
                    print(f"Error in dose_weighted_l1_loss: {e}")
                    losses['dose_weighted_l1'] = torch.nn.functional.l1_loss(pred, target)
            
            # Gradient loss
            if self.weights['gradient'] > 0:
                try:
                    losses['gradient'] = self.gradient_loss(pred, target)
                except Exception as e:
                    print(f"Error in gradient_loss: {e}")
                    losses['gradient'] = torch.tensor(0.0, device=pred.device)
            
            # Frequency loss
            if self.weights['frequency'] > 0:
                try:
                    losses['frequency'] = self.frequency_loss(pred, target)
                except Exception as e:
                    print(f"Error in frequency_loss: {e}")
                    losses['frequency'] = torch.tensor(0.0, device=pred.device)
            
            # NCC loss
            if self.weights['ncc'] > 0:
                try:
                    losses['ncc'] = self.ncc_loss(pred, target)
                except Exception as e:
                    print(f"Error in ncc_loss: {e}")
                    losses['ncc'] = torch.tensor(0.0, device=pred.device)
            
            # Calculate total weighted loss
            total_loss = torch.tensor(0.0, device=pred.device)
            for loss_name, loss_value in list(losses.items()):
                weight = self.weights.get(loss_name, 1.0)
                weighted_loss = weight * loss_value
                losses[f'{loss_name}_weighted'] = weighted_loss
                total_loss += weighted_loss
            
            losses['total'] = total_loss
            
        except Exception as e:
            print(f"Error in AdvancedDoseLoss: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple MSE loss
            losses['mse'] = torch.nn.functional.mse_loss(pred, target)
            losses['total'] = losses['mse']
        
        return losses
