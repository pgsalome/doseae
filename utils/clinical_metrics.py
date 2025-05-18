import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import warnings

# Try to import PyMedPhys
try:
    import pymedphys

    PYMEDPHYS_AVAILABLE = True
except ImportError:
    PYMEDPHYS_AVAILABLE = False
    warnings.warn("PyMedPhys not available, using custom gamma implementation")

# Try to import dicompyler-core for DVH
try:
    from dicompylercore import dvhcalc

    DICOMPYLER_AVAILABLE = True
except ImportError:
    DICOMPYLER_AVAILABLE = False
    warnings.warn("dicompyler-core not available, using custom DVH implementation")


class ClinicalMetricsCalculator:
    """Calculate clinical metrics for dose distributions."""

    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config.get('clinical_metrics', {})
        self.use_pymedphys = self.config.get('use_pymedphys_gamma', True) and PYMEDPHYS_AVAILABLE
        self.spacing = tuple(self.config.get('spacing', [2.0, 2.0, 2.0]))
        self.gamma_dose_threshold = self.config.get('gamma_dose_threshold', 3.0)
        self.gamma_distance_threshold = self.config.get('gamma_distance_threshold', 3.0)
        self.dose_threshold_cutoff = self.config.get('dose_threshold_cutoff', 10)
        self.dvh_num_bins = self.config.get('dvh_num_bins', 1000)

    def calculate_gamma(self, dose_ref: np.ndarray, dose_eval: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate gamma index analysis.

        Args:
            dose_ref: Reference dose distribution
            dose_eval: Evaluated dose distribution
            mask: Optional mask for ROI

        Returns:
            Dictionary with gamma metrics
        """
        if self.use_pymedphys:
            return self._calculate_gamma_pymedphys(dose_ref, dose_eval, mask)
        else:
            return self._calculate_gamma_custom(dose_ref, dose_eval, mask)

    def _calculate_gamma_pymedphys(self, dose_ref: np.ndarray, dose_eval: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate gamma using PyMedPhys."""
        # Set up coordinate system
        coords = (np.arange(dose_ref.shape[0]) * self.spacing[0],
                  np.arange(dose_ref.shape[1]) * self.spacing[1],
                  np.arange(dose_ref.shape[2]) * self.spacing[2])

        # Apply mask if provided
        if mask is not None:
            dose_ref = dose_ref.copy()
            dose_eval = dose_eval.copy()
            dose_ref[~mask] = 0
            dose_eval[~mask] = 0

        # Calculate gamma
        gamma = pymedphys.gamma(
            coords, dose_ref,
            coords, dose_eval,
            dose_percent_threshold=self.gamma_dose_threshold,
            distance_mm_threshold=self.gamma_distance_threshold,
            lower_percent_dose_cutoff=self.dose_threshold_cutoff,
            max_gamma=2.0
        )

        # Calculate metrics
        valid_gamma = gamma[~np.isnan(gamma)]
        if len(valid_gamma) == 0:
            return {'gamma_pass_rate': 0.0, 'gamma_mean': 0.0, 'gamma_max': 0.0}

        pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100
        mean_gamma = np.mean(valid_gamma)
        max_gamma = np.max(valid_gamma)

        return {
            'gamma_pass_rate': pass_rate,
            'gamma_mean': mean_gamma,
            'gamma_max': max_gamma,
            'gamma_map': gamma
        }

    def _calculate_gamma_custom(self, dose_ref: np.ndarray, dose_eval: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Simple custom gamma calculation."""
        # Normalize dose difference
        max_dose = dose_ref.max()
        dose_threshold = max_dose * self.gamma_dose_threshold / 100

        # Calculate dose difference
        dose_diff = np.abs(dose_ref - dose_eval)

        # Apply mask if provided
        if mask is not None:
            valid_mask = mask & (dose_ref > max_dose * self.dose_threshold_cutoff / 100)
        else:
            valid_mask = dose_ref > max_dose * self.dose_threshold_cutoff / 100

        # Simple gamma calculation (dose difference only, no DTA)
        gamma_values = dose_diff / dose_threshold
        gamma_values[~valid_mask] = np.nan

        # Calculate metrics
        valid_gamma = gamma_values[~np.isnan(gamma_values)]
        if len(valid_gamma) == 0:
            return {'gamma_pass_rate': 0.0, 'gamma_mean': 0.0, 'gamma_max': 0.0}

        pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100
        mean_gamma = np.mean(valid_gamma)
        max_gamma = np.max(valid_gamma)

        return {
            'gamma_pass_rate': pass_rate,
            'gamma_mean': mean_gamma,
            'gamma_max': max_gamma,
            'gamma_map': gamma_values
        }

    def calculate_dvh(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative DVH.

        Returns:
            Tuple of (dose_axis, cumulative_dvh)
        """
        if DICOMPYLER_AVAILABLE:
            return self._calculate_dvh_dicompyler(dose, mask)
        else:
            return self._calculate_dvh_custom(dose, mask)

    def _calculate_dvh_dicompyler(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Calculate DVH using dicompyler-core."""
        if mask is None:
            mask = np.ones_like(dose, dtype=bool)

        # Ensure dose is in Gy
        if dose.max() > 100:
            dose = dose / 100

        # Calculate DVH
        dvh_obj = dvhcalc.get_dvh(dose, mask, self.spacing)

        # Get cumulative DVH data
        return dvh_obj.dose_axis, dvh_obj.cumulative.data

    def _calculate_dvh_custom(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Custom DVH calculation."""
        # Apply mask
        if mask is not None:
            dose_values = dose[mask > 0]
        else:
            dose_values = dose.flatten()

        # Remove invalid values
        dose_values = dose_values[~np.isnan(dose_values)]
        dose_values = dose_values[dose_values >= 0]

        if len(dose_values) == 0:
            return np.array([0]), np.array([0])

        # Create histogram
        dose_max = dose_values.max()
        bins = np.linspace(0, dose_max * 1.1, self.dvh_num_bins)
        hist, _ = np.histogram(dose_values, bins=bins)

        # Calculate cumulative DVH
        cumulative = np.cumsum(hist[::-1])[::-1]
        cumulative = cumulative / len(dose_values) * 100

        # Use bin centers
        dose_axis = (bins[:-1] + bins[1:]) / 2

        return dose_axis, cumulative

    def calculate_dvh_metrics(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate standard DVH metrics."""
        dose_axis, dvh = self.calculate_dvh(dose, mask)

        if len(dose_axis) == 0:
            return {'D95': 0, 'D50': 0, 'D2': 0, 'Dmean': 0}

        # Find specific dose points
        d95 = self._find_dose_at_volume(dose_axis, dvh, 95)
        d50 = self._find_dose_at_volume(dose_axis, dvh, 50)
        d2 = self._find_dose_at_volume(dose_axis, dvh, 2)

        # Mean dose
        if mask is not None:
            dmean = np.mean(dose[mask > 0])
        else:
            dmean = np.mean(dose)

        return {
            'D95': d95,
            'D50': d50,
            'D2': d2,
            'Dmean': dmean
        }

    def _find_dose_at_volume(self, dose_axis: np.ndarray, dvh: np.ndarray,
                             volume_percent: float) -> float:
        """Find dose at specific volume percentage."""
        idx = np.argmin(np.abs(dvh - volume_percent))
        return dose_axis[idx]

    def compare_dose_distributions(self, dose_ref: np.ndarray, dose_eval: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive comparison of two dose distributions."""
        metrics = {}

        # Gamma analysis
        gamma_results = self.calculate_gamma(dose_ref, dose_eval, mask)
        metrics.update(gamma_results)

        # DVH metrics
        if self.config.get('calculate_dvh', True):
            dvh_ref = self.calculate_dvh_metrics(dose_ref, mask)
            dvh_eval = self.calculate_dvh_metrics(dose_eval, mask)

            # Calculate differences
            for key in dvh_ref:
                metrics[f'{key}_diff'] = abs(dvh_ref[key] - dvh_eval[key])
                if dvh_ref[key] > 0:
                    metrics[f'{key}_rel_diff'] = metrics[f'{key}_diff'] / dvh_ref[key] * 100

        return metrics