import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, some metrics won't be calculated")


def mse_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Mean squared error
    """
    if isinstance(y_true, torch.Tensor):
        return F.mse_loss(y_pred, y_true).item()
    else:
        return mean_squared_error(y_true.flatten(), y_pred.flatten())


def mae_loss(y_true, y_pred):
    """
    Calculate Mean Absolute Error.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Mean absolute error
    """
    if isinstance(y_true, torch.Tensor):
        return F.l1_loss(y_pred, y_true).item()
    else:
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())


def rmse_loss(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Root mean squared error
    """
    return np.sqrt(mse_loss(y_true, y_pred))


def nrmse_loss(y_true, y_pred):
    """
    Calculate Normalized Root Mean Squared Error.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Normalized root mean squared error
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
        y_range = y_true_np.max() - y_true_np.min()
    else:
        y_range = y_true.max() - y_true.min()

    return rmse_loss(y_true, y_pred) / y_range if y_range > 0 else float('inf')


def psnr_metric(y_true, y_pred):
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Peak signal-to-noise ratio
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_true_np = y_true
        y_pred_np = y_pred

    if SKIMAGE_AVAILABLE:
        try:
            data_range = y_true_np.max() - y_true_np.min()
            return peak_signal_noise_ratio(y_true_np, y_pred_np, data_range=data_range)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            # Fallback to manual calculation
            pass

    # Manual calculation
    mse = mse_loss(y_true, y_pred)
    if mse == 0:
        return float('inf')

    data_range = y_true_np.max() - y_true_np.min()
    return 20 * np.log10(data_range / np.sqrt(mse))


def ssim_metric(y_true, y_pred):
    """
    Calculate Structural Similarity Index.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values

    Returns:
        float: Structural similarity index or None if skimage is not available
    """
    if not SKIMAGE_AVAILABLE:
        return None

    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_true_np = y_true
        y_pred_np = y_pred

    try:
        data_range = y_true_np.max() - y_true_np.min()

        # Check dimensions
        if y_true_np.ndim == 2:
            # 2D image
            return structural_similarity(y_true_np, y_pred_np, data_range=data_range)
        elif y_true_np.ndim == 3:
            # 3D volume, calculate for each slice and average
            ssim_values = []
            for i in range(y_true_np.shape[0]):
                ssim_values.append(
                    structural_similarity(y_true_np[i], y_pred_np[i], data_range=data_range)
                )
            return np.mean(ssim_values)
        else:
            print(f"Warning: SSIM not implemented for {y_true_np.ndim}D data")
            return None
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return None


def dice_coefficient(y_true, y_pred, threshold=0.5):
    """
    Calculate Dice coefficient for segmentation comparison.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth segmentation
        y_pred (numpy.ndarray or torch.Tensor): Predicted segmentation
        threshold (float): Threshold for binarizing predictions

    Returns:
        float: Dice coefficient
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_true_np = y_true
        y_pred_np = y_pred

    # Binarize predictions
    y_pred_binary = (y_pred_np > threshold).astype(np.float32)
    y_true_binary = (y_true_np > threshold).astype(np.float32)

    # Calculate intersection and union
    intersection = np.sum(y_true_binary * y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary)

    # Calculate Dice coefficient
    if union == 0:
        return 1.0  # Both are empty, consider it a perfect match
    else:
        return 2.0 * intersection / union


def dvh_metrics(dose_true, dose_pred, roi_mask=None, dose_bins=100):
    """
    Calculate metrics based on Dose-Volume Histogram (DVH).

    Args:
        dose_true (numpy.ndarray): Ground truth dose distribution
        dose_pred (numpy.ndarray): Predicted dose distribution
        roi_mask (numpy.ndarray, optional): Region of interest mask
        dose_bins (int): Number of bins for the histogram

    Returns:
        dict: Dictionary of DVH metrics
    """
    if isinstance(dose_true, torch.Tensor):
        dose_true = dose_true.detach().cpu().numpy()
    if isinstance(dose_pred, torch.Tensor):
        dose_pred = dose_pred.detach().cpu().numpy()
    if isinstance(roi_mask, torch.Tensor):
        roi_mask = roi_mask.detach().cpu().numpy()

    # If no ROI mask is provided, use the entire volume
    if roi_mask is None:
        roi_mask = np.ones_like(dose_true)

    # Apply the ROI mask
    dose_true_roi = dose_true * roi_mask
    dose_pred_roi = dose_pred * roi_mask

    # Get non-zero values
    dose_true_values = dose_true_roi[roi_mask > 0]
    dose_pred_values = dose_pred_roi[roi_mask > 0]

    if len(dose_true_values) == 0:
        return {
            'dvh_diff_area': float('nan'),
            'dvh_max_diff': float('nan'),
            'dvh_d95_diff': float('nan'),
            'dvh_d50_diff': float('nan'),
            'dvh_v95_diff': float('nan')
        }

    # Calculate dose range
    dose_min = min(dose_true_values.min(), dose_pred_values.min())
    dose_max = max(dose_true_values.max(), dose_pred_values.max())

    # Create dose bins
    dose_edges = np.linspace(dose_min, dose_max, dose_bins + 1)

    # Calculate histograms
    hist_true, _ = np.histogram(dose_true_values, bins=dose_edges)
    hist_pred, _ = np.histogram(dose_pred_values, bins=dose_edges)

    # Normalize the histograms
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)

    # Calculate cumulative histograms (DVH)
    dvh_true = 1.0 - np.cumsum(hist_true)
    dvh_pred = 1.0 - np.cumsum(hist_pred)

    # Calculate metrics
    # Area between the DVH curves
    dvh_diff_area = np.abs(dvh_true - dvh_pred).sum() / dose_bins

    # Maximum difference between DVH curves
    dvh_max_diff = np.abs(dvh_true - dvh_pred).max()

    # D95 (dose to 95% of the volume)
    d95_idx_true = np.argmin(np.abs(dvh_true - 0.95))
    d95_idx_pred = np.argmin(np.abs(dvh_pred - 0.95))
    d95_true = dose_edges[d95_idx_true]
    d95_pred = dose_edges[d95_idx_pred]
    dvh_d95_diff = np.abs(d95_true - d95_pred) / (dose_max - dose_min)

    # D50 (median dose)
    d50_idx_true = np.argmin(np.abs(dvh_true - 0.5))
    d50_idx_pred = np.argmin(np.abs(dvh_pred - 0.5))
    d50_true = dose_edges[d50_idx_true]
    d50_pred = dose_edges[d50_idx_pred]
    dvh_d50_diff = np.abs(d50_true - d50_pred) / (dose_max - dose_min)

    # V95 (volume receiving 95% of the prescription dose)
    # Assuming prescription dose is the maximum dose
    v95_dose = 0.95 * dose_max
    v95_idx = np.argmin(np.abs(dose_edges - v95_dose))
    v95_true = dvh_true[v95_idx]
    v95_pred = dvh_pred[v95_idx]
    dvh_v95_diff = np.abs(v95_true - v95_pred)

    return {
        'dvh_diff_area': dvh_diff_area,
        'dvh_max_diff': dvh_max_diff,
        'dvh_d95_diff': dvh_d95_diff,
        'dvh_d50_diff': dvh_d50_diff,
        'dvh_v95_diff': dvh_v95_diff
    }


def gamma_analysis(dose_true, dose_pred, dta_criterion=3.0, dose_criterion=0.03, threshold=0.1):
    """
    Calculate gamma analysis metric for dose distributions.

    Args:
        dose_true (numpy.ndarray): Ground truth dose distribution
        dose_pred (numpy.ndarray): Predicted dose distribution
        dta_criterion (float): Distance to agreement criterion in mm
        dose_criterion (float): Dose difference criterion as a fraction of max dose
        threshold (float): Dose threshold as a fraction of max dose

    Returns:
        float: Gamma pass rate
    """
    # This is a placeholder for the gamma analysis function
    # A full implementation would require spatial information and is quite complex
    # Libraries like pymedphys can be used for this purpose

    print("Warning: gamma_analysis is a placeholder function")
    print("Consider using pymedphys.gamma or equivalent for a proper implementation")

    # For now, return a placeholder value
    return None


def calculate_all_metrics(y_true, y_pred, include_dvh=False, include_gamma=False, roi_mask=None):
    """
    Calculate all available metrics.

    Args:
        y_true (numpy.ndarray or torch.Tensor): Ground truth values
        y_pred (numpy.ndarray or torch.Tensor): Predicted values
        include_dvh (bool): Whether to include DVH metrics
        include_gamma (bool): Whether to include gamma analysis
        roi_mask (numpy.ndarray, optional): Region of interest mask for DVH metrics

    Returns:
        dict: Dictionary of all metrics
    """
    metrics = {
        'mse': mse_loss(y_true, y_pred),
        'mae': mae_loss(y_true, y_pred),
        'rmse': rmse_loss(y_true, y_pred),
        'nrmse': nrmse_loss(y_true, y_pred),
        'psnr': psnr_metric(y_true, y_pred)
    }

    # Add SSIM if available
    ssim = ssim_metric(y_true, y_pred)
    if ssim is not None:
        metrics['ssim'] = ssim

    # Add DVH metrics if requested
    if include_dvh:
        dvh_metrics_dict = dvh_metrics(y_true, y_pred, roi_mask)
        metrics.update(dvh_metrics_dict)

    # Add gamma analysis if requested
    if include_gamma:
        gamma_pass_rate = gamma_analysis(y_true, y_pred)
        if gamma_pass_rate is not None:
            metrics['gamma_pass_rate'] = gamma_pass_rate

    return metrics