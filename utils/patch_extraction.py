"""
Utilities for extracting informative patches from dose distribution data.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm


def extract_informative_patches_2d(image, patch_size=(64, 64), threshold=0.01):
    """
    Extract patches with information content (standard deviation > threshold).

    Args:
        image (numpy.ndarray): 2D image to extract patches from
        patch_size (tuple): Size of patches (height, width)
        threshold (float): Minimum standard deviation threshold for a patch to be considered informative

    Returns:
        numpy.ndarray: Array of informative patches
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h or p_w > i_w:
        raise ValueError("Patch size should be smaller than image dimensions.")

    patches = []
    for i in range(0, i_h - p_h + 1, p_h):
        for j in range(0, i_w - p_w + 1, p_w):
            patch = image[i : i + p_h, j : j + p_w]
            if np.std(patch) > threshold:
                patches.append(patch)

    return np.array(patches)


def process_single_npy_file(file_path, save_dir, patch_size=(64, 64), threshold=0.01):
    """
    Process a single .npy file to extract informative patches.

    Args:
        file_path (str): Path to the .npy file
        save_dir (str): Directory to save extracted patches
        patch_size (tuple): Size of patches to extract
        threshold (float): Minimum standard deviation threshold

    Returns:
        int: Number of patches extracted
    """
    image = np.load(file_path)
    print(f"Loaded image from {file_path} with shape {image.shape}")

    all_patches = []

    # Handle different dimensions
    if image.ndim == 3 and image.shape[0] > 1:  # Multiple slices
        for slice_idx in range(image.shape[0]):
            slice_image = image[slice_idx]
            patches = extract_informative_patches_2d(slice_image, patch_size, threshold)
            if patches.size > 0:
                all_patches.append(patches)
    elif image.ndim == 3 and image.shape[2] > 1:  # Different ordering of dimensions
        for slice_idx in range(image.shape[2]):
            slice_image = image[:, :, slice_idx]
            patches = extract_informative_patches_2d(slice_image, patch_size, threshold)
            if patches.size > 0:
                all_patches.append(patches)
    else:  # Single slice
        patches = extract_informative_patches_2d(image, patch_size, threshold)
        if patches.size > 0:
            all_patches.append(patches)

    if all_patches:
        all_patches = np.concatenate(all_patches, axis=0)
        print(f"Extracted {all_patches.shape[0]} informative patches of shape {patch_size} from image")

        file_name = os.path.basename(file_path)
        save_path = os.path.join(
            save_dir, os.path.splitext(file_name)[0] + "_informative_patches.npy"
        )
        np.save(save_path, all_patches)
        print(f"Informative patches extracted and saved to {save_path}")
        return all_patches.shape[0]
    else:
        print("No informative patches found.")
        return 0


def process_npy_folder(folder_path, save_dir, patch_size=(64, 64), threshold=0.01):
    """
    Process all .npy files in a folder to extract informative patches.

    Args:
        folder_path (str): Path to folder containing .npy files
        save_dir (str): Directory to save extracted patches
        patch_size (tuple): Size of patches to extract
        threshold (float): Minimum standard deviation threshold

    Returns:
        dict: Statistics of extraction process
    """
    os.makedirs(save_dir, exist_ok=True)

    total_files = 0
    total_patches = 0

    for file_name in tqdm(os.listdir(folder_path), desc="Processing files"):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            patches_extracted = process_single_npy_file(file_path, save_dir, patch_size, threshold)
            total_files += 1
            total_patches += patches_extracted

    stats = {
        'total_files_processed': total_files,
        'total_patches_extracted': total_patches,
        'avg_patches_per_file': total_patches / total_files if total_files > 0 else 0
    }

    print(f"Total files processed: {total_files}")
    print(f"Total patches extracted: {total_patches}")
    print(f"Average patches per file: {stats['avg_patches_per_file']:.2f}")

    return stats


def extract_patches_3d(volume, patch_size, max_patches=None, random_state=None):
    """
    Extract patches from a 3D volume.

    Args:
        volume (numpy.ndarray): Input volume (D, H, W)
        patch_size (tuple): Size of the patches (depth, height, width)
        max_patches (int or float, optional): Maximum number of patches to extract
        random_state (int or RandomState, optional): Random seed or state

    Returns:
        numpy.ndarray: Extracted patches
    """
    i_d, i_h, i_w = volume.shape[:3]
    p_d, p_h, p_w = patch_size

    if p_d > i_d or p_h > i_h or p_w > i_w:
        raise ValueError(
            "Patch dimensions should be less than the corresponding volume dimensions."
        )

    # Reshape to add a channel dimension if needed
    if volume.ndim == 3:
        volume = volume.reshape((i_d, i_h, i_w, 1))

    n_channels = volume.shape[3]

    # Calculate the total number of possible patches
    d_indices = range(0, i_d - p_d + 1, p_d)
    h_indices = range(0, i_h - p_h + 1, p_h)
    w_indices = range(0, i_w - p_w + 1, p_w)

    # Extract all patches
    patches = []
    for d in d_indices:
        for h in h_indices:
            for w in w_indices:
                patch = volume[d:d+p_d, h:h+p_h, w:w+p_w, :]
                # Only keep patches with information (non-zero std)
                if np.std(patch) > 0.01:
                    patches.append(patch)

    # Convert to numpy array
    patches = np.array(patches)

    # Reshape to remove channel dimension if it's 1
    if n_channels == 1:
        patches = patches.reshape(-1, p_d, p_h, p_w)

    # If max_patches is provided and less than the total number of patches,
    # randomly select a subset
    if max_patches and max_patches < len(patches):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(patches), size=max_patches, replace=False)
        patches = patches[indices]

    return patches


def resize_3d_volume(volume, target_shape):
    """
    Resize a 3D volume to the target shape.

    Args:
        volume (numpy.ndarray): Input 3D volume
        target_shape (tuple): Target shape (D, H, W)

    Returns:
        numpy.ndarray: Resized volume
    """
    from scipy.ndimage import zoom

    # Calculate scale factors
    factors = [t / s for t, s in zip(target_shape, volume.shape)]

    # Apply zoom
    resized = zoom(volume, factors, order=1, mode='nearest')

    return resized


def main():
    """
    Command-line interface for patch extraction.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Extract informative patches from dose distribution data")
    parser.add_argument('--input', type=str, required=True, help="Input .npy file or directory")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--patch-size', type=int, nargs=2, default=[64, 64], help="Patch size (height width)")
    parser.add_argument('--threshold', type=float, default=0.01, help="Minimum standard deviation threshold")
    parser.add_argument('--is-directory', action='store_true', help="Input is a directory of .npy files")

    args = parser.parse_args()

    if args.is_directory:
        process_npy_folder(args.input, args.output, tuple(args.patch_size), args.threshold)
    else:
        process_single_npy_file(args.input, args.output, tuple(args.patch_size), args.threshold)


if __name__ == "__main__":
    main()