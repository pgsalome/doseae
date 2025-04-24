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
            patch = image[i: i + p_h, j: j + p_w]
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
        volume (numpy.ndarray): Input 3D volume (D, H, W)
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

    # Calculate the total number of possible patches
    all_patches = (i_d - p_d + 1) * (i_h - p_h + 1) * (i_w - p_w + 1)

    # If max_patches is a fraction, calculate the actual number
    if max_patches is not None and isinstance(max_patches, float) and 0 < max_patches < 1:
        max_patches = int(max_patches * all_patches)

    # If we need to select patches randomly
    if max_patches is not None and max_patches < all_patches:
        rng = np.random.RandomState(random_state)

        # Generate random indices
        d_indices = rng.randint(0, i_d - p_d + 1, size=max_patches)
        h_indices = rng.randint(0, i_h - p_h + 1, size=max_patches)
        w_indices = rng.randint(0, i_w - p_w + 1, size=max_patches)

        # Extract the patches
        patches = np.array([
            volume[d:d + p_d, h:h + p_h, w:w + p_w]
            for d, h, w in zip(d_indices, h_indices, w_indices)
        ])
    else:
        # Extract all patches systematically
        patches = []
        for d in range(0, i_d - p_d + 1):
            for h in range(0, i_h - p_h + 1):
                for w in range(0, i_w - p_w + 1):
                    patch = volume[d:d + p_d, h:h + p_h, w:w + p_w]
                    patches.append(patch)
        patches = np.array(patches)

    return patches


def extract_informative_patches_3d(volume, patch_size=(16, 64, 64), threshold=0.01):
    """
    Extract 3D patches with information content (standard deviation > threshold).

    Args:
        volume (numpy.ndarray): 3D volume to extract patches from
        patch_size (tuple): Size of patches (depth, height, width)
        threshold (float): Minimum standard deviation threshold for a patch to be considered informative

    Returns:
        numpy.ndarray: Array of informative patches
    """
    i_d, i_h, i_w = volume.shape
    p_d, p_h, p_w = patch_size

    if p_d > i_d or p_h > i_h or p_w > i_w:
        raise ValueError("Patch size should be smaller than volume dimensions.")

    patches = []
    for d in range(0, i_d - p_d + 1, p_d):
        for h in range(0, i_h - p_h + 1, p_h):
            for w in range(0, i_w - p_w + 1, p_w):
                patch = volume[d:d + p_d, h:h + p_h, w:w + p_w]
                if np.std(patch) > threshold:
                    patches.append(patch)

    return np.array(patches)


def resize_2d_slices(npy_path, output_dir, new_shape=(256, 256)):
    """
    Resize slices in a .npy file.

    Args:
        npy_path (str): Path to .npy file
        output_dir (str): Output directory
        new_shape (tuple): New shape (height, width)

    Returns:
        str: Path to resized file
    """
    # Load the .npy file
    slices = np.load(npy_path)

    # Create an empty array to store the resized slices
    resized_slices = np.zeros((slices.shape[0], new_shape[0], new_shape[1]), dtype=slices.dtype)

    # Resize each slice
    for i in range(slices.shape[0]):
        original_slice = slices[i]
        resized_slice = cv2.resize(original_slice, new_shape, interpolation=cv2.INTER_AREA)
        resized_slices[i] = resized_slice

    # Generate the output file path
    base_name = os.path.basename(npy_path)
    resized_npy_path = os.path.join(output_dir, base_name)

    # Save the resized slices back to a .npy file
    np.save(resized_npy_path, resized_slices)

    print(f"Resized slices saved to {resized_npy_path}")
    return resized_npy_path


def main():
    """
    Command-line interface for patch extraction.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Extract informative patches from dose distribution data")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract patches command
    extract_parser = subparsers.add_parser("extract", help="Extract informative patches")
    extract_parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .npy files")
    extract_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted patches")
    extract_parser.add_argument("--patch-size", type=int, nargs=2, default=[64, 64], help="Patch size (height width)")
    extract_parser.add_argument("--threshold", type=float, default=0.01, help="Standard deviation threshold")

    # Resize command
    resize_parser = subparsers.add_parser("resize", help="Resize slices in .npy files")
    resize_parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .npy files")
    resize_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save resized files")
    resize_parser.add_argument("--new-shape", type=int, nargs=2, default=[256, 256], help="New shape (height width)")

    args = parser.parse_args()

    if args.command == "extract":
        process_npy_folder(args.input_dir, args.output_dir, tuple(args.patch_size), args.threshold)
    elif args.command == "resize":
        # Process all .npy files in the input directory
        os.makedirs(args.output_dir, exist_ok=True)
        for file_name in os.listdir(args.input_dir):
            if file_name.endswith(".npy"):
                npy_path = os.path.join(args.input_dir, file_name)
                resize_2d_slices(npy_path, args.output_dir, tuple(args.new_shape))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()