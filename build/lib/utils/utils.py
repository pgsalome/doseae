"""
Utility functions for dose distribution autoencoder project.
"""

import os
import torch
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def save_model(model, output_dir):
    """
    Save model to disk.

    Args:
        model (torch.nn.Module): Model to save
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Model saved to {output_dir}")


def save_history(history, output_dir):
    """
    Save training history to disk.

    Args:
        history (dict): Training history
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "history.txt"), "w") as f:
        for key, value in history.items():
            f.write(f"{key}: {value}\n")


def to_img(x):
    """
    Convert tensor to image format for visualization.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Normalized tensor in image format
    """
    # Normalize the tensor to range [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # Ensure the tensor has a channel dimension
    if x.ndim == 3:  # If x does not have channel dimension [B, H, W]
        x = x.unsqueeze(1)  # [B, 1, H, W]
    elif x.shape[1] != 1 and x.ndim == 4:  # Ensure single channel for 2D images
        x = x[:, 0:1, :, :]

    return x


def save_slice(slice_data, output_dir, base_filename, slice_index):
    """
    Save a 2D slice to an NRRD file.

    Args:
        slice_data (numpy.ndarray): 2D slice data
        output_dir (str): Output directory
        base_filename (str): Base filename
        slice_index (int): Slice index
    """
    output_filename = f"{base_filename}_slice_{slice_index:03d}.nrrd"
    output_path = os.path.join(output_dir, output_filename)
    nrrd.write(output_path, slice_data)
    print(f"Saved slice {slice_index} to {output_path}")


def extract_slices(nrrd_file, extract_mode="all", threshold=0.1):
    """
    Extract slices from an NRRD file.

    Args:
        nrrd_file (str): Path to NRRD file
        extract_mode (str): Extraction mode ('all' or 'mid50')
        threshold (float): Threshold for non-zero ratio

    Returns:
        list: List of extracted slices
    """
    # Read the NRRD file
    data, header = nrrd.read(nrrd_file)
    print(f"Data shape: {data.shape}")

    # Compute the 95th percentile value of the data
    norm_value = 0.95 * np.max(data)

    # Normalize the data to the 95th percentile
    if norm_value == 0:
        print("EMPTY")
        print(nrrd_file)
        return []

    data = np.clip(data / norm_value, 0, 1)

    # Determine slice range based on extract_mode
    if extract_mode == "mid50":
        # Find the slice with the maximum intensity value
        max_intensity_index = np.argmax(np.max(data, axis=(0, 1)))

        # Determine the slice indices to extract (25 before and 25 after the middle slice)
        start_index = max(0, max_intensity_index - 25)
        end_index = min(data.shape[2], max_intensity_index + 25 + 1)
    elif extract_mode == "all":
        start_index = 0
        end_index = data.shape[2]
    else:
        raise ValueError("Invalid extract mode. Choose 'mid50' or 'all'.")

    # Extract slices
    slices = []
    empty = True

    for slice_index in range(start_index, end_index):
        slice_data = data[:, :, slice_index]

        # Check if slice has any non-zero elements
        if np.any(slice_data):
            slices.append(slice_data)
            empty = False

    if empty:
        print("empty-" + nrrd_file)

    return slices


def save_slices_to_numpy(slices, output_path):
    """
    Save slices to a numpy file.

    Args:
        slices (list): List of slices
        output_path (str): Output path
    """
    np.save(output_path, slices)
    print(f"Saved {len(slices)} slices to {output_path}")


def visualize_nrrd_histogram(nrrd_file_path):
    """
    Visualize histogram of NRRD file data.

    Args:
        nrrd_file_path (str): Path to NRRD file
    """
    # Load the .nrrd file
    data, header = nrrd.read(nrrd_file_path)

    # Flatten the data to create a histogram
    flattened_data = data.flatten()

    # Plot histogram
    plt.hist(flattened_data, bins=50, range=(0, np.max(flattened_data)))
    plt.title(f"Histogram of data values in {nrrd_file_path}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print(f"Min value: {np.min(flattened_data)}")
    print(f"Max value: {np.max(flattened_data)}")
    print(f"Mean value: {np.mean(flattened_data)}")
    print(f"95th percentile: {np.percentile(flattened_data, 95)}")


def verify_normalization(file_path, expected_percentile_value=1.0):
    """
    Verify normalization of data in numpy file.

    Args:
        file_path (str): Path to numpy file
        expected_percentile_value (float): Expected value for 95th percentile

    Returns:
        tuple: Normalization statistics
    """
    try:
        data = np.load(file_path)
        if data.size == 0:
            raise ValueError("Empty data array")

        # Check for NaNs and infinite values
        if np.isnan(data).any():
            print("Data contains NaN values, replacing with zeros.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isinf(data).any():
            print("Data contains infinite values, replacing with zeros.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate 95th percentile
        current_95th_percentile = np.percentile(data, 95)
        is_normalized = (abs(current_95th_percentile - expected_percentile_value) < 0.01)

        # Additional diagnostics
        min_value = np.min(data)
        max_value = np.max(data)
        mean_value = np.mean(data)
        std_dev = np.std(data)

        return (
            current_95th_percentile,
            is_normalized,
            None,
            min_value,
            max_value,
            mean_value,
            std_dev,
            data,
        )
    except Exception as e:
        return None, None, str(e), None, None, None, None, None


def verify_single_file(file_path, show_histogram=True):
    """
    Verify normalization and display information about a single numpy file.

    Args:
        file_path (str): Path to numpy file
        show_histogram (bool): Whether to show histogram
    """
    (
        current_95th_percentile,
        is_normalized,
        error,
        min_value,
        max_value,
        mean_value,
        std_dev,
        data,
    ) = verify_normalization(file_path)

    print(f"Verification results for {file_path}:")
    print(f"95th Percentile Value: {current_95th_percentile}")
    print(f"Is Normalized: {is_normalized}")
    print(f"Error: {error}")
    print(f"Min Value: {min_value}")
    print(f"Max Value: {max_value}")
    print(f"Mean Value: {mean_value}")
    print(f"Standard Deviation: {std_dev}")

    # Plot histogram of data
    if data is not None and show_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(data.flatten(), bins=50, range=(0, 1))
        plt.title(f"Histogram of data values in {file_path}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    if error:
        try:
            data = np.load(file_path)
            print(f"Data type: {data.dtype}")
            print(f"Data shape: {data.shape}")
            print(f"Data sample: {data.flatten()[0:10]}")
        except Exception as e:
            print(f"Error loading data for diagnostics: {e}")


def analyze_datasets_statistics(directory, output_csv=None):
    """
    Analyze statistics of all .npy files in a directory.

    Args:
        directory (str): Directory containing .npy files
        output_csv (str, optional): Path to save CSV results

    Returns:
        pandas.DataFrame: DataFrame with statistics
    """
    # List to hold the data
    data = []

    # Traverse the directory and process each .npy file
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)

            # Load the .npy file
            array = np.load(filepath)

            # Get the shape of the array
            shape = array.shape

            # Calculate statistics
            mean = array.mean()
            std = array.std()
            min_val = array.min()
            max_val = array.max()
            percentile_95 = np.percentile(array, 95)
            non_zero_ratio = np.count_nonzero(array) / array.size

            # Append the data to the list
            data.append([
                filename,
                shape,
                mean,
                std,
                min_val,
                max_val,
                percentile_95,
                non_zero_ratio
            ])

    # Create a DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "Filename",
            "Shape",
            "Mean",
            "Standard Deviation",
            "Min",
            "Max",
            "95th Percentile",
            "Non-zero Ratio"
        ]
    )

    # Save the DataFrame to a CSV file if specified
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"CSV file '{output_csv}' created successfully.")

    return df


def visualize_patches(npy_file_path, num_patches=5):
    """
    Visualize patches from a .npy file.

    Args:
        npy_file_path (str): Path to .npy file containing patches
        num_patches (int): Number of patches to visualize
    """
    # Load patches
    patches = np.load(npy_file_path)
    print(f"Loaded {patches.shape[0]} patches from {npy_file_path}")

    # Determine patch dimensions
    if patches.ndim == 4:  # [N, D, H, W] - 3D patches
        is_3d = True
        # Select middle slice of each patch for visualization
        middle_idx = patches.shape[1] // 2
        visualized_patches = patches[:, middle_idx, :, :]
    else:  # [N, H, W] - 2D patches
        is_3d = False
        visualized_patches = patches

    # Create figure
    plt.figure(figsize=(15, 3))

    # Plot patches
    for i in range(min(num_patches, patches.shape[0])):
        plt.subplot(1, min(num_patches, patches.shape[0]), i + 1)
        plt.imshow(visualized_patches[i], cmap="viridis")
        plt.axis("off")
        plt.title(f"Patch {i + 1}" + (" (mid slice)" if is_3d else ""))

    plt.tight_layout()
    plt.show()


def visualize_dataset(dataset, num_samples=5):
    """
    Visualize samples from a dataset.

    Args:
        dataset (Dataset): Dataset to visualize
        num_samples (int): Number of samples to visualize
    """
    # Create DataLoader to sample from dataset
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # Get one batch
    batch = next(iter(loader))

    # Handle different dataset structures
    if isinstance(batch, tuple):
        images = batch[0]  # (input, target) format
    elif isinstance(batch, dict):
        images = batch["image"]  # Dictionary format
    else:
        images = batch

    # Determine if data is 2D or 3D
    if images.ndim == 5:  # [B, C, D, H, W]
        is_3d = True
        # Select middle slice of each volume for visualization
        middle_idx = images.shape[2] // 2
        visualized_images = images[:, :, middle_idx, :, :]
    else:  # [B, C, H, W]
        is_3d = False
        visualized_images = images

    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    # Plot images
    for i in range(min(num_samples, visualized_images.size(0))):
        # Get image and remove channel dimension if it's 1
        img = visualized_images[i, 0].cpu().numpy() if visualized_images.size(1) == 1 else visualized_images[
            i].cpu().numpy().transpose(1, 2, 0)

        # Plot
        if num_samples == 1:
            axes.imshow(img, cmap="viridis")
            axes.set_title(f"Sample {i + 1}" + (" (mid slice)" if is_3d else ""))
            axes.axis("off")
        else:
            axes[i].imshow(img, cmap="viridis")
            axes[i].set_title(f"Sample {i + 1}" + (" (mid slice)" if is_3d else ""))
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


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
    import cv2  # Import here to avoid dependency if function not used

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


def process_directory_resize(input_dir, output_dir, new_shape=(256, 256)):
    """
    Resize all .npy files in a directory.

    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        new_shape (tuple): New shape (height, width)

    Returns:
        int: Number of files processed
    """
    import cv2  # Import here to avoid dependency if function not used

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Counter for processed files
    processed_files = 0

    # Iterate over all .npy files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            npy_path = os.path.join(input_dir, file_name)
            resize_2d_slices(npy_path, output_dir, new_shape)
            processed_files += 1

    # Print a summary statement
    print(f"Total files processed: {processed_files}")
    print("All .npy arrays have been resized successfully.")

    return processed_files


def main():
    """
    Command-line interface for utility functions.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Utility functions for dose distribution data")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract slices command
    extract_parser = subparsers.add_parser("extract_slices", help="Extract slices from NRRD file")
    extract_parser.add_argument("--input", type=str, required=True, help="Input NRRD file")
    extract_parser.add_argument("--output", type=str, required=True, help="Output numpy file")
    extract_parser.add_argument("--mode", type=str, default="all", choices=["all", "mid50"], help="Extraction mode")
    extract_parser.add_argument("--threshold", type=float, default=0.1, help="Non-zero ratio threshold")

    # Verify normalization command
    verify_parser = subparsers.add_parser("verify", help="Verify normalization of numpy file")
    verify_parser.add_argument("--input", type=str, required=True, help="Input numpy file")
    verify_parser.add_argument("--expected-percentile", type=float, default=1.0, help="Expected 95th percentile value")

    # Analyze dataset command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze statistics of numpy files in directory")
    analyze_parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    analyze_parser.add_argument("--output-csv", type=str, help="Output CSV file")

    # Resize slices command
    resize_parser = subparsers.add_parser("resize", help="Resize slices in numpy file")
    resize_parser.add_argument("--input", type=str, required=True, help="Input numpy file")
    resize_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    resize_parser.add_argument("--new-shape", type=int, nargs=2, default=[256, 256], help="New shape (height width)")

    # Visualize patches command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize patches from numpy file")
    visualize_parser.add_argument("--input", type=str, required=True, help="Input numpy file")
    visualize_parser.add_argument("--num-patches", type=int, default=5, help="Number of patches to visualize")

    args = parser.parse_args()

    if args.command == "extract_slices":
        slices = extract_slices(args.input, args.mode, args.threshold)
        if slices:
            save_slices_to_numpy(slices, args.output)
    elif args.command == "verify":
        verify_single_file(args.input, expected_percentile_value=args.expected_percentile)
    elif args.command == "analyze":
        analyze_datasets_statistics(args.input_dir, args.output_csv)
    elif args.command == "resize":
        resize_2d_slices(args.input, args.output_dir, tuple(args.new_shape))
    elif args.command == "visualize":
        visualize_patches(args.input, args.num_patches)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()