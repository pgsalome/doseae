import h5py
import numpy as np
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# 1. DEFINE THE PATH to your current, inefficient HDF5 file
input_h5_path = '/data/pgsal/NSCLC-Cetuximab_results_05mm/patches_dataset_final.h5'

# 2. DEFINE THE PATH for the new, efficient HDF5 file that will be created
output_h5_path = '/data/pgsal/NSCLC-Cetuximab_results_05mm/patches_dataset_efficient.h5'
# --- END CONFIGURATION ---


def convert_hdf5_structure(input_path, output_path):
    """
    Reads an HDF5 file with many small groups and converts it to a file
    with one large, consolidated dataset for efficient loading.
    """
    print(f"Starting conversion...")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at {input_path}")
        return

    try:
        # --- Step 1: Read all data from the old file into memory ---
        all_patches_data = []
        with h5py.File(input_path, 'r') as f_in:
            patch_keys = sorted([key for key in f_in.keys() if key.startswith('patch_')])
            print(f"Found {len(patch_keys)} patches to convert.")

            for key in tqdm(patch_keys, desc="Reading old file"):
                all_patches_data.append(f_in[key]['data'][:])

        if not all_patches_data:
            print("No patch data found in the input file. Aborting.")
            return

        # --- Step 2: Write all data into the new, efficient file ---
        total_patches = len(all_patches_data)
        patch_shape = all_patches_data[0].shape

        with h5py.File(output_path, 'w') as f_out:
            # Create a single, large dataset named 'patches'
            dataset = f_out.create_dataset(
                'patches',
                shape=(total_patches, *patch_shape),
                dtype=np.float32,
                chunks=(1, *patch_shape), # Chunking by patch is efficient
                compression='gzip'
            )

            # Write the data from the list into the new dataset
            dataset[:] = all_patches_data

        print("\nConversion complete!")
        print(f"New efficient file created at: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during conversion: {e}")


if __name__ == "__main__":
    convert_hdf5_structure(input_h5_path, output_h5_path)