import os
import shutil

# Define paths
source_folder_ct = '/media/e210/HD81/Adhvaith_datasets/og_nrrd'
source_folder_ctr = '/media/e210/HD81/Adhvaith_datasets/dose_masks/ctr'
source_folder_ipsi = '/media/e210/HD81/Adhvaith_datasets/dose_masks/ipsi'
destination_base = '/media/e210/HD81/Adhvaith_datasets/nsclc_public'

# Create destination base if it doesn't exist
os.makedirs(destination_base, exist_ok=True)


# Function to copy and rename
def copy_and_rename(source_folder, destination_base, rename_to, file_suffix):
    for filename in os.listdir(source_folder):
        if filename.endswith(file_suffix):
            # Extract patient ID
            patient_id = filename.split('_')[0]

            # Create destination folder
            patient_folder = os.path.join(destination_base, patient_id)
            os.makedirs(patient_folder, exist_ok=True)

            # Full paths
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(patient_folder, rename_to)

            # Copy file
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} -> {destination_file}")


# --- Step 1: Copy Files ---
copy_and_rename(source_folder_ct, destination_base, rename_to='image_ct.nrrd', file_suffix='.nrrd')
copy_and_rename(source_folder_ctr, destination_base, rename_to='mask_lungctr.nrrd', file_suffix='LUNGCTR.nrrd')
copy_and_rename(source_folder_ipsi, destination_base, rename_to='mask_lungipsi.nrrd', file_suffix='LUNGIPSI.nrrd')

print("\n✅ File copying completed.")

# --- Step 2: Validate Files per Patient ---
print("\n🔍 Checking for missing files...")

# Expected filenames
expected_files = {'image_ct.nrrd', 'mask_lungctr.nrrd', 'mask_lungipsi.nrrd'}

# Track patients with missing files
patients_with_missing_files = []

# Loop through all patient folders
for patient_id in os.listdir(destination_base):
    patient_folder = os.path.join(destination_base, patient_id)
    if os.path.isdir(patient_folder):
        files_present = set(os.listdir(patient_folder))
        missing_files = expected_files - files_present
        if missing_files:
            patients_with_missing_files.append((patient_id, missing_files))

# --- Step 3: Report ---
if patients_with_missing_files:
    print("\n⚠️ Patients with missing files:")
    for patient_id, missing in patients_with_missing_files:
        print(f"Patient {patient_id} is missing: {', '.join(missing)}")
else:
    print("\n✅ All patients have the 3 expected files.")

