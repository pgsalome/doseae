# RUN WITH CONDA ENVIRONMENT prototwin-pet (environment.yml, install for any PC with conda env create -f environment.yml)

import os
import sys
import shutil
import subprocess
import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib.patches as mpatches
import numpy as np
import pymedphys
from scipy.interpolate import RegularGridInterpolator
from utils import (
    crop_save_npy,
    convert_CT_to_mhd,
    mhd_resolution_size,
    crop_resize_save,
    convert_mhd_to_dcm,
    read_rtplan,
    load_rtdose,
)


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ---------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#   PATIENT DATA AND OUTPUT FOLDERS
# Phantom #
rtplan = os.path.join(script_dir, "../data/DATOS EXPERIMENTOS CUN 2025/MANIQUI ANTROPOMORFICO JULIO 2025/PLAN TRATAMIENTO/RP1.2.752.243.1.1.20250627070211748.2000.13642.dcm")
rtdose = os.path.join(script_dir, "../data/DATOS EXPERIMENTOS CUN 2025/MANIQUI ANTROPOMORFICO JULIO 2025/PLAN TRATAMIENTO/RD1.2.752.243.1.1.20250627070211749.5000.20681.dcm")
beam_origin = 192.3    # also specified in prototwin-pet-cun/generate-dataset/original-fred-PHANTOM.inp,
                       # beam_origin given in rp.1.5... by VirtualSourceAxisDistances = 1923 (mm)
                       # taking the second one because it's the first point where the beam is deflected

voxel_size = np.array([2., 2., 2.]) # in mm. Tamaño de los voxeles considerando el num de voxeles y el tamaño de la placa.
cropped_shape = np.array([150, 150, 150]) # numero de voxelees en x,y,z

#
#   CHOOSING A DOSE VERIFICATION APPROACH
irradiation_time = [2.32]  # minutes  # time spent delivering the field
field_setup_time = [0]  # minutes  # time spent setting up the field (gantry rotation)
#
#   MONTE CARLO SIMULATION OF THE TREATMENT
N_sobps = 1 # número de planes SOBPS
Espread = 0.001  # fractional energy spread (for hitachi, <0.2 https://www.ptcog.site/archive/conference_p%26t%26v/PTCOG46/pdf/hiramoto_may_18.pdf)
target_dose = 2.    # Gy  (specifies 20Gy in 10, not biologically equivalent, I think)
                    # /home/prototwin/prototwin-pet-cun/data/DATOS EXPERIMENTOS CUN 2025/MANIQUI ANTROPOMORFICO 2025/PET/RP1.2.752.243.1.1.20250627070211748.2000.13642.dcm:
                    #   (300A,0026) DS TargetPrescriptionDose = 20   (300A,0078) IS NumberOfFractionsPlanned = 10
# -----------------------------------------------------------------------------------------------------------------------------------------

dataset_num='PHANTOM'
seed_number=42

dataset_folder = os.path.join(
    script_dir, f"../data/dataset{dataset_num}"
)  # Folder to save the numpy arrays for model training.
if not os.path.exists(dataset_folder): #si la carpeta no existe, se crea una nueva con carpeta de actividad y dosis
    os.makedirs(dataset_folder)
    os.makedirs(os.path.join(dataset_folder, "dose/"))
images_folder = os.path.join(script_dir, f"images/{dataset_num}")  # Folder to save the images for model training.  
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

CT_mhd_file = os.path.join(dataset_folder, "CT.mhd")  # mhd file with the CT

CT_dcm_dir = os.path.join(script_dir, "../data/DATOS EXPERIMENTOS CUN 2025/MANIQUI ANTROPOMORFICO JULIO 2025/CTmaniqui")
water_layer = True  # the patient table is like 8mm of water, so we add a water layer to the CT image
CT_isocenter, CT_origin = convert_CT_to_mhd(CT_mhd_file, dicom_dir=CT_dcm_dir, water_layer=water_layer)
print(f"CT isocenter: {CT_isocenter} mm") #isocentro del CT, que es el centro de la placa
# CT_isocenter[0] = -CT_isocenter[0]  ###

CT_voxel_size, uncropped_shape = mhd_resolution_size(CT_mhd_file)

print(f"CT voxel size: {CT_voxel_size} mm")
print(f"CT shape: {uncropped_shape} voxels")
    
#ahora ya tenemos un archivo de entrada MCGPU con los datos correctos.

final_shape = np.array(cropped_shape) # no lo he recortado. Final shape es el tamaño de la zona irradiada.

L_list = [
    uncropped_shape[0] * CT_voxel_size[0]/10,
    uncropped_shape[1] * CT_voxel_size[1]/10,
    uncropped_shape[2] * CT_voxel_size[2]/10,
]  # in cm. En vez de uncropped_shape le he puesto L
L_line_CT = f"    L=[{', '.join(map(str, L_list))}]"
hu2densities_path = os.path.join(script_dir, "../data/ipot-hu2materials.txt")
with open(hu2densities_path, "r+") as file:
    schneider_lines = file.readlines() #cada línea contiene valores de HU, densidad, poder de frenado... de cada region

fredinp_location = os.path.join(script_dir, f"original-fred-{dataset_num}.inp") #el fred original, que es el input, se localiza donde este mismo archivo.
# Replace activation line with the appropriate for the selected isotopes and include variance reduction
with open(fredinp_location, "r") as file: #lee el archivo de fred original
    fredinp_lines = file.readlines() #guarda todas las lineas en una lista
counter = 0  # first line for CT, second for field
with open(fredinp_location, "w") as file:
    for line in fredinp_lines:
        if line.lstrip().startswith("L=[") and counter == 0:
            line = L_line_CT + "\n"
            counter += 1
        if line.lstrip().startswith("CTscan"):
            line = f"    CTscan={CT_mhd_file}\n"
        file.write(line) #el resto de lineas se escriben sin cambios

# Accessing structs
num_fields = 1  # campos de irradiacion que se van a usar
isocenter = np.array([0, 0, 0])  ### Coloco el isocentro en el centro de la placa. La placa va de -5 a 5 cm en todas las direcciones (lados 10 cm)

df_rtplan_path = os.path.join(dataset_folder, "rtplan.csv")  # saving spot positions and energies in a CSV file
df_rtplan = read_rtplan(rtplan, df_rtplan_path)  # Read the RTPLAN dcm file to get the plan name and number of fractions

Trans_isocenter = [
        np.round(isocenter[0] * 10 / voxel_size[0]).astype(int),
        np.round(isocenter[1] * 10 / voxel_size[1]).astype(int),
        np.round(isocenter[2] * 10 / voxel_size[2]).astype(int),
    ] # Offset to place the beam and target in the center. Convierte el isocentro a unidades de voxel

Trans = (Trans_isocenter[0], Trans_isocenter[1], Trans_isocenter[2])  # offset for final crop

# Fix the random seed. seed se usa para que el numero aleatorio generado sea siempre el mismo
random.seed(seed_number)
np.random.seed(seed_number)
os.environ["PYTHONHASHSEED"] = str(seed_number)

# Cropping the CT
# CT_cropped HAS THE SHAPE OF THE CT CROPPED TO INCLUDE THE ENTIRE BODY, BUT THE FINAL CT USED
# FOR THE SIMULATION IS CROPPED TO THE FINAL SHAPE, ONLY INCLUDING THE AREAS WHERE ACTIVITY AND DOSE ARE PRESENT
# SO THE CT SAVED AT CT_npy_path IS MORE CROPPED THAN CT_cropped however ironic it is
CT_cropped = crop_resize_save(
    CT_mhd_file,
    os.path.join(dataset_folder, "CT.npy"),  # Path to save the cropped CT
    final_voxel_size=voxel_size,
    final_shape=cropped_shape
)

CT_uncropped = crop_resize_save(
    CT_mhd_file,
    os.path.join(dataset_folder, "CT_uncropped.npy"),  # Path to save the uncropped CT
    final_voxel_size=voxel_size,
)

plt.figure()
plt.imshow(CT_cropped[:, CT_cropped.shape[0] // 2, :], cmap="gray", vmin=-120, vmax=225)  # Show the middle slice
plt.axis("off")
plt.savefig(os.path.join(dataset_folder, "CT_cropped_y.png"), bbox_inches="tight")

plt.figure()
plt.imshow(CT_cropped[CT_cropped.shape[0] // 2, :, :], cmap="gray", vmin=-120, vmax=225)  # Show the middle slice
plt.axis("off")
plt.savefig(os.path.join(dataset_folder, "CT_cropped_x.png"), bbox_inches="tight")
plt.figure()
plt.imshow(CT_cropped[:, :, CT_cropped.shape[2] // 2], cmap="gray", vmin=-120, vmax=225)  # Show the middle slice
plt.axis("off")
plt.savefig(os.path.join(dataset_folder, "CT_cropped_z.png"), bbox_inches="tight")

sobp_start = 0 #numero inicial de planes SOBP (Spread-Out Bragg Peak, para combinar energias y picos)
for sobp_num in range(sobp_start, sobp_start + N_sobps): #cada iteracion es un nueevo SOBP
    # Create folder for each new deviated plan, which we call sobp because of the original name for the prostate
    sobp_folder_name = f"sobp{sobp_num}" #nomnbre de cada plan. Se guarda en plans_info

    # Iterating over the fields
    plan_pb_num = 0  # to keep track of all bixels, or pencil beams (pb) in the plan
    total_dose = 0  # to add the dose of all fields
    total_uncropped_dose = 0
    total_delivered_particles = 0

    for field_num in range(num_fields): #itera sobre cada campo de irradiación
        print(f"\nField {field_num} / {num_fields} of sobp {sobp_num}")
        field_pb_num = (
            0  # to keep track of all bixels, or pencil beams (pb) in the field
        )
        pencil_beams = []  # to store all field pencil beams
        sobp_folder_location = os.path.join(
            dataset_folder, sobp_folder_name, f"field{field_num}"
        ) # crea una carpeta para el campo llamada field{num_campo}
        os.makedirs(sobp_folder_location, exist_ok=True)
        fredinp_destination = os.path.join(
            sobp_folder_location, "fred.inp"
        )  # copy fred.inp intro new folder. Ruta y nombre del archivo que se va a copiar dentro de field
        shutil.copy(fredinp_location, fredinp_destination) #se copia el fred original y se pega en la ruta anterior
        df_field = df_rtplan[df_rtplan["field_num"] == field_num]  # Get the field data from the RTPLAN CSV file
        for pb in df_field.itertuples(): # iterate over the pencil beams in the field
            pb_energy = pb.energy
            pos_target_x = -pb.pos_target_x - CT_isocenter[0]/ 10  
            pos_target_z = pb.pos_target_z - CT_isocenter[2]/ 10  # Position of the target in the CT coordinates
            FWHM_x = pb.FWHMx
            FWHM_z = pb.FWHMz
            weight = pb.weight

            pos_target_spot = np.array([pos_target_x, 0., pos_target_z])
            pos_source_spot = np.array([0., beam_origin, 0.])  
            
            v = pos_target_spot - pos_source_spot  # vector from the source to the target
            v /= np.linalg.norm(v)  # normalize the vector
            vx, vy, vz = map(float, v)

            N = weight * (6.093 * pb_energy**3 - 7810 * pb_energy**2 + 5.281e6 * pb_energy + 6.248e7)
            total_delivered_particles += N  # total number of particles delivered in the field
            nprim = int(N / 1000.)  # reducing the number of primaries to 0.1% of the real value
            pencil_beam_line = (
                f"pb: {field_pb_num} origin; particle=proton; T={pb_energy:.6g}; Espread={float(Espread):.6g}; "
                f"v=[{vx:.15g}, {vy:.15g}, {vz:.15g}]; P=[0,0,0]; " # P = start position WRT field
                f"Xsec=gauss; FWHMx={float(FWHM_x):.6g}; FWHMy={float(FWHM_z):.6g}; "
                f"nprim={nprim}; N={N:.6g};"
            ) #genera una linea para fred.inp con la informacion del pencil beam.

            field_pb_num += 1
            plan_pb_num += 1
            pencil_beams.append(pencil_beam_line)

        with open(fredinp_destination, "a", encoding="utf-8") as file:
            file.write("\n".join(pencil_beams)) #se escribe la linea que se habia creado con la informacion del pencil beam.
            file.write("\n")
            file.writelines(schneider_lines) #luego se escribe toda la infomracion del schneider, que contiene los datos del material segun regiones

        # Crop and delete larger files
        # mhd_folder_path = os.path.join(sobp_folder_location, "out/reg/Phantom")  # For FRED v 3.6
        mhd_folder_path = os.path.join(
            sobp_folder_location, "out/score"
        )  # For FRED v 3. Carpeta donde se va a guardar la activación de los isotopos

        # Dose
        # dose_file_path = os.path.join(mhd_folder_path, 'Dose.mhd')  # For FRED v 3.6
        dose_file_path = os.path.join(
            mhd_folder_path, "Phantom.Dose.mhd" #despues de ejecutar fred se ha creado un documento .mhd con la dosis. Formato ID.Dose.mhd
        )  # For FRED v 3. La dosis se va a guardar en un documento Phantom-Dose.mhd dentro de la carpeta out/score

        force_simulation = True  # if True, simulate the treatment, if False, use the existing dose file
        if not os.path.exists(dose_file_path) or force_simulation:  # Simulate treatment if not already done
            print("Simulating treatment...")
            # time the simulation
            start_time = time.time()
            # Execute fred
            command = ["fred"]
            subprocess.run(command, cwd=sobp_folder_location) # se ejecuta fred. es como ejecutarlo desde la terminal.
            end_time = time.time()
            print(f"Simulation completed in {end_time - start_time} seconds.")
            # save the timing to a file
            with open(os.path.join(sobp_folder_location, "simulation_time.txt"), "w") as f:
                f.write(f"Simulation time: {end_time - start_time} seconds\n")
        else:
            print("Dose file already exists, skipping simulation.")

        # save dcm
        convert_mhd_to_dcm(dose_file_path, os.path.join(dataset_folder, "dose/dcm_file.dcm"),
                           dcm_template=rtdose,
                           CT_dcm_template=os.path.join(CT_dcm_dir, "CT1.3.12.2.1107.5.1.4.78660.30000021021512515990600000989.dcm"))  # Save the dose as a DICOM file, using a template DICOM file

        total_dose += crop_resize_save(
            dose_file_path,
            os.path.join(dataset_folder, f"dose/sobp{sobp_num}.npy"),  # Path to save the cropped dose
            final_voxel_size=voxel_size,
            final_shape=cropped_shape
        )
        
        total_uncropped_dose += crop_resize_save(
            dose_file_path,
            os.path.join(dataset_folder, f"dose/sobp{sobp_num}-uncropped.npy"),  # Path to save the uncropped dose
            final_voxel_size=voxel_size,
        )

        remaining_fields = num_fields - field_num - 1

    scaling_factor = 1.1  #biological equivalent dose scaling factor
    
    # save the total delivered particles as a txt
    with open(os.path.join(sobp_folder_location, "delivered_particles.txt"), "w") as f:
        f.write(f"Total delivered particles: {total_delivered_particles}\n")
    
    # Gamma analysis
    distance_mm_threshold = 3.0  # mm
    lower_percent_dose_cutoff = 10  # 10% of the maximum dose
    planned_dose, origin = load_rtdose(rtdose)  
    planned_dose /= 10  # 10 fractions
    
    # removing couch and other stuff from the planned dose and the total uncropped dose
    planned_dose[:, -35:, :] = 0.# removing the last rows, which correspond to the couch
    total_uncropped_dose[:, -65:, :] = 0.
    total_uncropped_dose *= scaling_factor  # biological equivalent dose
    
    # save dcm
    convert_mhd_to_dcm(dose_file_path, os.path.join(dataset_folder, "dose/RD_PHANTOM_FRED.dcm"),
                        dcm_template=rtdose,
                        CT_dcm_template=os.path.join(CT_dcm_dir, "CT1.3.12.2.1107.5.1.4.78660.30000021021512515990600000989.dcm"),
                        img=total_uncropped_dose,
                        voxel_size=voxel_size)  # Save the cropped dose as a DICOM file, using a template DICOM file

    air_mask = np.zeros(CT_uncropped.shape, dtype=bool)  # mask for the air
    air_mask[CT_uncropped < -900] = True  # air
    total_uncropped_dose[air_mask] = 0.  # removing the air from the total uncropped dose
    
    uncropped_dose_axes_reference = (
        voxel_size[0] * np.arange(total_uncropped_dose.shape[0]) + CT_origin[0],
        - voxel_size[1] * np.arange(total_uncropped_dose.shape[1]) - CT_origin[1],
        voxel_size[2] * np.arange(total_uncropped_dose.shape[2]) + CT_origin[2]
    ) 
    
    planned_dose_axes_reference = (
        voxel_size[0] * np.arange(planned_dose.shape[0]) + origin[0], 
        - voxel_size[1] * np.arange(planned_dose.shape[1]) - origin[1],
        voxel_size[2] * np.arange(planned_dose.shape[2]) + origin[2]
    )

    # Plot the histogram of the planned dose and the total uncropped dose with max = 2.3Gy
    plt.figure(figsize=(10, 8))
    plt.hist(planned_dose[planned_dose > 1e-2].flatten(), bins=100, alpha=0.7, label='Planned Dose', color='red', range=(0, 2.3))
    plt.hist(total_uncropped_dose[total_uncropped_dose > 1e-2], bins=100, alpha=0.7, label='Simulated Dose', color='blue', range=(0, 2.3))
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(images_folder, "dose_histogram.png"), bbox_inches="tight")

    print(f"Planned dose shape: {planned_dose.shape}, min: {np.min(planned_dose)}, max: {np.max(planned_dose)}, origin: {origin}, 99th percentile: {np.percentile(planned_dose, 99)}")
    print(f"Total uncropped dose shape: {total_uncropped_dose.shape}, min: {np.min(total_uncropped_dose)}, max: {np.max(total_uncropped_dose)}, origin: {CT_origin}, 99th percentile: {np.percentile(total_uncropped_dose, 99)}")
    # plot planned dose and total_uncropped_dose side by side
    # SAGITTAL
    fig, ax = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(planned_dose[planned_dose.shape[0] // 2, :, :], cmap="Reds", alpha=0.8, 
               extent=(
                       origin[2], origin[2] + planned_dose.shape[2] * voxel_size[2],
                       -(origin[1] + planned_dose.shape[1] * voxel_size[1]), -origin[1]))
    im2 = plt.imshow(total_uncropped_dose[total_uncropped_dose.shape[0] // 2, :, :], cmap="Blues", alpha=0.6,
                extent=(CT_origin[2], CT_origin[2] + total_uncropped_dose.shape[2] * voxel_size[2],
                        -(CT_origin[1] + total_uncropped_dose.shape[1] * voxel_size[1]), -CT_origin[1]))
    im3 = plt.imshow(CT_uncropped[CT_uncropped.shape[0] // 2, :, :], cmap="gray", alpha=0.4,
                extent=(CT_origin[2], CT_origin[2] + total_uncropped_dose.shape[2] * voxel_size[2],
                        -(CT_origin[1] + total_uncropped_dose.shape[1] * voxel_size[1]), -CT_origin[1]))
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    cbar_ax1 = fig.add_axes([0.9, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax1, label='Planned')
    cbar_ax2 = fig.add_axes([0.9, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label='Simulated')
    plt.tight_layout(rect=[0, 0, 0.96, 1])
    plt.savefig(os.path.join(images_folder, "planned_vs_simulated_dose_phantom_sagittal.png"), bbox_inches="tight")

    # # Plot the dose difference with a diverging colormap
    reference_coordinates = uncropped_dose_axes_reference
    reference_dose = total_uncropped_dose
    
    # Interpolate uncropped dose onto planned dose grid
    interpolator = RegularGridInterpolator(
        planned_dose_axes_reference, 
        planned_dose, 
        method='linear', 
        bounds_error=False, 
        fill_value=0.0
    )
    
    # Create meshgrid for planned dose coordinates
    X, Y, Z = np.meshgrid(
        uncropped_dose_axes_reference[0],
        uncropped_dose_axes_reference[1], 
        uncropped_dose_axes_reference[2],
        indexing='ij'
    )
    
    # Points to interpolate at
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Interpolate uncropped dose
    interpolated_uncropped = interpolator(points).reshape(total_uncropped_dose.shape)
    # Calculate difference (uncropped - planned)
    dose_difference = interpolated_uncropped - reference_dose
    print(f"Dose difference shape: {dose_difference.shape}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(dose_difference[dose_difference.shape[0] // 2, :, :], cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Dose difference (Gy)')
    plt.imshow(CT_uncropped[CT_uncropped.shape[0] // 2, :, :], cmap="gray", alpha=0.5,)
    plt.xticks([])
    plt.yticks([])
    plt.title('Dose difference (Simulated - Planned)')
    plt.savefig(os.path.join(images_folder, "dose_difference_sagittal.png"), bbox_inches="tight")

    # CORONAL
    fig, ax = plt.subplots(figsize=(10, 8))
    im1 = plt.imshow(planned_dose[:, 85, :], cmap="Reds", alpha=0.7, 
               extent=(origin[2], origin[2] + planned_dose.shape[2] * voxel_size[2],
                       origin[0], origin[0] + planned_dose.shape[0] * voxel_size[0]))
    im2 = plt.imshow(total_uncropped_dose[:, 134, :], cmap="Blues", alpha=0.7,
                extent=(CT_origin[2], CT_origin[2] + total_uncropped_dose.shape[2] * voxel_size[2],
                        CT_origin[0], CT_origin[0] + total_uncropped_dose.shape[0] * voxel_size[0]))
    im3 = plt.imshow(CT_uncropped[:, 134, :], cmap="gray", alpha=0.5,
                extent=(CT_origin[2], CT_origin[2] + total_uncropped_dose.shape[2] * voxel_size[2],
                        CT_origin[0], CT_origin[0] + total_uncropped_dose.shape[0] * voxel_size[0]))
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax1, label='planned')
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label='simulated')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(images_folder, "planned_vs_simulated_dose_phantom_coronal.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(dose_difference[:, 134, :], cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.colorbar(label='Dose difference (Gy)')
    plt.imshow(CT_uncropped[:, 134, :], cmap="gray", alpha=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('Dose difference (Simulated - Planned)')
    plt.savefig(os.path.join(images_folder, "dose_difference_coronal.png"), bbox_inches="tight")

    gamma_img_path = os.path.join(dataset_folder, "gamma_img.npy")
    force_gamma_recalculation = True  # Set to True to force recalculation of the gamma image
    if os.path.exists(gamma_img_path) and not force_gamma_recalculation:
        gamma_img = np.load(gamma_img_path)
    else:
        gamma_img = pymedphys.gamma(
            uncropped_dose_axes_reference,
            total_uncropped_dose,
            planned_dose_axes_reference,
            planned_dose,
            dose_percent_threshold=3,  # 3% dose difference
            distance_mm_threshold=distance_mm_threshold,  # 3 mm distance to agreement
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,  # only consider points with at least 10% of the maximum dose
            max_gamma=1.1,
        )
        np.save(gamma_img_path, gamma_img)  # Save the gamma image
        
    valid_gamma = gamma_img[~np.isnan(gamma_img)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    print("Pass ratio for sample is: ", pass_ratio)
    # print("AFTER GAMMA CALCULATION")
    gamma_pass_img = (gamma_img < 1.0) # & (
    #     ~np.isnan(gamma_img)
    # )  # plot only the gamma values above 1.0 (not passing)
    gamma_pass_img = gamma_pass_img.astype(np.int8)
    gamma_cmap = cm.ListedColormap(["red", "green"])
    red_patch = mpatches.Patch(color='red', label='Fail')
    green_patch = mpatches.Patch(color='green', label='Pass')
    gamma_mask = total_uncropped_dose < lower_percent_dose_cutoff/100 * np.max(total_uncropped_dose)
    gamma_pass_img = np.ma.masked_where(gamma_mask, gamma_pass_img)  # Mask the gamma pass image where

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_img[gamma_img.shape[0] // 2, :, :], cmap="jet", vmin=0, vmax=1.1)
    plt.axis("off")
    plt.colorbar(label="Gamma index")
    plt.savefig(os.path.join(images_folder, "gamma_analysis_phantom.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_pass_img[gamma_pass_img.shape[0] // 2, :, :], cmap=gamma_cmap, vmin=0, vmax=1)
    plt.imshow(CT_uncropped[CT_uncropped.shape[0] // 2, :, :], cmap="gray", alpha=0.5)
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.axis("off")
    plt.title(f"Gamma pass ratio: {pass_ratio:.2%}")
    plt.savefig(os.path.join(images_folder, "gamma_pass_analysis_phantom.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_img[10 + gamma_img.shape[0] // 2, :, :], cmap="jet", vmin=0, vmax=1.1)
    plt.axis("off")
    plt.colorbar(label="Gamma index")
    plt.savefig(os.path.join(images_folder, "gamma_analysis_phantom_sagittal_1.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_pass_img[10 + gamma_pass_img.shape[0] // 2, :, :], cmap=gamma_cmap, vmin=0, vmax=1)
    plt.imshow(CT_uncropped[10 + CT_uncropped.shape[0] // 2, :, :], cmap="gray", alpha=0.5)
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.axis("off")
    plt.title(f"Gamma pass ratio: {pass_ratio:.2%}")
    plt.savefig(os.path.join(images_folder, "gamma_pass_analysis_phantom_sagittal_1.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_img[gamma_img.shape[0] // 2 - 10, :, :], cmap="jet", vmin=0, vmax=1.1)
    plt.axis("off")
    plt.colorbar(label="Gamma index")
    plt.savefig(os.path.join(images_folder, "gamma_analysis_phantom_sagittal_2.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_pass_img[gamma_pass_img.shape[0] // 2 - 10, :, :], cmap=gamma_cmap, vmin=0, vmax=1)
    plt.imshow(CT_uncropped[CT_uncropped.shape[0] // 2 - 10, :, :], cmap="gray", alpha=0.5)
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.axis("off")
    plt.title(f"Gamma pass ratio: {pass_ratio:.2%}")
    plt.savefig(os.path.join(images_folder, "gamma_pass_analysis_phantom_sagittal_2.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_img[:, 10 + gamma_img.shape[1] // 2, :], cmap="jet", vmin=0, vmax=1.1)
    plt.axis("off")
    plt.colorbar(label="Gamma index")
    plt.savefig(os.path.join(images_folder, "gamma_analysis_phantom_coronal_2.png"), bbox_inches="tight")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_pass_img[:, 10 + gamma_pass_img.shape[1] // 2, :], cmap=gamma_cmap, vmin=0, vmax=1)
    plt.imshow(CT_uncropped[:, 10 + CT_uncropped.shape[1] // 2, :], cmap="gray", alpha=0.5)
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.axis("off")
    plt.title(f"Gamma pass ratio: {pass_ratio:.2%}")
    plt.savefig(os.path.join(images_folder, "gamma_pass_analysis_phantom_coronal_2.png"), bbox_inches="tight")

    plt.figure(figsize=(10, 8))
    plt.imshow(gamma_pass_img[:, 134, :], cmap=gamma_cmap, vmin=0, vmax=1)
    plt.imshow(CT_uncropped[:, 134, :], cmap="gray", alpha=0.5)
    plt.legend(handles=[red_patch, green_patch], loc='upper right')
    plt.axis("off")
    plt.title(f"Gamma pass ratio: {pass_ratio:.2%}")
    plt.savefig(os.path.join(images_folder, "gamma_pass_analysis_phantom_coronal.png"), bbox_inches="tight")

    # Scaling the dose to the target dose
    # scaling_factor = target_dose / np.percentile(total_dose, 99.99)
    # print(f"Scaling factor for dose: {scaling_factor}")
    total_dose = total_dose * scaling_factor

    # Cropping and saving:
    # Saving dose
    dose_npy_path = os.path.join(dataset_folder, f"dose/sobp{sobp_num}.npy")
    dose_raw_path = None  # os.path.join(mhd_folder_path, 'Dose.raw')
    total_dose = crop_save_npy(
        total_dose,
        dose_npy_path,
        raw_path=dose_raw_path,
        Trans=Trans,
        HL=final_shape // 2,
    )

    # 1) Dose overlaid on CT, masking out values < 1% of max
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    total_dose_max = total_dose.max()
    mask_threshold = 0.01 * total_dose_max
    # plot y
    mid_idx = CT_cropped.shape[1] // 2 + 25
    dose_slice = np.flip(total_dose[:, mid_idx, :].T, axis=0)
    dose_mask = dose_slice < mask_threshold

    masked_dose = np.ma.array(dose_slice, mask=dose_mask)
    ax[0].imshow(np.flip(CT_cropped[:, mid_idx, :].T, axis=0), cmap="gray")
    im_dose = ax[0].imshow(masked_dose, cmap="jet", alpha=0.7, vmax=total_dose_max)
    ax[0].set_title("Dose (coronal)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # plot z
    mid_idx = CT_cropped.shape[2] // 2
    dose_slice = total_dose[:, :, mid_idx].T
    mask_threshold = 0.01 * dose_slice.max()
    dose_mask = dose_slice < mask_threshold
    masked_dose = np.ma.array(dose_slice, mask=dose_mask)
    ax[1].imshow(CT_cropped[:, :, mid_idx].T, cmap="gray")
    im_dose = ax[1].imshow(masked_dose, cmap="jet", alpha=0.7, vmax=total_dose_max)
    ax[1].set_title("Dose (axial)")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # plot x
    mid_idx = CT_cropped.shape[0] // 2
    dose_slice = total_dose[mid_idx, :, :]
    mask_threshold = 0.01 * dose_slice.max()
    dose_mask = dose_slice < mask_threshold
    masked_dose = np.ma.array(dose_slice, mask=dose_mask)
    ax[2].imshow(CT_cropped[mid_idx, :, :], cmap="gray")
    im_dose = ax[2].imshow(masked_dose, cmap="jet", alpha=0.7, vmax=total_dose_max)
    ax[2].set_title("Dose (sagittal)")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    # add colorbar
    cbar = plt.colorbar(im_dose, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Dose (Gy)")

    plt.savefig(os.path.join(images_folder, "plot_doses.png"))
    