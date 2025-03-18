import os
import pydicom
import numpy as np
import pandas as pd
from skimage.transform import resize
from scipy.ndimage import zoom

# Load DICOM files and metadata
def load_dicom_files(dicom_folder):
    slices = []
    dicom_metadata = {}
    for dicom_file in sorted(os.listdir(dicom_folder)):
        if dicom_file.endswith('.dcm'):
            dicom_path = os.path.join(dicom_folder, dicom_file)
            try:
                ds = pydicom.dcmread(dicom_path)
                slices.append(ds.pixel_array)
                if not dicom_metadata:
                    dicom_metadata = {
                        'PatientsSex': getattr(ds, 'PatientSex', None),
                        'PatientsAge': getattr(ds, 'PatientAge', None),
                        'SliceThickness': getattr(ds, 'SliceThickness', 1.0),
                        'SpacingBetweenSlices': getattr(ds, 'SpacingBetweenSlices', 1.0),
                        'MagneticFieldStrength': getattr(ds, 'MagneticFieldStrength', None),
                        'PixelSpacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
                        'Manufacturer': getattr(ds, 'Manufacturer', None),
                        'DeviceSerialNumber': getattr(ds, 'DeviceSerialNumber', None)
                    }
            except Exception as e:
                print(f"Error reading {dicom_path}: {e}")
                continue  # Skip this file and continue with the next one

    if slices:
        slices = np.stack(slices, axis=0)
        return slices, dicom_metadata
    #if slices:
    #    slices = np.stack(slices, axis=0)
    #    slice_thickness = dicom_metadata['SliceThickness']
    #    spacing_between_slices = dicom_metadata['SpacingBetweenSlices']
    #    pixel_spacing = dicom_metadata['PixelSpacing']
    #    spacing = np.array([spacing_between_slices] + list(pixel_spacing))
    #    new_spacing = np.array([1.0, 1.0, 1.0])
    #    resize_factor = spacing / new_spacing
    #    resampled_slices = zoom(slices, resize_factor, order=1)
    #    return resampled_slices, dicom_metadata
    else:
        return None, dicom_metadata

def extract_metadata(sequence_folder):
    view = "unknown"
    sequence_type = "unknown"
    fat_sat = "No"
    pd_val = "No"
    spgr = "No"
    view_keywords = {"AX": "axial", "Ax": "axial", "ax": "axial", "axial": "axial", "Axial": "axial", "AXIAL": "axial", 
                     "transverse": "axial", "Transverse": "axial", "TRANSVERSE": "axial", "tra": "axial", "Tra": "axial", "TRA": "axial",
                     "SAG": "sagittal", "Sag": "sagittal", "sag": "sagittal",
                     "COR": "coronal", "Cor": "coronal", "cor": "coronal",
                     "ABERS": "ABERS",  "Abers": "ABERS",  "abers": "ABERS", "ABER": "ABERS", "Aber": "ABERS", "aber": "ABERS", 
                     "LOC": "3-plane", "3_PLANE": "3-plane", "3-PLANE": "3-plane", "3-PL": "3-plane", "SURVEY": "3-plane", "localizer" : "3-plane"}
    sequence_keywords = {"_T2_": "T2", "_t2_": "T2", "_T1_": "T1", "_t1_": "T1"}
    for key, value in view_keywords.items():
        if key in sequence_folder:
            view = value
            break
    for key, value in sequence_keywords.items():
        if key in sequence_folder:
            sequence_type = value
            break
    if "_FS_" in sequence_folder or "_Fs_" in sequence_folder or "_fs_" in sequence_folder:
        fat_sat = "Yes"
    if "_PD_" in sequence_folder or "_Pd_" in sequence_folder or "_pd_" in sequence_folder:
        pd_val = "Yes"
    if "_SPGR_" in sequence_folder or "_Spgr_" in sequence_folder or "_spgr_" in sequence_folder:
        spgr = "Yes"
    return view, sequence_type, fat_sat, pd_val, spgr

def process_mri_data(base_folder, batch_size=10, save_folder='/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/preprocessed_MRIs'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_records = []
    counter = 0
    for patient_folder in os.listdir(base_folder):
        patient_folder_path = os.path.join(base_folder, patient_folder)
        if os.path.isdir(patient_folder_path):
            for mri_folder in os.listdir(patient_folder_path):
                mri_folder_path = os.path.join(patient_folder_path, mri_folder)
                if os.path.isdir(mri_folder_path):
                    parts = mri_folder.split('_')
                    if len(parts) > 1:
                        mri_id = parts[1]
                        mri_type = parts[-1]
                    else:
                        continue
                    mri_save_folder = os.path.join(save_folder, mri_id)
                    if not os.path.exists(mri_save_folder):
                        os.makedirs(mri_save_folder)
                    for sequence_folder in os.listdir(mri_folder_path):
                        sequence_folder_path = os.path.join(mri_folder_path, sequence_folder)
                        if os.path.isdir(sequence_folder_path):
                            pixel_array, dicom_metadata = load_dicom_files(sequence_folder_path)
                            if pixel_array is not None:
                                view, sequence_type, fat_sat, pd_val, spgr = extract_metadata(sequence_folder)
                                unique_id = f"{patient_folder}_{mri_id}_{sequence_folder}"
                                record = {'patient_folder': patient_folder, 'mri_id': mri_id, 'mri_type': mri_type,
                                          'sequence_folder': sequence_folder, 'view': view, 'sequence_type': sequence_type,
                                          'fat_sat': fat_sat, 'PD': pd_val, 'SPGR': spgr, 'PatientsSex': dicom_metadata.get('PatientsSex'),
                                          'PatientsAge': dicom_metadata.get('PatientsAge'), 'SliceThickness': dicom_metadata.get('SliceThickness'),
                                          'MagneticFieldStrength': dicom_metadata.get('MagneticFieldStrength'),
                                          'SpacingBetweenSlices': dicom_metadata.get('SpacingBetweenSlices'), 
                                          'Manufacturer': dicom_metadata.get('Manufacturer'), 
                                          'DeviceSerialNumber': dicom_metadata.get('DeviceSerialNumber'), 'unique_id': unique_id}
                                data_records.append(record)
                                np.save(os.path.join(mri_save_folder, f'{unique_id}.npy'), pixel_array)
                                counter += 1
                                if counter >= batch_size:
                                    print(f'Processed {counter} MRI volumes...')
                                    counter = 0
    df = pd.DataFrame(data_records)
    return df

# HPC Filepaths
base_folder = '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/extracted_MRIs'
save_folder = '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs'

#Local Filepaths
# base_folder = '/Users/sahilsethi/Desktop/Ortho ML Labrum Tears/MRIs'
# save_folder = '/Users/sahilsethi/Desktop/Ortho ML Labrum Tears/preprocessed_MRIs'

mri_data_df = process_mri_data(base_folder, save_folder=save_folder)

mri_data_df['PatientsAge'] = mri_data_df['PatientsAge'].apply(lambda x: int(x[:-1]) if isinstance(x, str) and x.endswith('Y') else x)
mri_data_df['laterality'] = mri_data_df['mri_type'].apply(lambda x: 'R' if x.endswith('RT') else ('L' if x.endswith('LT') else 'unknown'))

# Save the metadata dataframe to a CSV file
mri_data_df.to_csv(os.path.join(save_folder, 'metadata.csv'), index=False)

print("All MRI data has been processed and saved.")

import os
import pandas as pd
import shutil

# Paths
preprocessed_folder = save_folder
semi_external_folder = os.path.join(preprocessed_folder, 'semiexternal_validation')
os.makedirs(semi_external_folder, exist_ok=True)

# Load the list of semiexternal mri_ids
list_filepath = '/gpfs/data/orthopedic-lab/list_of_semiexternal.xlsx'
semiexternal_ids_df = pd.read_excel(list_filepath)
semiexternal_ids = semiexternal_ids_df['mri_id'].tolist()

# Load metadata
metadata_path = os.path.join(preprocessed_folder, 'metadata.csv')
metadata_df = pd.read_csv(metadata_path)

# Function to move files to semi-external validation folder
def move_to_semi_external(mri_ids, source_folder, destination_folder):
    for mri_id in mri_ids:
        mri_id_folder = os.path.join(source_folder, mri_id)
        destination_mri_id_folder = os.path.join(destination_folder, mri_id)
        if os.path.exists(mri_id_folder):
            shutil.move(mri_id_folder, destination_mri_id_folder)
            print(f"Moved {mri_id_folder} to {destination_mri_id_folder}")
        else:
            print(f"Folder not found: {mri_id_folder}")

# Execute the function
move_to_semi_external(semiexternal_ids, preprocessed_folder, semi_external_folder)

# Update metadata to remove moved entries
semiexternal_metadata_df = metadata_df[metadata_df['mri_id'].isin(semiexternal_ids)]
metadata_df = metadata_df[~metadata_df['mri_id'].isin(semiexternal_ids)]

# Save the updated metadata files
semiexternal_metadata_path = os.path.join(semi_external_folder, 'metadata.csv')
semiexternal_metadata_df.to_csv(semiexternal_metadata_path, index=False)
metadata_df.to_csv(metadata_path, index=False)

print("Semi-external validation set separation complete.")
