import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import shutil
import argparse

def load_metadata(base_folder):
    metadata_path = os.path.join(base_folder, 'metadata_filtered.csv')
    return pd.read_csv(metadata_path)

def load_labels(filepath, label_columns):
    labels_df = pd.read_excel(filepath)
    return labels_df[['Anon Accession'] + label_columns]

def filter_mri_data(mri_data_df, labels_df, label_columns, views, sequence_types, fat_sat, pd, spgr, sex, min_age, max_age, magnetic_field_strength):
    merged_df = mri_data_df.merge(labels_df, left_on='mri_id', right_on='Anon Accession', how='inner')
    
    if views != "all":
        views = views.split(',')
        merged_df = merged_df[merged_df['view'].isin(views)]
    
    if sequence_types != "all":
        sequence_types = sequence_types.split(',')
        merged_df = merged_df[merged_df['sequence_type'].isin(sequence_types)]
    
    if fat_sat != "all":
        merged_df = merged_df[merged_df['fat_sat'] == fat_sat]
    
    if pd != "all":
        merged_df = merged_df[merged_df['PD'] == pd]
    
    if spgr != "all":
        merged_df = merged_df[merged_df['SPGR'] == spgr]
    
    if sex != "all":
        merged_df = merged_df[merged_df['PatientsSex'] == sex]
    
    if min_age != "all":
        merged_df = merged_df[merged_df['PatientsAge'] >= int(min_age)]
    
    if max_age != "all":
        merged_df = merged_df[merged_df['PatientsAge'] <= int(max_age)]
    
    if magnetic_field_strength != "all":
        merged_df = merged_df[merged_df['MagneticFieldStrength'] == float(magnetic_field_strength)]

    return merged_df

def stratified_split_data_by_mri_id(mri_data_df, label_columns, test_size=0.2, val_size=0.1, random_state=None):
    mri_ids = mri_data_df['mri_id'].unique()
    labels = mri_data_df.drop_duplicates(subset='mri_id')[label_columns].values

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(stratified_split.split(mri_ids, labels))
    
    train_val_ids = mri_ids[train_indices]
    test_ids = mri_ids[test_indices]

    labels_train_val = labels[train_indices]
    
    stratified_split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_indices, val_indices = next(stratified_split_val.split(train_val_ids, labels_train_val))

    train_ids = train_val_ids[train_indices]
    val_ids = train_val_ids[val_indices]

    train_df = mri_data_df[mri_data_df['mri_id'].isin(train_ids)]
    val_df = mri_data_df[mri_data_df['mri_id'].isin(val_ids)]
    test_df = mri_data_df[mri_data_df['mri_id'].isin(test_ids)]
    return train_df, val_df, test_df

def copy_files_by_mri_id(df, source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for mri_id in df['mri_id'].unique():
        mri_id_folder = os.path.join(source_folder, mri_id)
        target_mri_id_folder = os.path.join(target_folder, mri_id)
        if os.path.exists(mri_id_folder):
            unique_ids = df[df['mri_id'] == mri_id]['unique_id'].values
            for unique_id in unique_ids:
                npy_file = os.path.join(mri_id_folder, f"{unique_id}.npy")
                if os.path.exists(npy_file):
                    if not os.path.exists(target_mri_id_folder):
                        os.makedirs(target_mri_id_folder)
                    shutil.copy(npy_file, target_mri_id_folder)
                else:
                    print(f'npy file not found: {npy_file}')
        else:
            print(f'Folder not found: {mri_id_folder}')

def main(args):
    mri_data_df = load_metadata(args.base_folder)
    labels_df = load_labels(args.labels_filepath, [args.label_column1, args.label_column2, args.label_column3])

    filtered_mri_data_df = filter_mri_data(mri_data_df, labels_df, [args.label_column1, args.label_column2, args.label_column3], args.views, args.sequence_types, args.fat_sat, args.pd, args.spgr, args.sex, args.min_age, args.max_age, args.magnetic_field_strength)

    train_df, val_df, test_df = stratified_split_data_by_mri_id(filtered_mri_data_df, [args.label_column1, args.label_column2, args.label_column3])

    output_folder = os.path.join(args.base_folder, args.output_subfolder)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    copy_files_by_mri_id(train_df, args.base_folder, train_folder)
    copy_files_by_mri_id(val_df, args.base_folder, val_folder)
    copy_files_by_mri_id(test_df, args.base_folder, test_folder)

    train_df.to_csv(os.path.join(train_folder, 'metadata.csv'), index=False)
    val_df.to_csv(os.path.join(val_folder, 'metadata.csv'), index=False)
    test_df.to_csv(os.path.join(test_folder, 'metadata.csv'), index=False)

    print("Data has been filtered and split into train, val, and test sets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MRI data into training, validation, and testing sets.")
    parser.add_argument('--base_folder', type=str, required=True, help='Base folder containing the preprocessed MRIs')
    parser.add_argument('--labels_filepath', type=str, required=True, help='Filepath to the Excel file containing labels')
    parser.add_argument('--label_column1', type=str, required=True, help='Column name for the first label in the Excel file')
    parser.add_argument('--label_column2', type=str, required=True, help='Column name for the second label in the Excel file')
    parser.add_argument('--label_column3', type=str, required=True, help='Column name for the third label in the Excel file')
    parser.add_argument('--views', type=str, default='all', help='Comma-separated list of views to include, or "all" to include all views')
    parser.add_argument('--sequence_types', type=str, default='all', help='Comma-separated list of sequence types to include, or "all" to include all sequence types')
    parser.add_argument('--fat_sat', type=str, default='all', help='Filter by fat saturation (Yes, No, or all)')
    parser.add_argument('--pd', type=str, default='all', help='Filter by PD (Yes, No, or all)')
    parser.add_argument('--spgr', type=str, default='all', help='Filter by SPGR (Yes, No, or all)')
    parser.add_argument('--sex', type=str, default='all', help='Filter by patient sex (M, F, or all)')
    parser.add_argument('--min_age', type=str, default='all', help='Minimum patient age to include, or "all" for no minimum')
    parser.add_argument('--max_age', type=str, default='all', help='Maximum patient age to include, or "all" for no maximum')
    parser.add_argument('--magnetic_field_strength', type=str, default='all', help='Magnetic field strength to include, or "all" to include all')
    parser.add_argument('--output_subfolder', type=str, required=True, help='Subfolder within the base folder to save the train, val, and test sets')
    args = parser.parse_args()

    main(args)
