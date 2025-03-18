import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage import exposure, filters
import random
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandFlipd, RandRotated, RandGaussianNoised, RandScaleIntensityd, RandAffined, RandGridDistortiond
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split

SEED = 42  # Or any fixed seed value
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_determinism(seed=SEED)  #seed MONAI transforms


if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Define a global worker initialization function
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create a global generator for consistency
g = torch.Generator()
g.manual_seed(SEED)

##MRNet dataset loader code
def extract_sequence_metadata(view, magnet_strength='3T'):
    metadata = {
        '3T': {
            'Cor T1': {'SliceThickness': 2.5, 'SectGap': 0, 'FOV': 150, 'AcquisitionMatrix': [416, 224]},
            'Cor PD FS': {'SliceThickness': 2.5, 'SectGap': 0, 'FOV': 150, 'AcquisitionMatrix': [384, 224]},
            'Sag PD': {'SliceThickness': 2.5, 'SectGap': 0, 'FOV': 150, 'AcquisitionMatrix': [384, 224]},
            'Sag T2 FS': {'SliceThickness': 2.5, 'SectGap': 0, 'FOV': 150, 'AcquisitionMatrix': [384, 192]},
            'Ax PD FS': {'SliceThickness': 3, 'SectGap': 0.3, 'FOV': 150, 'AcquisitionMatrix': [512, 224]}
        },
        '1.5T': {
            'Cor T1': {'SliceThickness': 4, 'SectGap': 1, 'FOV': 160, 'AcquisitionMatrix': [512, 192]},
            'Cor T2 FS': {'SliceThickness': 4, 'SectGap': 1, 'FOV': 160, 'AcquisitionMatrix': [448, 192]},
            'Sag PD': {'SliceThickness': 3, 'SectGap': 1, 'FOV': 160, 'AcquisitionMatrix': [512, 192]},
            'Sag T2 FS': {'SliceThickness': 3.5, 'SectGap': 0.5, 'FOV': 160, 'AcquisitionMatrix': [448, 192]},
            'Ax PD FS': {'SliceThickness': 3.5, 'SectGap': 0.5, 'FOV': 130, 'AcquisitionMatrix': [256, 192]}
        }
    }

    return metadata[magnet_strength].get(view, {'SliceThickness': 2.5, 'SectGap': 0, 'FOV': 150, 'AcquisitionMatrix': [512, 512]})

def calculate_pixel_spacing(fov, acquisition_matrix):
    return [fov / acquisition_matrix[0], fov / acquisition_matrix[1]]

def resize_mri_volume(volume, slice_thickness, sect_gap, target_z, target_shape=(224, 224, 224)):
    # Calculate new_spacing and resize_factor
    spacing_between_slices = slice_thickness + sect_gap
    original_z = volume.shape[0]
    original_spacing_z = spacing_between_slices
    new_spacing_z = (original_z * original_spacing_z) / target_z
    resize_factor_z = new_spacing_z / original_spacing_z
    
    # Resample volume along the z-axis
    resampled_volume = zoom(volume, [resize_factor_z, 1, 1], order=1)
    
    # Resize to target shape
    resized_volume = resize(resampled_volume, target_shape, anti_aliasing=True)
    
    return resized_volume

def determine_metadata(example_mri_path):
    if "axial" in example_mri_path:
        view = 'Ax PD FS'
    elif "sagittal" in example_mri_path:
        view = 'Sag T2 FS'
    elif "coronal" in example_mri_path:
        view = 'Cor T1'
    else:
        raise ValueError("Unknown MRI view type in path")

    # Load the example MRI to determine the magnet strength
    example_volume = np.load(example_mri_path)
    num_slices = example_volume.shape[0]
    magnet_strength = '1.5T' if num_slices < 40 else '3T'

    return example_volume, view, magnet_strength

# Helper function to load labels from CSV
def load_labels(base_folder, target):
    train_labels = pd.read_csv(os.path.join(base_folder, f'train-{target}.csv'), header=None, names=['mri_id', target])
    test_labels = pd.read_csv(os.path.join(base_folder, f'valid-{target}.csv'), header=None, names=['mri_id', target])
    return train_labels, test_labels

# Helper function to load and preprocess MRIs
def load_and_preprocess_mri(base_folder, mri_id, view, is_train=True, target_shape=(224, 224, 224), use_file_column=False, labels_df=None, idx=None):
    sequence_metadata = {
        'axial': 'Ax PD FS',
        'sagittal': 'Sag T2 FS',
        'coronal': 'Cor T1'
    }
    
    # Use the file path from the DataFrame if in cross-validation mode
    if use_file_column:
        example_mri_path = labels_df.iloc[idx]['file_path']
    else:
        set_type = 'train' if is_train else 'valid'
        example_mri_path = os.path.join(base_folder, set_type, view, f"{mri_id:04d}.npy")

    view_type = sequence_metadata[view]
    
    try:
        # Verify path existence
        if not os.path.exists(example_mri_path):
            print(f"File not found: {example_mri_path}")
            return None

        # Determine the metadata based on the path and MRI data
        example_volume, view, magnet_strength = determine_metadata(example_mri_path)
        
        # Extract metadata for the given view and magnet strength
        metadata = extract_sequence_metadata(view_type, magnet_strength)

        # Resize the volume
        target_z = example_volume.shape[0]
        resized_volume = resize_mri_volume(example_volume, metadata['SliceThickness'], metadata['SectGap'], target_z, target_shape)

        return resized_volume
    
    except Exception as e:
        print(f"An error occurred while processing {example_mri_path}: {e}")
    
    return None

#Define transforms for MRNetDataset3D
transform = Compose([
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Vertical flip
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # Horizontal flip
    RandRotated(keys=["image"], prob=0.5, range_x=5 * 3.1415926535 / 180, range_y=0, range_z=0, mode="bilinear")  # Rotation in HxW plane
])

# Define a dataset class for MRNet data
class MRNetDataset3D(Dataset):
    def __init__(self, base_folder, labels_df, view, target_label, is_train=True, target_shape=(224, 224, 224), augment=False, augment_factor=20, transform_val=True, return_unique_id=False, use_file_column=False):
        self.base_folder = base_folder
        self.labels_df = labels_df
        self.view = view
        self.target_label = target_label
        self.is_train = is_train
        self.target_shape = target_shape
        self.augment = augment
        self.augment_factor = augment_factor
        self.transform = transform_val
        self.return_unique_id = return_unique_id
        self.use_file_column = use_file_column

    def __len__(self):
        return len(self.labels_df) * (self.augment_factor if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            idx = idx % len(self.labels_df)
        mri_id = self.labels_df.iloc[idx]['mri_id']
        label = self.labels_df.iloc[idx][self.target_label]
        volume = load_and_preprocess_mri(self.base_folder, mri_id, self.view, self.is_train, target_shape=self.target_shape, use_file_column=self.use_file_column, labels_df=self.labels_df, idx=idx)
        
        if volume is None:
            print(f"Skipping MRI ID {mri_id} due to loading issue.")
            return self.__getitem__((idx + 1) % len(self))
        
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            volume = self.apply_transforms(volume)
        
        if self.return_unique_id:
            unique_id = mri_id
            return volume, torch.tensor(label, dtype=torch.float32), mri_id, unique_id
        else:
            return volume, torch.tensor(label, dtype=torch.float32)

    def apply_transforms(self, volume):
        transformed = transform({"image": volume})
        return transformed["image"]


def create_stratified_validation_set(train_labels, val_size=120, target_label='abnormal'):
    # Stratified split
    train_labels, val_labels = train_test_split(train_labels, test_size=val_size, stratify=train_labels[target_label], random_state=SEED)
    return train_labels, val_labels

class MRNetDataset2D(Dataset):
    def __init__(self, base_folder, labels_df, view, label_column, is_train=True, augment=False, augment_factor=1, transform_val=True, return_unique_id=False, use_file_column=False):
        self.base_folder = base_folder
        self.labels_df = labels_df
        self.view = view
        self.label_column = label_column
        self.is_train = is_train
        self.augment = augment
        self.augment_factor = augment_factor
        self.transform_val = transform_val
        self.return_unique_id = return_unique_id
        self.use_file_column = use_file_column

        # Define optional transforms for augmentation
        self.transform = Compose([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Vertical flip
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # Horizontal flip
            RandRotated(keys=["image"], prob=0.5, range_x=7.5 * np.pi / 180, range_y=0, range_z=0, mode="bilinear"),
            RandScaleIntensityd(keys=["image"], prob=0.5, factors=(0.975, 1.025)),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.05),
        ])

        print(f"Initialized MRNet 2D Dataset with {len(labels_df)} samples for view {view}")

    def __len__(self):
        return len(self.labels_df) * (self.augment_factor if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            idx = idx % len(self.labels_df)
        mri_id = self.labels_df.iloc[idx]['mri_id']
        label = self.labels_df.iloc[idx][self.label_column]

        # Use the file path from the DataFrame if in cross-validation mode
        if self.use_file_column:
            volume_path = self.labels_df.iloc[idx]['file_path']
        else:
            set_type = 'train' if self.is_train else 'valid'
            volume_path = os.path.join(self.base_folder, set_type, self.view, f"{mri_id:04d}.npy")

        # Load the full 3D volume
        volume = np.load(volume_path)  # Shape: S x H x W

        #Define resize dim
        input_dim = 224
        resize_dim = 256

        # Resize each slice
        slices_resized = [resize(slice, (resize_dim, resize_dim), anti_aliasing=True) for slice in volume]

        # Stack slices into a 3D volume (S x 224 x 224)
        volume_resized = np.stack(slices_resized, axis=0)

        # Crop the middle region to 224x224
        pad = int((resize_dim - input_dim) / 2)
        volume_cropped = volume_resized[:, pad:pad + input_dim, pad:pad + input_dim]

        # Normalize the volume
        volume_normalized = (volume_cropped - volume_cropped.min()) / (volume_cropped.max() - volume_cropped.min())

        # Convert to tensor
        volume = torch.FloatTensor(volume_normalized).unsqueeze(0)  # Add channel dimension (1 x S x 224 x 224)

        # Apply optional transforms
        if self.augment or self.transform_val:
            volume = self.apply_transforms(volume)

        if self.return_unique_id:
            unique_id = mri_id
            return volume, torch.tensor(label, dtype=torch.float32), mri_id, unique_id
        else:
            return volume, torch.tensor(label, dtype=torch.float32)

    def apply_transforms(self, volume):
        transformed = self.transform({"image": volume}) if self.transform else {"image": volume}
        return transformed["image"]

###Original labrum loader code

# Constants for normalization
INPUT_DIM = 224
RESIZE_DIM = 400 #256

# Values for mean and stddev pixel intensities for each MRI type
MEANS_STDEVS = {
    "MERGE_No": {"mean": 516.1850844194399, "stddev": 578.3723950290351
},
    "T1_Yes": {"mean": 800.141442515429, "stddev": 1115.3957140601867
},
    "T1_No": {"mean": 927.7411116373514, "stddev": 1133.2128776878026
},
    "T2_Yes": {"mean": 319.12482550820556, "stddev": 317.53349217471504},
    "T2_No": {"mean": 206.88264083198587, "stddev": 375.577692978832
},
    "PD_Yes": {"mean": 604.8688651646372, "stddev": 657.3120444814552
},
    "PD_No": {"mean": 1024.1772106340304, "stddev": 1171.7056582395717
},
    "STIR_No": {"mean": 206.88570459409667, "stddev": 199.68716774895478
},
    "overall": {"mean": 686.1778893877388, "stddev": 927.4920803859206
}
}

# Function to load metadata from a specific subfolder
def load_metadata(subfolder, folder):
    metadata_path = os.path.join(folder, subfolder, 'metadata.csv')
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        print(f"Metadata file not found at {metadata_path}")
        return None

# Define preprocessing functions
def denoise_image(image, method='gaussian', sigma=1):
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma)
    elif method == 'median':
        from skimage.filters import median
        return median(image)
    elif method == 'non_local_means':
        from skimage.restoration import denoise_nl_means, estimate_sigma
        sigma_est = np.mean(estimate_sigma(image, multichannel=False))
        patch_kw = dict(patch_size=5, patch_distance=6, multichannel=False)
        return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
    return image

def normalize_volume(volume):
    volume = volume.astype('float32')
    min_val = np.min(volume)
    max_val = np.max(volume)
    volume = (volume - min_val) / (max_val - min_val)
    return volume

def apply_clahe(image, clip_limit=0.005):
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)

#Define 3D dataset class
class MRIDataset3D(Dataset):
    def __init__(self, df, base_folder, view, label_column, sequence_type='all', fat_sat='all', contrast_or_no='all', target_shape=(224, 224, 224), normalize=True, enhance_contrast=False, clip_limit = 0.005, denoise=False, denoise_method='gaussian', sigma=1, augment=False, augment_factor=1, augment_factor_0=1, transform_val=True, return_unique_id=False, use_file_column=False):
        if sequence_type != 'all':
            df = df[df['sequence_type'] == sequence_type]
        if fat_sat != 'all':
            df = df[df['fat_sat'] == fat_sat]
        if contrast_or_no != 'all':
            df = df[df['contrast_or_no'] == contrast_or_no]
        self.df = df[df['view'] == view]  # Filter by view
        self.base_folder = base_folder
        self.label_column = label_column
        self.target_shape = target_shape
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.clip_limit = clip_limit
        self.denoise = denoise
        self.denoise_method = denoise_method
        self.sigma = sigma
        self.augment = augment
        self.augment_factor = augment_factor
        self.augment_factor_0 = augment_factor_0
        self.transform_val = transform_val
        self.return_unique_id = return_unique_id
        self.use_file_column = use_file_column

        # Define optional transforms for augmentation
        self.transform = Compose([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # Vertical flip
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # Horizontal flip
            RandRotated(keys=["image"], prob=0.5, range_x=5 * np.pi / 180, range_y=0, range_z=0, mode="bilinear"),
        ])

                # Create a list to store the augmented index mapping
        self.augmented_indices = []

        # Iterate through the DataFrame by row number (original index)
        for original_idx in range(len(self.df)):
            label = self.df.iloc[original_idx][self.label_column]
            unique_id = self.df.iloc[original_idx]['unique_id']

            # Append the original index for each sample
            self.augmented_indices.append((original_idx, unique_id))

            # If the label is 1, augment by the specified factor
            if label == 1 and self.augment:
                for _ in range(self.augment_factor - 1):
                    # Append the same original index multiple times for augmentation
                    self.augmented_indices.append((original_idx, unique_id))

            # If the label is 0, augment by the specified factor
            if label == 0 and self.augment:
                for _ in range(self.augment_factor_0 - 1):
                    # Append the same original index multiple times for augmentation
                    self.augmented_indices.append((original_idx, unique_id))

        # Directly print the entire augmented_indices list
        #print(f"Augmented Indices: {self.augmented_indices[:5]}...")  # Print the first 20 for brevity

        # Verify the augmented_indices list
        if any(i >= len(self.df) for i, _ in self.augmented_indices):
            print(f"Error: Found out of bounds index in augmented_indices")

        print(f"Initialized MRIDataset3D with {len(self.df)} samples for view {view}")
        print(f"Augmented dataset has {len(self.augmented_indices)} indices")

    def __len__(self):
        total_length = len(self.augmented_indices)
        # Print the original and augmented dataset sizes for debugging
        print(f"Original dataset size: {len(self.df)}, Augmented dataset size: {total_length}")

        return total_length
        #return len(self.df) * (self.augment_factor if self.augment else 1)

    def __getitem__(self, idx):
        #if self.augment:
        #    idx = idx % len(self.df)

         # Ensure idx is within the bounds of augmented_indices
        if idx >= len(self.augmented_indices) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.augmented_indices)}")

        original_idx, unique_id = self.augmented_indices[idx]

        #print(f"Original index: {original_idx}, Accessed idx: {idx}, Unique ID: {unique_id}")
        label = self.df.iloc[original_idx][self.label_column]

        mri_id = self.df.iloc[original_idx]['mri_id']
        unique_id = self.df.iloc[original_idx]['unique_id']
        sequence_type = self.df.iloc[original_idx]['sequence_type']
        fat_sat = self.df.iloc[original_idx]['fat_sat']


        #mri_id = self.df.iloc[idx]['mri_id']
        patient_number = mri_id[:-3]
        #unique_id = self.df.iloc[idx]['unique_id']

        # Conditionally use the file_column if we are in cross-validation mode
        if self.use_file_column:
            volume_path = self.df.iloc[original_idx]['file_path']
        else:
            volume_path = os.path.join(self.base_folder, mri_id, f"{unique_id}.npy")

        #print(f"Loading volume from {volume_path}")

        #volume = np.load(volume_path)
        try:
            volume = np.load(volume_path)
        except Exception as e:
            print(f"Error loading volume {volume_path}: {e}")
            return None, None
        
        #print(f"Loaded volume shape: {volume.shape}")

        #label = self.df.iloc[idx][self.label_column]
        volume_resized = resize(volume, self.target_shape, anti_aliasing=True)
        if self.denoise:
            for i in range(volume_resized.shape[-1]):
                volume_resized[:, :, i] = denoise_image(volume_resized[:, :, i], method=self.denoise_method, sigma=self.sigma)
        if self.enhance_contrast:
            for i in range(volume_resized.shape[-1]):
                volume_resized[:, :, i] = apply_clahe(volume_resized[:, :, i], clip_limit=self.clip_limit)
        if self.normalize:
            volume_resized = normalize_volume(volume_resized)
        
        #print(f"Volume shape after resizing and preprocessing: {volume_resized.shape}")

        volume_resized = np.expand_dims(volume_resized, axis=-1)  # Add channel dimension
        
        volume_resized = torch.from_numpy(volume_resized).clone().detach().permute(3, 0, 1, 2).float()
        # Apply optional transforms
        if (self.augment or self.transform_val):
            volume_resized = self.apply_transforms(volume_resized)
        
        if self.return_unique_id:
            return volume_resized, torch.tensor(label, dtype=torch.float32), unique_id, mri_id
        else:
            return volume_resized, torch.tensor(label, dtype=torch.float32)

    def apply_transforms(self, volume):
        transformed = self.transform({"image": volume})
        return transformed["image"]

#Define 2D dataset class
class MRIDataset2D(Dataset):
    def __init__(self, df, base_folder, view, label_column, sequence_type='all', fat_sat='all', contrast_or_no='all', target_shape=(224, 224), normalize=False, enhance_contrast=False, clip_limit=0.005, denoise=False, denoise_method='gaussian', sigma=1, augment=False, augment_factor=1, augment_factor_0 = 1, transform_val=True, return_unique_id=False, use_file_column=False):
        if sequence_type != 'all':
            df = df[df['sequence_type'] == sequence_type]
        if fat_sat != 'all':
            df = df[df['fat_sat'] == fat_sat]
        if contrast_or_no != 'all':
            df = df[df['contrast_or_no'] == contrast_or_no]
        self.df = df[df['view'] == view]  # Filter by view
        self.base_folder = base_folder
        self.label_column = label_column
        self.target_shape = target_shape
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.clip_limit = clip_limit
        self.denoise = denoise
        self.denoise_method = denoise_method
        self.sigma = sigma
        self.augment = augment
        self.augment_factor = augment_factor
        self.augment_factor_0 = augment_factor_0
        self.transform_val = transform_val
        self.return_unique_id = return_unique_id
        self.use_file_column = use_file_column

        # Define optional transforms for augmentation
        self.transform = Compose([
            RandFlipd(keys=["image"], prob=0.8, spatial_axis=1),  # Vertical flip
            RandFlipd(keys=["image"], prob=0.8, spatial_axis=2),  # Horizontal flip
            RandRotated(keys=["image"], prob=0.5, range_x=10 * np.pi / 180, range_y=0, range_z=0, mode="bilinear"),
            #RandScaleIntensityd(keys=["image"], prob=0.5, factors=(0.975, 1.025)),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.03),

            #RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1))

            RandAffined(
                keys=["image"],
                prob=0.3,
                #rotate_range=(0, 0, np.pi / 18),  # Smaller rotations
                scale_range=(0.9, 1.1),          # Slight scaling
                translate_range=(0.02, 0.02),    # Small translations
                mode="bilinear",
            ),

            #RandGridDistortiond(
            #    keys=["image"],
            #    prob=0.5,
            #    distort_limit=(-0.03, 0.03),
            #    mode="bilinear",
            #),

        ])


        # Create a list to store the augmented index mapping
        self.augmented_indices = []

        # Iterate through the DataFrame by row number (original index)
        for original_idx in range(len(self.df)):
            label = self.df.iloc[original_idx][self.label_column]
            unique_id = self.df.iloc[original_idx]['unique_id']

            # Append the original index for each sample
            self.augmented_indices.append((original_idx, unique_id))

            # If the label is 1, augment by the specified factor
            if label == 1 and self.augment:
                for _ in range(self.augment_factor - 1):
                    # Append the same original index multiple times for augmentation
                    self.augmented_indices.append((original_idx, unique_id))

            # If the label is 0, augment by the specified factor
            if label == 0 and self.augment:
                for _ in range(self.augment_factor_0 - 1):
                    # Append the same original index multiple times for augmentation
                    self.augmented_indices.append((original_idx, unique_id))

        # Directly print the entire augmented_indices list
        #print(f"Augmented Indices: {self.augmented_indices[:5]}...")  # Print the first 20 for brevity

        # Verify the augmented_indices list
        if any(i >= len(self.df) for i, _ in self.augmented_indices):
            print(f"Error: Found out of bounds index in augmented_indices")

        print(f"Initialized MRIDataset2D with {len(self.df)} samples for view {view}")
        print(f"Augmented dataset has {len(self.augmented_indices)} indices")

    def __len__(self):
        #return len(self.df) * (self.augment_factor if self.augment else 1)
        
        total_length = len(self.augmented_indices)
        # Print the original and augmented dataset sizes for debugging
        #print(f"Original dataset size: {len(self.df)}, Augmented dataset size: {total_length}")
        
        return total_length

    def __getitem__(self, idx):
        #if self.augment:
        #    idx = idx % len(self.df)

        # Print the index being accessed
        #print(f"Accessing index {idx} out of {len(self.augmented_indices)}")

        # Ensure idx is within the bounds of augmented_indices
        if idx >= len(self.augmented_indices) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.augmented_indices)}")

        original_idx, unique_id = self.augmented_indices[idx] 

        #print(f"Original index: {original_idx}, Accessed idx: {idx}, Unique ID: {unique_id}")
        label = self.df.iloc[original_idx][self.label_column]

        mri_id = self.df.iloc[original_idx]['mri_id']
        unique_id = self.df.iloc[original_idx]['unique_id']
        sequence_type = self.df.iloc[original_idx]['sequence_type']
        fat_sat = self.df.iloc[original_idx]['fat_sat']
        
        # Conditionally use the file_column if we are in cross-validation mode
        if self.use_file_column:
            volume_path = self.df.iloc[original_idx]['file_path']
        else:
            volume_path = os.path.join(self.base_folder, mri_id, f"{unique_id}.npy")

        try:
            volume = np.load(volume_path)
        except Exception as e:
            print(f"Error loading volume {volume_path}: {e}")
            return None, None

        # Resize each slice to 256x256
        slices_resized = [resize(slice, (RESIZE_DIM, RESIZE_DIM), anti_aliasing=True) for slice in volume]
        volume_resized = np.stack(slices_resized, axis=0)

        if self.denoise:
            for i in range(volume_resized.shape[-1]):
                volume_resized[:, :, i] = denoise_image(volume_resized[:, :, i], method=self.denoise_method, sigma=self.sigma)
        if self.enhance_contrast:
            for i in range(volume_resized.shape[-1]):
                volume_resized[:, :, i] = apply_clahe(volume_resized[:, :, i], clip_limit=self.clip_limit)

        # Crop the middle region to 224x224
        pad = int((RESIZE_DIM - INPUT_DIM) / 2)
        volume_cropped = volume_resized[:, pad:pad + INPUT_DIM, pad:pad + INPUT_DIM]

        # Determine the mean and stddev for this MRI type
        key = f"{sequence_type}_{fat_sat}"
        mean = MEANS_STDEVS.get(key, MEANS_STDEVS["overall"])["mean"]
        stddev = MEANS_STDEVS.get(key, MEANS_STDEVS["overall"])["stddev"]

        # Normalize using mean and stddev
        volume_normalized = (volume_cropped - mean) / stddev

        # Normalize to [0, 1]
        volume_normalized = (volume_normalized - volume_normalized.min()) / (volume_normalized.max() - volume_normalized.min())

        # No need to convert to 3-channel RGB-like images here, the model code handles it
        volume_tensor = torch.FloatTensor(volume_normalized).unsqueeze(1)  # Add channel dimension

        # Apply optional transforms
        if self.augment or self.transform_val:
            volume_tensor = self.apply_transforms(volume_tensor)

        # Permute to match the required shape: [batch_size, channels, depth, height, width]
        volume_tensor = volume_tensor.permute(1, 0, 2, 3)  # Shape: [1, depth, height, width]
        if self.return_unique_id:
            return volume_tensor, torch.tensor(label, dtype=torch.float32), unique_id, mri_id
        else:
            return volume_tensor, torch.tensor(label, dtype=torch.float32)

    def apply_transforms(self, volume):
        transformed = self.transform({"image": volume})
        return transformed["image"]

def prepare_datasets(batch_size, preprocessed_folder, label_column, views=['sagittal', 'coronal', 'axial', 'ABERS'], sequence_type='all', fat_sat='all', contrast_or_no='all', augment=True, augment_factor=2, augment_factor_0=1, transform_val=True, num_workers=4, model_type='AlexNet', dataset_type='labrum', ret_val_probs=False):
    # Determine the appropriate views based on the dataset type
    if dataset_type == 'MRNet':
        # Limit views to those available in MRNet
        views = ['sagittal', 'coronal', 'axial']

        # Load labels for MRNet dataset
        train_labels, test_labels = load_labels(preprocessed_folder, label_column)
        train_labels, val_labels = create_stratified_validation_set(train_labels, val_size=120, target_label=label_column)

        # Select the correct dataset class based on the model type (2D or 3D)
        if model_type == 'CNN3D':
            train_datasets = {view: MRNetDataset3D(preprocessed_folder, train_labels, view, label_column, target_shape=(224, 224, 224), augment=augment, augment_factor=augment_factor, transform_val=transform_val) for view in views}
            val_datasets = {view: MRNetDataset3D(preprocessed_folder, val_labels, view, label_column, target_shape=(224, 224, 224), augment=False, transform_val=False) for view in views}
            test_datasets = {view: MRNetDataset3D(preprocessed_folder, test_labels, view, label_column, target_shape=(224, 224, 224), is_train=False, augment=False, transform_val=False, return_unique_id=True) for view in views}
            val_test_datasets = {view: MRNetDataset3D(preprocessed_folder, val_labels, view, label_column, target_shape=(224, 224, 224), augment=False, transform_val=False, return_unique_id=True) for view in views}

        else:
            train_datasets = {view: MRNetDataset2D(preprocessed_folder, train_labels, view, label_column, augment=augment, augment_factor=augment_factor, transform_val=transform_val) for view in views}
            val_datasets = {view: MRNetDataset2D(preprocessed_folder, val_labels, view, label_column, augment=False, transform_val=False) for view in views}
            test_datasets = {view: MRNetDataset2D(preprocessed_folder, test_labels, view, label_column, is_train=False, augment=False, transform_val=False, return_unique_id=True) for view in views}
            val_test_datasets = {view: MRNetDataset2D(preprocessed_folder, val_labels, view, label_column, augment=False, transform_val=False, return_unique_id=True) for view in views}
    else:
        # Load metadata for labrum dataset
        train_df = load_metadata('train', preprocessed_folder)
        val_df = load_metadata('val', preprocessed_folder)
        test_df = load_metadata('test', preprocessed_folder)

        print("Initial train DataFrame shape:", train_df.shape if train_df is not None else "Not Found")
        print("Initial validation DataFrame shape:", val_df.shape if val_df is not None else "Not Found")
        print("Initial test DataFrame shape:", test_df.shape if test_df is not None else "Not Found")

        # For labrum dataset, include all possible views including ABERS
        if model_type == 'CNN3D':
            train_datasets = {view: MRIDataset3D(train_df, os.path.join(preprocessed_folder, 'train'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=augment, augment_factor=augment_factor, augment_factor_0=augment_factor_0, transform_val=transform_val) for view in views}
            val_datasets = {view: MRIDataset3D(val_df, os.path.join(preprocessed_folder, 'val'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False) for view in views}
            test_datasets = {view: MRIDataset3D(test_df, os.path.join(preprocessed_folder, 'test'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False, return_unique_id=True) for view in views}
            val_test_datasets = {view: MRIDataset3D(val_df, os.path.join(preprocessed_folder, 'val'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False, return_unique_id=True) for view in views}

        else:
            train_datasets = {view: MRIDataset2D(train_df, os.path.join(preprocessed_folder, 'train'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=augment, augment_factor=augment_factor, augment_factor_0=augment_factor_0, transform_val=transform_val) for view in views}
            val_datasets = {view: MRIDataset2D(val_df, os.path.join(preprocessed_folder, 'val'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False) for view in views}
            test_datasets = {view: MRIDataset2D(test_df, os.path.join(preprocessed_folder, 'test'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False, return_unique_id=True) for view in views}
            val_test_datasets = {view: MRIDataset2D(val_df, os.path.join(preprocessed_folder, 'val'), view, label_column, sequence_type, fat_sat, contrast_or_no, augment=False, transform_val=False, return_unique_id=True) for view in views}


    # Create DataLoaders for each view
    
    train_loaders = {view: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    ) for view, dataset in train_datasets.items()}

    val_loaders = {view: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    ) for view, dataset in val_datasets.items()}

    test_loaders = {view: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    ) for view, dataset in test_datasets.items()}

    val_test_loaders = {view: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    ) for view, dataset in val_test_datasets.items()}

    if ret_val_probs: 
        return train_loaders, val_loaders, test_loaders, val_test_loaders
    else:
        return train_loaders, val_loaders, test_loaders


class DynamicCombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.indices = {view: list(range(len(dataset))) for view, dataset in datasets.items()}

    def __len__(self):
        return sum(len(indices) for indices in self.indices.values())

    def __getitem__(self, idx):
        dataset_key = random.choice(list(self.datasets.keys()))
        dataset = self.datasets[dataset_key]
        index = self.indices[dataset_key][idx % len(self.indices[dataset_key])]
        return dataset[index]

def create_dynamic_combined_loader(datasets, batch_size, num_workers=0):
    combined_dataset = DynamicCombinedDataset(datasets)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return combined_loader, combined_dataset


def prepare_and_create_loaders(args, num_workers=4, ret_val_probs=False):
    # Prepare datasets based on the provided arguments

    #ret_val_probs=args.ret_val_probs
    ret_val_probs = getattr(args, 'ret_val_probs', False)

    print(f"ret_val_probs: {ret_val_probs}")

    if ret_val_probs:
        train_loaders, val_loaders, test_loaders, val_test_loaders = prepare_datasets(
            batch_size=args.batch_size,
            preprocessed_folder=args.preprocessed_folder,
            label_column=args.label_column,
            sequence_type=args.sequence_type,
            fat_sat=args.fat_sat,
            contrast_or_no=args.contrast_or_no,
            augment=args.augment,
            augment_factor=args.augment_factor,
            augment_factor_0=args.augment_factor_0,
            transform_val=args.transform_val,
            model_type=args.model_type,
            dataset_type=args.dataset_type,  # Determine dataset type (labrum or MRNet)
            num_workers=num_workers,
            ret_val_probs=True,
        )
    
    else: 
        train_loaders, val_loaders, test_loaders = prepare_datasets(
            batch_size=args.batch_size,
            preprocessed_folder=args.preprocessed_folder,
            label_column=args.label_column,
            sequence_type=args.sequence_type,
            fat_sat=args.fat_sat,
            contrast_or_no=args.contrast_or_no,
            augment=args.augment,
            augment_factor=args.augment_factor,
            augment_factor_0=args.augment_factor_0,
            transform_val=args.transform_val,
            model_type=args.model_type,
            dataset_type=args.dataset_type,  # Determine dataset type (labrum or MRNet)
            num_workers=num_workers,
        )

    # Initialize datasets based on selected view
    train_datasets = {view: train_loaders[view].dataset for view in train_loaders if train_loaders[view] is not None}
    val_datasets = {view: val_loaders[view].dataset for view in val_loaders if val_loaders[view] is not None}
    test_datasets = {view: test_loaders[view].dataset for view in test_loaders if test_loaders[view] is not None}

    # Create the combined data loaders if training on all views at once
    if args.view == 'all':
        combined_train_loader, combined_train_dataset = create_dynamic_combined_loader(train_datasets, args.batch_size, num_workers=num_workers)
        train_dataset_size = len(combined_train_dataset)

        combined_val_loader, combined_val_dataset = create_dynamic_combined_loader(val_datasets, args.batch_size, num_workers=num_workers)
        val_dataset_size = len(combined_val_dataset)
        
        combined_test_loader, combined_test_dataset = create_dynamic_combined_loader(test_datasets, args.batch_size, num_workers=num_workers)
        test_dataset_size = len(combined_test_dataset)
    else:
        print(f"View: {args.view}")
        combined_train_loader = train_loaders[args.view]
        train_dataset_size = len(train_loaders[args.view].dataset)

        combined_val_loader = val_loaders[args.view]
        val_dataset_size = len(val_loaders[args.view].dataset)

        combined_test_loader = test_loaders[args.view]
        test_dataset_size = len(test_loaders[args.view].dataset)

    # Print the sizes of the train, validation, and test datasets
    print(f"Train dataset size: {train_dataset_size}")
    print(f"Validation dataset size: {val_dataset_size}")
    print(f"Test dataset size: {test_dataset_size}")

    if ret_val_probs:
        val_test_datasets = {view: val_test_loaders[view].dataset for view in val_test_loaders if val_test_loaders[view] is not None}
        if args.view == 'all':
            combined_val_test_loader, combined_val_test_dataset = create_dynamic_combined_loader(val_test_datasets, args.batch_size, num_workers=num_workers)
            val_test_dataset_size = len(combined_val_test_dataset)
        else: 
            combined_val_test_loader = val_test_loaders[args.view]
            val_test_dataset_size = len(val_test_loaders[args.view].dataset)
        print(f"Val Test dataset size: {val_test_dataset_size}")
        return combined_train_loader, combined_val_loader, combined_test_loader, combined_val_test_loader

    else:
        return combined_train_loader, combined_val_loader, combined_test_loader

def prepare_and_create_loaders_from_params(
    preprocessed_folder, label_column, view, batch_size, augment, augment_factor, 
    transform_val, model_type, dataset_type, sequence_type=None, fat_sat=None, 
    contrast_or_no=None, num_workers=4, augment_factor_0=1
):
    # Simulate the 'args' dictionary as would be passed to 'prepare_and_create_loaders'
    args = {
        'preprocessed_folder': preprocessed_folder,
        'label_column': label_column,
        'view': view,
        'batch_size': batch_size,
        'augment': augment,
        'augment_factor': augment_factor,
        'augment_factor_0': augment_factor_0,
        'transform_val': transform_val,
        'model_type': model_type,
        'dataset_type': dataset_type,
        'sequence_type': sequence_type,
        'fat_sat': fat_sat,
        'contrast_or_no': contrast_or_no,
    }

    # Convert the dictionary to a type that behaves like 'args'
    class ArgsNamespace:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args = ArgsNamespace(**args)

    # Use the existing function with this constructed 'args'
    return prepare_and_create_loaders(args, num_workers=num_workers, ret_val_probs=False)

