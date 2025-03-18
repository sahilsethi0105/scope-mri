import argparse
import os
import torch
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1, ResNet34, DenseNet, EfficientNet
from training_functions import evaluate_model, load_model_weights
from tqdm import tqdm
import re
import numpy as np

# Constants for normalization
INPUT_DIM = 224
RESIZE_DIM = 400 
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

class SemiexternalDataset(Dataset):
    def __init__(self, base_folder, target_label, view='all', contrast_or_no='all', target_shape=(224, 224), return_unique_id=True):
        self.df = pd.read_csv(os.path.join(base_folder, 'metadata.csv'))

        # Apply filters
        if view != 'all':
            self.df = self.df[self.df['view'] == view]
        if contrast_or_no != 'all':
            self.df = self.df[self.df['contrast_or_no'] == contrast_or_no]

        self.base_folder = base_folder
        self.target_label = target_label
        self.target_shape = target_shape
        self.return_unique_id = return_unique_id

        print(f"Initialized SemiexternalDataset with {len(self.df)} samples for view {view} and contrast {contrast_or_no}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mri_id = row['mri_id']
        unique_id = row['unique_id']
        label = row[self.target_label]
        sequence_type = row['sequence_type']
        fat_sat = row['fat_sat']

        print(f"Processing MRI ID: {mri_id}, Unique ID: {unique_id}")
        # Load the MRI volume
        mri_folder = os.path.join(self.base_folder, mri_id)
        volume_path = os.path.join(mri_folder, f"{unique_id}.npy")

        try:
            volume = np.load(volume_path)
            print(f"Loaded volume with shape: {volume.shape}")
        except Exception as e:
            print(f"Error loading volume {volume_path}: {e}")
            return None, None

        if volume is None:
            print(f"Returning None for MRI ID: {mri_id}, Unique ID: {unique_id}")
            return None, None

        # Resize each slice to the target shape (e.g., 400x400)
        slices_resized = [resize(slice, (RESIZE_DIM, RESIZE_DIM), anti_aliasing=True) for slice in volume]
        volume_resized = np.stack(slices_resized, axis=0)

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
        volume_normalized = volume_cropped 
        volume_normalized = (volume_normalized - volume_normalized.min()) / (volume_normalized.max() - volume_normalized.min())

        # Convert to tensor and add the channel dimension
        volume_tensor = torch.FloatTensor(volume_normalized).unsqueeze(1)  # Shape: [1, depth, height, width]

        # Permute to match the required shape: [channels, depth, height, width]
        volume_tensor = volume_tensor.permute(1, 0, 2, 3)  # Shape: [1, depth, height, width]
        
        print(volume_tensor.shape)

        if self.return_unique_id:
            return volume_tensor, torch.tensor(label, dtype=torch.float32), unique_id, mri_id
        else:
            return volume_tensor, torch.tensor(label, dtype=torch.float32)

# Define the main evaluation function
def evaluate_semiexternal_model(args, num_epochs=0):
    # Set device to CUDA if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_class = globals()[args.model_type]
    model = model_class().to(device)
    model = load_model_weights(model, args.model_weights, device, exclude_final=False)

    # Define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the dataset
    dataset = SemiexternalDataset(base_folder=args.base_folder, target_label=args.label_column, view=args.view, contrast_or_no=args.contrast_or_no)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # TensorBoard logging setup
    trial_num = 0
    base_log_dir = f"/gpfs/data/orthopedic-lab/tb_logs/{args.job_name}"

    if os.path.exists(base_log_dir):
        subdirs = os.listdir(base_log_dir)
        trial_dirs = [d for d in subdirs if re.match(r'trial_\d+', d)]
        if trial_dirs:
            trial_numbers = [int(re.search(r'\d+', d).group()) for d in trial_dirs]
            trial_num = max(trial_numbers) + 1

    log_dir = os.path.join(base_log_dir, f"trial_{trial_num}")
    writer = SummaryWriter(log_dir=log_dir, comment="")

    print(f"Logging to TensorBoard directory: {log_dir}")
    print(f"Current trial number: {trial_num}")

    # Define the path to save probabilities
    prob_csv_path = f'/gpfs/data/orthopedic-lab/ortho_ml/experiments/{args.job_name}_probs.csv'

    # Evaluate the model
    y_true, y_pred, y_prob = evaluate_model(model, dataloader, criterion, device=device, writer=writer, csv_path=prob_csv_path)

        # Classification report
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    print("Model Training Complete")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensitivity (Recall): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Calculate AUC
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc:.4f}")

    # Log the final metrics to TensorBoard
    writer.add_scalar('Test Acc', accuracy)
    writer.add_scalar('Test Recall', recall)
    writer.add_scalar('Test Specificity', specificity)
    writer.add_scalar('Test Precision', precision)
    writer.add_scalar('Test F1_Score', f1)
    writer.add_scalar('Test AUC', auc)

    # Log the confusion matrix as an image
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_normalized, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='.2f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Log the confusion matrix as a figure
    writer.add_figure('Confusion Matrix', plt.gcf(), num_epochs)
    plt.close()

    # Plot ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Log the ROC-AUC plot as a figure
    writer.add_figure('ROC-AUC', plt.gcf(), num_epochs)
    plt.close()

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on semiexternal validation dataset')
    parser.add_argument('--base_folder', type=str, required=True, help='Base folder where MRI subfolders and metadata.csv are located')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--label_column', type=str, required=True, help='Column in metadata CSV with the target label')
    parser.add_argument('--model_type', type=str, required=True, help='Model type (e.g., ResNet50, AlexNet, etc.)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--contrast_or_no', type=str, default='all', choices=['W', 'WO', 'all'], help='Filter MRIs based on contrast (W, WO, or all).')
    parser.add_argument('--view', type=str, default='all', help='Select an MRI view (i.e. coronal, sagittal, or axial)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--job_name', type=str, required=True, help='Job name for logging')

    args = parser.parse_args()
    evaluate_semiexternal_model(args)

