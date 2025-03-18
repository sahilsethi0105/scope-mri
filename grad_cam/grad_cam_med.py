from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
import sys
sys.path.append("../")
from models import CNN3D, ResNet50, AlexNet, VisionTransformer, SwinTransformerV1, SwinTransformerV2, ResNet34, DenseNet, EfficientNet
from training_functions import load_model_weights
from loader import prepare_and_create_loaders, load_metadata
import argparse
from typing import Callable, List, Optional, Tuple
import os
from pathlib import Path

# Function to convert string to boolean
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_folder', type=str, choices=["/gpfs/data/orthopedic-lab/MRNet-v1.0/MRNet-v1.0", '/gpfs/data/orthopedic-lab/HIRO_Image-Data/HIR 17404/non_resized_MRIs/final_SLAP_data'], default="/gpfs/data/orthopedic-lab/MRNet-v1.0/MRNet-v1.0",)
parser.add_argument('--label_column', type=str, default="acl",)
parser.add_argument('--view', type=str, default="sagittal", help='Selected view (axial, sagittal, coronal, ABERS, all)')
parser.add_argument('--batch_size', type=int, default=1, )
parser.add_argument('--num_epochs', type=int, default=0, )
parser.add_argument('--job_name', type=str, default="grad_cam", )
#NOTE(MS): arg to vary
parser.add_argument('--model_type', default="VisionTransformer", choices=["VisionTransformer","ResNet50", "AlexNet", "SwinTransformerV1"], type=str, help='Model type (ResNet50, MRNet, or CNN3D)')
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--augment', type=str2bool, nargs='?', const=True, default=False, help="Whether to apply data augmentation during training")
parser.add_argument('--augment_factor', type=int, default=20, )
parser.add_argument('--augment_factor_0', type=int, default=1)
parser.add_argument('--model_weights', type=str, default='/gpfs/data/orthopedic-lab/MRNet-v1.0/MRNet-v1.0/mrnet_hband7/mrnet_hband7_trial_38_model.pth', help='Path to model weights to initialize from')
parser.add_argument('--transform_val', type=str2bool, nargs='?', const=True, default=True, help="Whether to apply data augmentation during training")
parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='Scheduler type (ReduceLROnPlateau, CosineAnnealingLR)')
parser.add_argument('--sequence_type', type=str, default='all', help='Sequence type to include (e.g., T1, T2, all)')
parser.add_argument('--fat_sat', type=str, default='all', help='Fat saturation to include (Yes, No, all)')
parser.add_argument('--contrast_or_no', type=str, default='all', help='Contrast type to include (WO, W, WWO, all)')
parser.add_argument('--dataset_type', type=str, default="MRNet",choices=['labrum', 'MRNet'], help='Dataset type (labrum or MRNet)')
parser.add_argument('--pos_weight', type=str, default="automatic", help='Set pos_weight to "automatic" or an integer value.')
parser.add_argument('--script_mode', type=str, default='train', help='Script mode (train or CV)')
parser.add_argument('--n_cycles', type=int, default=5, help='Number of folds for CV')
parser.add_argument('--use_label', type=int, default=0, choices=[1,0], help='1 for using label in grad cam, 0 for using no label')
parser.add_argument('--all_slices', type=int, default=0, choices=[1,0], help='1 for iterating through all slices, 0 for middle slice.')
parser.add_argument('--result_dir', type=str, default='results', help='director to output masked images')

args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if "SwinTransformer" in args.model_type:
    def reshape_transform(tensor, height=7, width=7):
        #print("shape of layer output: ", tensor.shape)
        result = tensor
        #NOTE(MS): Typically we would reshape, but not needed here
        #result = tensor.reshape(tensor.size(0),
        #    height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        #print("shape of transformed layer: ", result.shape)
        return result
    if args.model_type == "SwinTransformerV1":
        model = SwinTransformerV1(dropout_rate=args.dropout_rate).to(device)
        target_layers = [model.model.features]
        #target_layers = [model.model.features[5][5].norm1]

if args.model_type == "AlexNet":
    def reshape_transform(tensor):
        return tensor
    model = AlexNet(dropout_rate=args.dropout_rate).to(device)
    #print(model)
    target_layers = [model.model.features]

if args.model_type == "VisionTransformer":
    def reshape_transform(tensor, height=14, width=14):
        #print("shape of layer output: ", tensor.shape)
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),
            height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        #print("shape of transformed layer: ", result.shape)
        return result
    model = VisionTransformer(dropout_rate=args.dropout_rate).to(device)
    target_layers = [model.model.encoder.layers.encoder_layer_10.ln_1]

#model_path = '/gpfs/data/orthopedic-lab/MRNet-v1.0/MRNet-v1.0/mrnet_hband7/mrnet_hband7_trial_38_model.pth'
model = load_model_weights(model, args.model_weights, device, exclude_final=False)
model.eval()
print("loaded model")

#Load in images
print(args)
train_loader, val_loader, test_loader = prepare_and_create_loaders(args, num_workers=0)
print("loaded data")

############ NOTE(MS): custom grad cam for VIT model
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            if len(img.shape) > 2:
                img = zoom(np.float32(img), [
                           (t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])])
            else:
                #print("image shape", img.shape)
                #NOTE that the issue here is the we had too many dimensions
                target_size = (target_size[1], target_size[2])
                #print("target_size:", target_size)
                img = cv2.resize(np.float32(img), target_size)

        result.append(img)
    result = np.float32(result)

    return result


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
    ) -> None:
        #print("USING CUSTOM BASE CAM")
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return height, depth, width
            #return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        #print("GRAD CAM TARGET SIZE:", target_size)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

import numpy as np

#from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        #print("using custom grad cam")
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
############
for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Batches")):
    print(f"new batch {batch_idx}")
    if len(batch) == 4:
        inputs, labels, unique_id, mri_id = batch
        save_csv = True
    else:
        inputs, labels = batch
        unique_id = ["N/A"] * len(inputs)
        mri_id = ["N/A"] * len(inputs)

    # Iterate over each image in the batch to check for the target label
    for i in range(len(labels)):
        # Process the input image
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        target_label_value = int(labels[i].item())

        print(f"Processing batch number: {batch_idx}, image number: {i} with label value: {target_label_value}")
        print(f"mri_id: {mri_id[i]}")


        # We have to specify the target we want to generate the CAM for.
        if args.use_label:
            #NOTE (MS): single binary classifier, the classifier output target is supposed to be the index, not the binary value
            # the index will always be 0
            targets = [ClassifierOutputTarget(0)]
            #targets = [ClassifierOutputTarget(target_label_value)]
        else:
            targets = None

        if args.all_slices:
            num_slices = inputs.shape[2]  # Assuming the slice dimension is the third axis
            slice_idxs = range(num_slices)
        else:
            num_slices = inputs.shape[2]  # Assuming the slice dimension is the third axis
            slice_idx = num_slices // 2
            slice_idxs = [slice_idx,]
            
        for slice_idx in slice_idxs:
            input_tensor = torch.unsqueeze(inputs[:,:,slice_idx,:,:], 2)


            # Construct the CAM object once, and then re-use it on many images.
            with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
              # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
              grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
              # In this example grayscale_cam has only one image in the batch:
              grayscale_cam = grayscale_cam[0, :]
              #visualization = show_cam_on_image(input_tensor[0,0,:,:,:].numpy(), grayscale_cam, use_rgb=False)
              # You can also get the model outputs without having to redo inference
              model_outputs = cam.outputs

              base_mri = input_tensor[0,0,0,:,:].cpu().numpy()
              base_mri = cv2.cvtColor(base_mri, cv2.COLOR_GRAY2RGB)
              #plt.imshow(base_mri,cmap='gray')

              print(f"Got grad cam mask. Label: {labels[0].item()}, MRI ID: {mri_id[0]}, Slice idx: {slice_idx}")

              visualization = show_cam_on_image(base_mri, grayscale_cam, use_rgb=True, image_weight=0.7)
                 
              im = Image.fromarray(visualization)

              # set static out dir based on args
              # I don't want to record the # of rounds, incase this changes in future experiments
              model_weights = args.model_weights.split("/")[-1]
              model_weights = model_weights.split(".")[0]
              arg_path = "_".join(map(str, list(vars(args).values())[1:]))
              # Need to remove any . or / to ensure a single continuous file path
              arg_path = arg_path.replace(".", "")
              #arg_path = arg_path.replace("/", "")
              run_dir = Path(f"{args.result_dir}/{args.job_name}/{model_weights}/mri_{mri_id[0]}_label_{target_label_value}_batch_{batch_idx}/use_label_{args.use_label}/")
              if not os.path.exists(run_dir):
                    os.makedirs(run_dir)
              im.save(f"{run_dir}/slice_{slice_idx}.jpeg")

              #break
