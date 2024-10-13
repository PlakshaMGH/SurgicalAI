import os
import sys
import gc
import argparse
import pandas as pd

XMem_path = os.path.abspath("../../../external/XMem")  # Parent folder /app/mount
sys.path.append(XMem_path)

from inspect import getsource
from pathlib import Path
from os import path

import cv2
import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import Dice

# from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
# from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper

from inference.interact.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
)

torch.set_grad_enabled(False)

# default configuration
config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}

if torch.cuda.is_available():
    print("Using GPU")
    device = "cuda"
else:
    print("CUDA not available. Please connect to a GPU instance if possible.")
    device = "cpu"


torch.cuda.empty_cache()

COLOR = (3, 192, 60)

main_folder = Path("../../../data/endovis/endo17/data")
VIDEOS_PATH = main_folder / "frames" / "endo17_test_frames"
MASKS_PATH = main_folder / "masks" / "endo17_test_masks" / "binary_masks"


def binary2color(binary_mask, color):
    binary_mask = torch_prob_to_numpy_mask(binary_mask)
    pred_mask = np.tile(binary_mask[..., np.newaxis], (1, 1, 3))  # Make it 3 Channel
    mask = np.where(pred_mask == (1,) * 3, color, 0).astype(
        "uint8"
    )  # Convert Prediction with Color
    return mask


def frames2video(frames_dict, folder_save_path, video_name, FPS=5):
    frame = frames_dict[list(frames_dict.keys())[-1]]
    size1, size2, _ = frame.shape
    out = cv2.VideoWriter(
        f"{folder_save_path}/{video_name}_{FPS}FPS.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (size2, size1),
        True,
    )
    # Sorting the frames according to frame number eg: frame_007.png
    for _, i in sorted(frames_dict.items(), key=lambda x: x[0]):
        out_img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        out.write(out_img)
    out.release()


def getIoU(pred_frames, gt_path):
    metric = BinaryJaccardIndex()
    dice_metric = Dice()

    IoU = []
    dice = []

    for frame_name, mask in pred_frames.items():
        mask = torch_prob_to_numpy_mask(mask)
        try:
            truth_mask = io.imread(gt_path / frame_name)
        except FileNotFoundError:
            continue
        truth_mask = np.where(truth_mask == 255, 1, truth_mask)
        if np.sum(truth_mask) == 0:
            continue
        truth_mask = torch.tensor(truth_mask)
        IoU.append(metric(torch.tensor(mask), truth_mask).item())
        dice.append(dice_metric(torch.tensor(mask), truth_mask).item())

    meanIoU = sum(IoU) / len(IoU)
    meanDice = sum(dice) / len(dice)

    return meanIoU, IoU, meanDice, dice


im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def resize_mask(mask, size):
    mask = mask.unsqueeze(0).unsqueeze(0)
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    return F.interpolate(
        mask, (int(h / min_hw * size), int(w / min_hw * size)), mode="nearest"
    )[0]


def singleVideoInference(images_paths, first_mask, processor, size=-1):
    predictions = {}
    frames = {}
    with torch.cuda.amp.autocast(enabled=True):

        images_paths = sorted(images_paths)

        # First Frame
        frame = io.imread(images_paths[0])
        shape = frame.shape[:2]
        if size < 0:
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                ]
            )
        else:
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
                ]
            )

        frame_torch = im_transform(frame).to(device)
        first_mask = first_mask.astype(np.uint8)
        if size > 0:
            first_mask = torch.tensor(first_mask).to(device)
            first_mask = resize_mask(first_mask, size)
        else:
            NUM_OBJECTS = 1  # Binary Segmentation
            first_mask = index_numpy_to_one_hot_torch(first_mask, NUM_OBJECTS + 1).to(
                device
            )
            first_mask = first_mask[1:]

        prediction = processor.step(frame_torch, first_mask)

        for image_path in images_paths[1:]:
            frame = io.imread(image_path)
            # convert numpy array to pytorch tensor format
            frame_torch = im_transform(frame).to(device)

            prediction = processor.step(frame_torch)
            # Upsample to original size if needed
            if size > 0:
                prediction = F.interpolate(
                    prediction.unsqueeze(1), shape, mode="bilinear", align_corners=False
                )[:, 0]
            predictions[image_path.name] = prediction
            frames[image_path.name] = frame

    return frames, predictions


def firstMaskGT(image_files, mask_folder):
    image_files = sorted(image_files)

    for idx, image_path in enumerate(image_files):

        # Getting the Path to Mask Ground Truth using RGB Image path
        mask_path = mask_folder / image_path.parent.name / image_path.name

        mask = io.imread(mask_path)
        # All 255 Values replaced with 1, other values remain as it is.
        mask = np.where(mask == 255, 1, mask)

        if np.sum(mask) > 0:
            return mask, idx

    return None, -1


def doInference(
    network_path,
    config,
    frames_folder,
    mask_folder,
    subset=None,
    pred_mask_folder=None,
    size=-1,
):
    overallIoU = []
    overallDice = []
    for video_folder in tqdm(sorted(frames_folder.iterdir())):

        if subset is not None and video_folder.name not in subset:
            continue

        # Clearing GPU Cache
        torch.cuda.empty_cache()
        network = XMem(config, network_path).eval().to(device)
        processor = InferenceCore(network, config=config)
        NUM_OBJECTS = 1  # Binary Segmentation
        processor.set_all_labels(range(1, NUM_OBJECTS + 1))

        # All Images
        image_files = sorted(list(video_folder.iterdir()))
        if pred_mask_folder:
            mask_path = [
                i for i in pred_mask_folder.iterdir() if video_folder.name in i.name
            ][0]
            mask = io.imread(mask_path)
            # All 0 pixel is 0, everything else(which is mask) is 1
            mask = np.where(mask == 0, 0, 1)
            # seq_01_0.png -> Two Splits, one on '_', other on '.'
            start_idx = int((mask_path.name.split("_")[-1]).split(".")[0])
        else:  # Ground Truth
            mask, start_idx = firstMaskGT(image_files, mask_folder)

        print(f"Running Inference on {video_folder.name}...")
        frames, predictions = singleVideoInference(
            image_files[start_idx:], mask, processor, size=size
        )
        IoU, _, dice, _ = getIoU(predictions, mask_folder / video_folder.name)
        print(f'Video "{video_folder.name}", mean IoU is: {IoU}')
        print(f'Video "{video_folder.name}", mean dice is: {dice}')

        overallIoU.append(IoU)
        overallDice.append(dice)
        print()

        del network, processor
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Average IoU over all videos is: {sum(overallIoU)/len(overallIoU)}.")
    print(f"Average Dice over all videos is: {sum(overallDice)/len(overallDice)}.")

    return overallIoU, overallDice


parser = argparse.ArgumentParser()
parser.add_argument("--network_dir", type=str, required=True)
parser.add_argument("--video_names", type=str, nargs="+", required=False)
args = parser.parse_args()


# constructing the test subset
test_subset_names = args.video_names
if test_subset_names is not None:
    test_subset = {i.name for i in VIDEOS_PATH.iterdir() if i.name in test_subset_names}
else:
    test_subset_names = {i.name for i in VIDEOS_PATH.iterdir()}
    test_subset = None

# construct the paths to the networks
network_dir = args.network_dir
network_dir = Path(network_dir)
paths = []
if network_dir.is_dir():
    for network_path in network_dir.iterdir():
        if "checkpoint" in network_path.name or ".pth" not in network_path.name:
            continue
        paths.append(network_path)
else:
    # will test only a single model
    paths.append(network_dir)

IoUs = {}
DiceScores = {}
for network_path in sorted(
    paths, key=lambda x: int(x.name.split("_")[-1].split(".")[0])
):
    print(network_path.name)
    overallIoU, overallDice = doInference(
        network_path,
        config,
        VIDEOS_PATH,
        MASKS_PATH,
        subset=test_subset,
        size=384,
    )
    # store the IoU and Dice scores in a dictionary according to the name of the test subset video
    named_IoUs = {name: iou for name, iou in zip(test_subset_names, overallIoU)}
    named_DiceScores = {
        name: dice for name, dice in zip(test_subset_names, overallDice)
    }
    IoUs[network_path.name] = named_IoUs
    DiceScores[network_path.name] = named_DiceScores
    print("*" * 100)

print("Inference Completed")

# saving the results into a json file
output_name = f"{network_dir.name}"
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

IoUs_df = pd.DataFrame(IoUs)
IoUs_df.to_json(output_dir / f"{output_name}_IoUs.json")

DiceScores_df = pd.DataFrame(DiceScores)
DiceScores_df.to_json(output_dir / f"{output_name}_DiceScores.json")
