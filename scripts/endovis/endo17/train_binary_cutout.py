import datetime
import math
import os
from os import path
from pathlib import Path
import sys
from tqdm import tqdm

import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.distributed as distributed

distributed.init_process_group(
    backend="nccl", init_method="tcp://localhost:25763", rank=0, world_size=1
)
print(f"CUDA Device count: {torch.cuda.device_count()}")

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f"I am rank {local_rank} in this world of size {world_size}!")

XMem_path = os.path.abspath("../../../external/XMem")  # Parent folder /app/mount
sys.path.append(XMem_path)
#!wget -P ../../../external/XMem/saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth

from util.logger import TensorboardLogger

from model.trainer import XMemTrainer

torch.backends.cudnn.benchmark = True

im_mean = (124, 116, 104)

im_normalization = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_im_trans = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class BinaryInstrumentDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(
        self,
        im_root,
        gt_root,
        max_jump,
        subset=None,
        num_frames=3,
        max_num_obj=1,
        finetune=False,
    ):

        self.im_root = im_root  # Root directory for Images
        self.gt_root = gt_root  # Root directory for ground truth data
        self.max_jump = max_jump  # Maximum distance between frames
        # self.is_bl = is_bl # Is it a blender dataset
        self.num_frames = num_frames  # Number of frames to be sampled
        self.max_num_obj = max_num_obj  # Maximum number of objects

        # Initialize lists for storing video and frame information
        self.videos = []  # List of videos
        self.frames = {}  # Dictionary mapping video to its frames

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            # If looking for a subset of videos only
            if subset is not None:
                vid_name = vid
                if vid_name not in subset:
                    continue

            # List frames in each video directory
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))

            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print(
            "%d out of %d videos accepted in %s."
            % (len(self.videos), len(vid_list), im_root)
        )

        # only going to perform cutout on the images within a video
        self.resize_transform = A.Resize(384, 384)
        self.cutout_transforms = A.CoarseDropout(
            num_holes_range=[2, 5],
            hole_height_range=[50, 100],
            hole_width_range=[50, 100],
            mask_fill_value=0.0,  # blackout the image
            fill_value=0.0,  # blackout the image
        )
        self.horizontal_flip_transform = A.HorizontalFlip()
        # Final transform without randomness
        self.normalise_img = im_normalization

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(
                range(
                    max(0, frames_idx[-1] - this_max_jump),
                    min(length, frames_idx[-1] + this_max_jump + 1),
                )
            ).difference(set(frames_idx))
            while len(frames_idx) < num_frames:
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(
                    range(
                        max(0, frames_idx[-1] - this_max_jump),
                        min(length, frames_idx[-1] + this_max_jump + 1),
                    )
                )
                acceptable_set = acceptable_set.union(new_set).difference(
                    set(frames_idx)
                )

            frames_idx = sorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx]
                png_name = frames[f_idx]
                info["frames"].append(jpg_name)

                reseed(sequence_seed)
                this_im = cv2.imread(path.join(vid_im_path, jpg_name))
                this_gt = cv2.imread(path.join(vid_gt_path, png_name))
                this_im = cv2.cvtColor(this_im, cv2.COLOR_BGR2RGB)
                this_gt = cv2.cvtColor(this_gt, cv2.COLOR_BGR2GRAY)
                transformed = self.resize_transform(image=this_im, mask=this_gt)
                this_im, this_gt = transformed["image"], transformed["mask"]
                transformed = self.horizontal_flip_transform(
                    image=this_im, mask=this_gt
                )
                this_im, this_gt = transformed["image"], transformed["mask"]

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                cutout_transform = self.cutout_transforms(image=this_im, mask=this_gt)
                this_im = cutout_transform["image"]
                this_gt = cutout_transform["mask"]

                transformed = self.normalise_img(image=this_im, mask=this_gt)
                this_im, this_gt = transformed["image"], transformed["mask"]
                transformed = ToTensorV2()(image=this_im, mask=this_gt)
                this_im, this_gt = transformed["image"], transformed["mask"]
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(
                target_objects, size=self.max_num_obj, replace=False
            )

        info["num_objects"] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = masks == l
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = this_mask[0]
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [
            1 if i < info["num_objects"] else 0 for i in range(self.max_num_obj)
        ]
        selector = torch.FloatTensor(selector)

        data = {
            "rgb": images,
            "first_frame_gt": first_frame_gt,
            "cls_gt": cls_gt,
            "selector": selector,
            "info": info,
        }

        return data

    def __len__(self):
        return len(self.videos)


main_folder = Path("../../../data/endovis/endo17/data")
VIDEOS_PATH = main_folder / "frames" / "endo17_train_frames"
MASKS_PATH = main_folder / "masks" / "endo17_train_masks" / "binary_masks"

sample_dataset = BinaryInstrumentDataset(
    VIDEOS_PATH,
    MASKS_PATH,
    max_jump=20,
    subset=None,
    num_frames=8,
    max_num_obj=1,
    finetune=True,
)

raw_config = {
    "benchmark": False,
    "bl_root": "../BL30K",
    "davis_root": "../DAVIS",
    "debug": False,
    "deep_update_prob": 0.2,
    "exp_id": "NULL",
    "gamma": 0.1,
    "hidden_dim": 64,
    "key_dim": 64,
    "load_checkpoint": None,
    "load_network": None,
    "log_image_interval": 100,
    "log_text_interval": 100,
    "no_amp": False,
    "num_workers": 0,
    "s3_batch_size": 2,
    "s3_end_warm": 700,
    "s3_finetune": 0,
    "s3_iterations": 3_000,
    "s3_lr": 1e-5,
    "s3_num_frames": 16,
    "s3_num_ref_frames": 3,
    "s3_start_warm": 200,
    "s3_steps": [2400],
    "save_checkpoint_interval": 100,
    "save_network_interval": 100,
    "stages": "3",
    "static_root": "../static",
    "value_dim": 512,
    "weight_decay": 0.05,
    "yv_root": "../YouTube",
}

# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

stage = "s" + raw_config["stages"] if len(raw_config["stages"]) == 1 else 3


config = {
    "batch_size": raw_config[stage + "_batch_size"],
    "iterations": raw_config[stage + "_iterations"],
    "finetune": raw_config[stage + "_finetune"],
    "steps": raw_config[stage + "_steps"],
    "lr": raw_config[stage + "_lr"],
    "num_ref_frames": raw_config[stage + "_num_ref_frames"],
    "num_frames": raw_config[stage + "_num_frames"],
    "start_warm": raw_config[stage + "_start_warm"],
    "end_warm": raw_config[stage + "_end_warm"],
}

config["num_workers"] = raw_config["num_workers"]

config["deep_update_prob"] = raw_config["deep_update_prob"]
config["weight_decay"] = raw_config["weight_decay"]
config["gamma"] = raw_config["gamma"]
config["amp"] = not raw_config["no_amp"]

config["log_text_interval"] = raw_config["log_text_interval"]
config["log_image_interval"] = raw_config["log_image_interval"]
config["save_network_interval"] = raw_config["save_network_interval"]
config["save_checkpoint_interval"] = raw_config["save_checkpoint_interval"]

config["debug"] = raw_config["debug"]
config["exp_id"] = "EndoVis17_Binary"

long_id = "%s_%s" % (
    datetime.datetime.now().strftime("%b%d_%H.%M.%S"),
    config["exp_id"],
)

git_info = "XMem"
logger = TensorboardLogger(config["exp_id"], long_id, git_info)
logger.log_string("hyperpara", str(config))

model = XMemTrainer(
    config,
    logger=logger,
    save_path=path.join("saves", long_id, long_id) if long_id is not None else None,
    local_rank=0,
    world_size=1,
).train()
model.load_network("../../../external/XMem/saves/XMem.pth")

total_iter = 0


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def renew_loader(video_path, masks_path, max_skip, subset=None, finetune=True):
    dataset = BinaryInstrumentDataset(
        video_path,
        masks_path,
        max_jump=max_skip,
        subset=subset,
        num_frames=config["num_frames"],
        max_num_obj=1,
        finetune=finetune,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, rank=local_rank, shuffle=True
    )
    train_loader = DataLoader(
        dataset,
        config["batch_size"],
        sampler=train_sampler,
        num_workers=config["num_workers"],
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    return train_sampler, train_loader


MAX_SKIP_VALUES = [10, 15, 5, 5]

increase_skip_fraction = [0.1, 0.3, 0.9, 100]

train_subset = []
test_subset = []

for folder in VIDEOS_PATH.iterdir():
    if "test" in folder.name:
        test_subset.append(folder.name)
    else:
        train_subset.append(folder.name)

train_sampler, train_loader = renew_loader(
    VIDEOS_PATH, MASKS_PATH, 20, subset=train_subset, finetune=False
)

total_epoch = math.ceil(config["iterations"] / len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f"Current epoch is {current_epoch}.")
print(f"We approximately use {total_epoch} epochs.")

change_skip_iter = [round(config["iterations"] * f) for f in increase_skip_fraction]
# Skip will only change after an epoch, not in the middle
print(
    f"The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}"
)

finetuning = False
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30 - 1) + local_rank * 100)
try:
    pbar = tqdm(total=config["iterations"] + config["finetune"])
    while total_iter < config["iterations"] + config["finetune"]:

        # Crucial for randomness!
        train_sampler.set_epoch(current_epoch)
        current_epoch += 1
        pbar.update()
        pbar.set_description(f"Epoch {current_epoch}")

        # Train loop
        model.train()
        for data in train_loader:
            # Update skip if needed
            if stage != "0" and total_iter >= change_skip_iter[0]:
                while total_iter >= change_skip_iter[0]:
                    cur_skip = MAX_SKIP_VALUES[0]
                    max_skip_values = MAX_SKIP_VALUES[1:]
                    change_skip_iter = change_skip_iter[1:]
                print(f"Changing skip to {cur_skip=}")
                train_sampler, train_loader = renew_loader(
                    VIDEOS_PATH,
                    MASKS_PATH,
                    cur_skip,
                    subset=train_subset,
                    finetune=False,
                )
                break

            # fine-tune means fewer augmentations to train the sensory memory
            if (
                config["finetune"] > 0
                and not finetuning
                and total_iter >= config["iterations"]
            ):
                train_sampler, train_loader = renew_loader(
                    VIDEOS_PATH,
                    MASKS_PATH,
                    cur_skip,
                    subset=train_subset,
                    finetune=True,
                )
                finetuning = True
                model.save_network_interval = 50
                break

            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= config["iterations"] + config["finetune"]:
                break
    pbar.close()
finally:
    model.save_network(total_iter)
    model.save_checkpoint(total_iter)

print("Training Completed")
