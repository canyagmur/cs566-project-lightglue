import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf


from gluefactory.multipoint.datasets import ImagePairDataset

from gluefactory.geometry.wrappers import Camera, Pose
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import DATA_PATH
from gluefactory.utils.image import ImagePreprocessor, load_image
from gluefactory.utils.tools import fork_rng
from gluefactory.visualization.viz2d import plot_heatmaps, plot_image_grid
from gluefactory.datasets.base_dataset import BaseDataset
from gluefactory.datasets.utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"



def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class MPDataset(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "megadepth/",
        "depth_subpath": "depth_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",  # @TODO: intrinsics problem?
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": "test_scenes_clean.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  # only with views==3
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        'mp_path': '../xpoint-beta-github/configs/cmt-srhenlighter-test.yaml',
        # features from cache
        "load_features": {
            "do": False,
            "path": None
        },
    }

    def _init(self, conf):
        pass


    def get_dataset(self, split):
        return _PairDataset(self.conf, split)


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        self.conf = conf

        import yaml
        with open(self.conf.mp_path, 'r') as f:
            self.mp_config = yaml.load(f, Loader=yaml.FullLoader)
        self.mydataset = ImagePairDataset( self.mp_config['dataset'])
        self.image_shape = torch.Tensor([self.mydataset[0]['optical']['image'].shape[1],self.mydataset[0]['optical']['image'].shape[2]])



    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):

        #print(self.mydataset[idx].keys())
        data0 = {"cache": {},"image": self.mydataset[idx]['optical']['image'],"scales":torch.Tensor([1.0,1.0]),"image_size":self.image_shape,"original_image_size":self.image_shape,"transform":self.mydataset[idx]['optical']['homography']} #1,256,256 since 1 is channel dim but be aware the problem!
        data1 = {"cache": {},"image": self.mydataset[idx]['thermal']['image'],"scales":torch.Tensor([1.0,1.0]),"image_size":self.image_shape,"original_image_size":self.image_shape,"transform":self.mydataset[idx]['thermal']['homography']}

        if self.conf.load_features.do:
            data0["cache"]["keypoints"] = self.mydataset[idx]['optical']["keypoints"]
            data0["cache"]["descriptors"] = self.mydataset[idx]["optical"]["descriptor"]
            #data0["cache"]["keypoint_scores"] = self.mydataset[idx]["optical"]["keypoint_scores"]

            data1["cache"]["keypoints"] = self.mydataset[idx]['thermal']["keypoints"]
            data1["cache"]["descriptors"] = self.mydataset[idx]["thermal"]["descriptor"]
            #data1["cache"]["keypoint_scores"] = self.mydataset[idx]["thermal"]["keypoint_scores"]

            #data0["cache"][""]
        data = {
            "view0": data0,
            "view1": data1,
        }
        data["name"] = self.mydataset[idx]['name']
        data["idx"] = idx
        data["scales"] = torch.Tensor([1.0,1.0])


        # Calculate the inverse of the optical homography
        H_optical_inv = np.linalg.inv(self.mydataset[idx]['optical']['homography'])

        # Compute the homography from optical to thermal
        H_optical_to_thermal = self.mydataset[idx]['thermal']['homography'] @ H_optical_inv

        data["H_0to1"] = H_optical_to_thermal
        
        return data

    def __len__(self):
        return len(self.mydataset)






def visualize(args):
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MPDataset(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info("The dataset has elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    plt.show()


if __name__ == "__main__":
    # from gluefactory import logger  # overwrite the logger
    #import yaml
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-y', '--yaml-config', default='../xpoint-beta-github/configs/cmt.yaml', help='YAML config file')
    #args = parser.parse_args()


    # parser.add_argument("--split", type=str, default="val")
    # parser.add_argument("--num_items", type=int, default=4)
    # parser.add_argument("--dpi", type=int, default=100)
    # parser.add_argument("dotlist", nargs="*")
    # args = parser.parse_intermixed_args()
    # visualize(args)
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    dataset = MPDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has elements.", len(loader))

