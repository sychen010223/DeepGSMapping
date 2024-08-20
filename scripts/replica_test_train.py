import argparse
import os
import random
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import open3d as o3d
import copy
from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_update_params
from utils.eval_helpers import report_loss, report_progress, eval, eval_psnr
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians
from utils.gs_external import (
    densify, get_expon_lr_func, update_learning_rate
)
from utils.geometry_utils import transform_to_global_KITTI
from utils.vis_2dpose import plot_global_pose

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from gaussian_splatting import get_loss_gs

# L-net
# import set_path
import importlib
import functools
print = functools.partial(print,flush=True)
import torch.optim as optim
import loss
from loss import *
from deepmapping import DeepMapping2
from init_pose import Lnet2pose, w2c_to_c2w
from group_matrix_generate import group_matrix_generate

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    # param_groups = [{'params': params[k], 'name': k, 'lr': v} for k, v in lrs.items() if k in params]

    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam

def initialize_param_curr(dataset, num_frames, intrinsics, scene_radius_depth_ratio, \
                        mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, frame_id=0):
    # Get RGB-D Data & Camera Parameters
    color, depth, _, pose = dataset[frame_id]
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    w2c = torch.linalg.inv(pose)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables
    else:
        return params, variables

def get_Lnet_data(dataset, frame_idx, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[frame_idx]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)

    point_cloud_np, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, transform_pts=False, mask = mask, compute_mean_sq_dist=True, mean_sq_dist_method=mean_sq_dist_method)

    point_cloud_np = point_cloud_np.cpu().detach().numpy()
    # 设置点云的点和颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])  # 提取点云坐标
    # pcd.colors = o3d.utility.Vector3dVector(point_cloud_np[:, 3:])  # 提取点云颜色
    # 保存点云数据为 PCD 文件
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # print(len(pcd.points))
    # o3d.io.write_point_cloud(f'./pcd_save/replica/pcd/{frame_idx:04d}.pcd', pcd)
    # print(f'save {frame_idx:04}.pcd success!')

    pcd_pt = np.asarray(pcd.points)
    return pcd_pt


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params

        
        
def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

def plot_and_save(data_indices, losses, losses_init, epoch, range_start, range_end, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_indices, losses, marker='o', linestyle='-', color='blue', label='Estimate Loss', markersize=3)
    ax.plot(data_indices, losses_init, marker='o', linestyle='-', color='green', label='Initial Loss', markersize=3)
    ax.set_title(f'Loss over Frames {range_start}-{range_end} - Epoch {epoch}')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(f"{save_path}/loss_epochs{epoch}-{range_start}-{range_end}.jpg")
    plt.close()
    
def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(  
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    print("Loading gt poses ... ")
    gt_w2c_all_frames = []
    gt_all_frames_c2w = []
    file_path = f"./pcd_save/{config['run_name']}/gt_all_frames_c2w.npy"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        for time_idx in tqdm(range(num_frames)):
            # Load RGBD frames incrementally instead of all frames
            _, _, _, gt_pose = dataset[time_idx]
            # get gt_c2w to draw traj.
            quat = matrix_to_quaternion(gt_pose[:3, :3].unsqueeze(0).detach())
            trans = gt_pose[:3, 3].unsqueeze(0).detach()
            gt_c2w = torch.cat([trans, quat], dim=1).squeeze(0).detach().cpu().numpy()
            gt_all_frames_c2w.append(gt_c2w)
            # get gt_w2c for evaluation
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, gt_all_frames_c2w)

    else:
        print(f"File already exists. No need to generate again.")
        

    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    norm_pcd = get_Lnet_data(dataset, 0, config['scene_radius_depth_ratio'], config['mean_sq_dist_method'])
    max_num_pcd = norm_pcd.shape[0]

    # Load Checkpoint
    checkpoint_time_idx = config['checkpoint_time_idx']
    print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
    ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
    #params = dict(np.load(f"{config['workdir']}/freiburg1_desk2_seed1/params.npz", allow_pickle=True))
    params = dict(np.load(f"{config['workdir']}/room0_0/params.npz", allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    params_init = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()

    print('number of gaussians:', params['means3D'].shape[0])
    checkpoint_time_idx = 0

    
    # TODO: creating model
    print('creating model')
    model = DeepMapping2(n_points=max_num_pcd, rotation_representation='quaternion', device=device).to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    #scheduler = ExponentialLR(optimizer, gamma=0.95)
    scaler = torch.cuda.amp.GradScaler()
    n_epochs = config['num_epochs']
    num_group = 5
    mode = config['mode']
    track = config['track']
    setdate_GS_per_iter = config['setdate_GS_per_iter']
    update_GS_per_iter = config['update_GS_per_iter']
    GSupdate = config['GSupdate']
    print('mode:', mode)
    print('track:', track)
    print('GSupdate:', GSupdate)

    print('generate group matrix ...')
    full_group_matrix = np.zeros((num_frames, 20))
    #full_group_matrix = np.load(f"./pcd_save/{config['run_name']}/Group_list.npy")
    full_group_matrix = np.load(f"./pcd_save/{config['scene_run']}/Group_list.npy")
    print('full_group_matrix:', full_group_matrix.shape)


    if (mode == 'train'):
        print('load model')
        save_dir = os.path.join('./pcd_save', config['run_name'], f"GSsetup_best_model.pth")
        
        #base_dir = os.path.join('./pcd_save', config['run_name'])
        #train_dir = create_train_subdir(base_dir)
        #save_dir = os.path.join(train_dir, f"GSsetup_best_model.pth")
        
        if os.path.exists(save_dir):
            checkpoint = torch.load(save_dir)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('No model found at:', save_dir)

    initial_pose = []
    for idx in range(num_frames):
                pose = torch.cat((params['cam_trans'][..., idx], params['cam_unnorm_rots'][..., idx]), dim=1).reshape(1, 7).squeeze(0)
                initial_pose.append(pose)
    initial_pose = torch.stack(initial_pose)
    initial_pose = w2c_to_c2w(initial_pose).detach().cpu().numpy() # trans initial pose to c2w
    np.save(f"./pcd_save/{config['run_name']}/initial_pose.npy", initial_pose)

    # Iterate over Scan
    if mode == 'train':
        if track:
            loss_records = []
            loss_init_records = []
            #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
            for epoch in tqdm(range(n_epochs)):
                Lnet_pose_est = []
                current_iters = max(n_epochs - 2 * epoch, 1)
                for time_idx in tqdm(range(num_frames)):
                    # setup_GS_per_iter
                    #if setdate_GS_per_iter:
                    if setdate_GS_per_iter and epoch > 0:
                        num_iters_mapping = 100
                        GSnum_group = 10
                        selected_keyframes = full_group_matrix[time_idx, :GSnum_group].astype(int)
                        # initial 3DGS-params
                        params_curr, variables_curr = initialize_param_curr(dataset, num_frames, intrinsics, config['scene_radius_depth_ratio'], config['mean_sq_dist_method'], gaussian_distribution=config['gaussian_distribution'], frame_id=time_idx)
                        params_curr['cam_unnorm_rots'] = params['cam_unnorm_rots']
                        params_curr['cam_trans'] = params['cam_trans']
                        # add new gaussians
                        for idx in range(1, len(selected_keyframes)):
                            frame_id = selected_keyframes[idx]
                            iter_color, iter_depth, _, gt_pose = dataset[frame_id]
                            iter_color = iter_color.permute(2, 0, 1) / 255
                            iter_depth = iter_depth.permute(2, 0, 1)
                            iter_gt_w2c = gt_w2c_all_frames[:frame_id+1]
                            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 
                                        'id': frame_id, 'intrinsics': intrinsics, 
                                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                            params_curr, variables_curr = add_new_gaussians(params_curr, variables_curr, iter_data, config['mapping']['sil_thres'], frame_id, config['mean_sq_dist_method'], config['gaussian_distribution'])
                        optimizer = initialize_optimizer(params_curr, config['mapping']['lrs'], tracking=False)
                        for iter in tqdm(range(num_iters_mapping)):
                            rand_idx = np.random.randint(0, len(selected_keyframes))
                            iter_time_idx = selected_keyframes[rand_idx]
                            iter_color, iter_depth, _, gt_pose = dataset[iter_time_idx]
                            iter_color = iter_color.permute(2, 0, 1) / 255
                            iter_depth = iter_depth.permute(2, 0, 1)
                            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 
                                        'id': iter_time_idx, 'intrinsics': intrinsics, 
                                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}

                            # Loss for current frame
                            loss, variables_curr, losses = get_loss(params_curr, iter_data, variables_curr, iter_time_idx, config['mapping']['loss_weights'], config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'], config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                            print(loss)
                            loss.backward()
                            with torch.no_grad():
                                # Prune Gaussians
                                if config['mapping']['prune_gaussians']:
                                    params_curr, variables_curr = prune_gaussians(params_curr, variables_curr, optimizer, iter, config['mapping']['pruning_dict'])
                                # Gaussian-Splatting's Gradient-based Densification
                                if config['mapping']['use_gaussian_splatting_densification']:
                                    params_curr, variables_curr = densify(params_curr, variables_curr, optimizer, iter, config['mapping']['densify_dict'])
                                # Optimizer Update
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                    else:
                        params_curr, variables_curr = params, variables

                    # TODO: L-net Gaussian method:
                    if epoch < 1:
                        optimizer = optim.Adam(model.parameters(),lr=1e-4)
                    else:
                        optimizer = optim.Adam(model.parameters(),lr=1e-6)

                    #scheduler = ExponentialLR(optimizer, gamma=0.8)
                    #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
                    temp_pcd = []
                    obs_local = []
                    group_data = []
                    group_matrix = full_group_matrix[time_idx, :num_group].astype(int)
                    #temp = full_group_matrix[time_idx, :10]
                    #np.random.shuffle(temp)
                    #group_matrix = temp[:num_group].astype(int) 
                    for idx in range(len(group_matrix)):
                        # Load RGBD frames incrementally instead of all frames
                        frame_id = group_matrix[idx]
                        color, depth, _, gt_pose = dataset[frame_id]
                        # Process poses
                        gt_w2c = torch.linalg.inv(gt_pose)
                        # Process RGB-D Data
                        color = color.permute(2, 0, 1) / 255
                        depth = depth.permute(2, 0, 1)
                        curr_gt_w2c = gt_w2c_all_frames[:frame_id + 1]
                        # Optimize only current time step for tracking
                        iter_time_idx = frame_id
                        # Initialize Mapping Data for selected frame
                        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                        group_data.append(curr_data)
                    for i in group_matrix:
                        temp_pcd = get_Lnet_data(dataset, i, config['scene_radius_depth_ratio'], config['mean_sq_dist_method'])
                        if(temp_pcd.shape[0] > max_num_pcd):
                            temp_pcd = temp_pcd[:max_num_pcd]
                        elif(temp_pcd.shape[0] < max_num_pcd):
                            temp_pcd = np.pad(temp_pcd, ((0, max_num_pcd-temp_pcd.shape[0]), (0, 0)))
                        # print(temp_pcd.shape)
                        obs_local.append(temp_pcd)
                    obs_local = torch.from_numpy(np.stack(obs_local)).float().to(device)
                    # print(obs_local.shape)

                    time_start = time.time()
                    loss_init = 0
                    flag = 0
                    flag_lr = True
                    loss = 0
                    lr = 0
                    lr_change = 1
                    loss_ = []
                    model.train()
                    min_loss = float('inf')
                    best_model_params = None
                    if epoch < 1:
                                
                        for iter in tqdm(range(10)):
                            loss_init, flag, loss = model(obs_local, params_curr, group_matrix, group_data, variables_curr, loss_init=loss_init, flag=flag, params_init=params_init)

                            for param_group in optimizer.param_groups:
                                if not torch.isnan(loss) and flag == 0:
                                    param_group['lr'] = 3e-4
                                elif not torch.isnan(loss) and flag == 1:
                                    param_group['lr'] = config['tracking']['lr']
                                    print("Current learning rate:", param_group['lr'])
                                    #param_group['lr'] = 1e-6
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            if not torch.isnan(loss) and loss < min_loss:
                                min_loss = loss
                                best_model_params = copy.deepcopy(model.state_dict())
                            #scheduler.step()
                            #cuent_lr = optimizer.param_groups[0]['lr']
                            #print("Current learning rate:", current_lr)
                            print(f'frame{time_idx} est loss:', float(loss))
                            print(f'frame{time_idx} init loss:', float(loss_init))
                            # loss_records.append((time_idx, float(loss)))
                            # loss_init_records.append((time_idx, float(loss_init)))

                    else:
                            
                        for iter in tqdm(range(10)):
                            loss_init, flag, loss = model(obs_local, params_curr, group_matrix, group_data, variables_curr, loss_init=loss_init, flag=flag, params_init=params_init)   

                            for param_group in optimizer.param_groups:
                                print("Current learning rate:", param_group['lr'])
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            if not torch.isnan(loss) and loss < min_loss:
                                min_loss = loss
                                best_model_params = copy.deepcopy(model.state_dict())
                            #scheduler.step()
                            #current_lr = optimizer.param_groups[0]['lr']
                            #print("Current learning rate:", current_lr)
                            print(f'frame{time_idx} est loss:', float(loss))
                            print(f'frame{time_idx} init loss:', float(loss_init))
                    model.load_state_dict(best_model_params)
                    print('----------------------------------------')
                    print(f'Lowest loss achieved: {min_loss}')
                    print(f'frame{time_idx} init loss:', float(loss_init))
                    loss_records.append((time_idx, float(min_loss)))
                    loss_init_records.append((time_idx, float(loss_init)))
                    #scheduler.step()
                    with torch.no_grad():
                        pose_est = model.pose_est
                        Lnet_pose_est.append(pose_est[0])
                        if loss < loss_init:
                            params['cam_unnorm_rots'][..., group_matrix[0]] = pose_est[0, 3:].float() 
                            params['cam_trans'][..., group_matrix[0]] = pose_est[0, :3].float()
          
                    # update_GS_per_iter
                    if update_GS_per_iter:
                        # update GS
                        num_iters_mapping = 60
                        GSnum_group = 10
                        selected_keyframes = full_group_matrix[time_idx, :GSnum_group].astype(int)
                        optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
                        for iter in tqdm(range(num_iters_mapping)):
                            rand_idx = np.random.randint(0, len(selected_keyframes))
                            iter_time_idx = selected_keyframes[rand_idx]
                            iter_color, iter_depth, _, gt_pose = dataset[iter_time_idx]
                            iter_color = iter_color.permute(2, 0, 1) / 255
                            iter_depth = iter_depth.permute(2, 0, 1)
                            iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                            iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 
                                        'id': iter_time_idx, 'intrinsics': intrinsics, 
                                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}

                            # Loss for current frame
                            loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'], config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'], config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                            print('loss:', float(loss))
                            loss.backward()
                            with torch.no_grad():
                                # Prune Gaussians
                                if config['mapping']['prune_gaussians']:
                                    params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                                # Gaussian-Splatting's Gradient-based Densification
                                if config['mapping']['use_gaussian_splatting_densification']:
                                    params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                                # Optimizer Update
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)

                #scheduler.step()
                # save L-net model & Est_poses
                save_dir = os.path.join('./pcd_save', config['run_name'], f"GSsetup_best_model.pth")
                model_update = os.path.join('./pcd_save', config['run_name'], f"modelepoch{epoch}.pth")
                #save_dir = os.path.join(train_dir, "GSsetup_best_model.pth")
                state = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch}
                torch.save(state, save_dir)
                torch.save(state, model_update)
                print('model saved to {}'.format(save_dir))
                print('model saved to {}'.format(model_update))

                # save Est_poses
                Lnet_pose_est = torch.stack(Lnet_pose_est)
                Lnet_pose_est = w2c_to_c2w(Lnet_pose_est).detach().cpu().numpy() # change w2c pose to c2w
                save_est_dir = os.path.join('./pcd_save', config['run_name'], f"GSsetup_Lnet_est_pose.npy")
                est_pose_save = os.path.join('./pcd_save', config['run_name'], f"GSsetup_Lnet_est_pose{epoch}.npy")
                #save_est_dir = os.path.join(train_dir, "GSsetup_Lnet_est_pose.npy")
                np.save(save_est_dir, Lnet_pose_est)
                np.save(est_pose_save, Lnet_pose_est)
                print('Est_poses saved to {}'.format(save_est_dir))
                # draw traj.
                Lnet_est_pose = np.load(f"./pcd_save/{config['run_name']}/GSsetup_Lnet_est_pose.npy")
                #Lnet_est_pose = np.load(f"./pcd_save/{config['run_name']}/GSsetup_Lnet_est_pose{epoch}.npy")
                initial_pose = np.load(f"./pcd_save/{config['run_name']}/initial_pose.npy")
                gt_pose = np.load(f"./pcd_save/{config['run_name']}/gt_all_frames_c2w.npy")
                
                
                #plot_global_pose(Lnet_est_pose, output_dir=f"./results/{config['run_name']}", Init_location=initial_pose, gt_location=None)
                plot_global_pose(Lnet_est_pose, output_dir=f"./results/{config['run_name']}", Init_location=initial_pose, gt_location=gt_pose, epoch = epoch)

                # Save Parameters
                output_dir = os.path.join('./pcd_save', config['run_name'])
                save_update_params(params, output_dir, epoch=epoch)
                
                # save loss
                loss_records = sorted(loss_records, key=lambda x: x[0])
                time_indices, loss_est = zip(*loss_records)  # 解包
                loss_init_records = sorted(loss_init_records, key=lambda x: x[0])
                _ , losses_init = zip(*loss_init_records) 
                # plt.figure(figsize=(10, 5))
                
 
                ranges = [(100,150), (150, 200), (200, 250), (250, 300), (0,300)]

                for start, end in ranges:
                    indices = [i for i, t in enumerate(time_indices) if start <= t <= end]
                    if indices:
                        range_indices = time_indices[indices[0]:indices[-1]+1]
                        range_losses = loss_est[indices[0]:indices[-1]+1]
                        range_losses_init = losses_init[indices[0]:indices[-1]+1]
                        save_path = f"./pcd_save/{config['run_name']}"
                        plot_and_save(range_indices, range_losses, range_losses_init, epoch, start, end, save_path)               
                loss_records = []
                loss_init_records = []

                # change to eval mode to get estimate poses
                Lnet_pose_est_eval = []
                for test_time_idx in tqdm(range(num_frames)):
                    temp_pcd = []
                    obs_local = []
                    true_pcd = []                                                  
                    group_matrix = [test_time_idx]
                    group_data = []
                    for idx in range(len(group_matrix)):
                        # Load RGBD frames incrementally instead of all frames
                        frame_id = group_matrix[idx]
                        color, depth, _, gt_iter_pose = dataset[frame_id]
                        # Process poses
                        gt_w2c = torch.linalg.inv(gt_iter_pose)
                        # Process RGB-D Data
                        color = color.permute(2, 0, 1) / 255
                        depth = depth.permute(2, 0, 1)
                        curr_gt_w2c = gt_w2c_all_frames[:frame_id + 1]
                        # Optimize only current time step for tracking
                        iter_time_idx = frame_id
                        # Initialize Mapping Data for selected frame
                        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                        group_data.append(curr_data)
                    for i in group_matrix:
                        temp_pcd = get_Lnet_data(dataset, i, config['scene_radius_depth_ratio'], config['mean_sq_dist_method'])
                        if(temp_pcd.shape[0] > max_num_pcd):
                            temp_pcd = temp_pcd[:max_num_pcd]
                        elif(temp_pcd.shape[0] < max_num_pcd):
                            temp_pcd = np.pad(temp_pcd, ((0, max_num_pcd-temp_pcd.shape[0]), (0, 0)))
                        # print(temp_pcd.shape) 
                        obs_local.append(temp_pcd)
                    obs_local = torch.from_numpy(np.stack(obs_local)).float().to(device)
                    # print(obs_local.shape)

                    with torch.no_grad():
                        model.eval()
                        model(obs_local, params_curr, group_matrix, group_data, variables_curr, loss_init=loss_init, flag=flag, params_init=params_init)
                        pose_est = model.pose_est
                        Lnet_pose_est_eval.append(pose_est[0])
                        params['cam_unnorm_rots'][..., test_time_idx] = pose_est[0][3:]
                        params['cam_trans'][..., test_time_idx] = pose_est[0][:3]
                        print(Lnet_pose_est_eval[-1])
                        
                        
                save_dir = os.path.join('./results', config['run_name'], f"eval/poses/est_pose_weighted_{epoch}.npy")
                if not os.path.exists(os.path.dirname(save_dir)):
                    os.makedirs(os.path.dirname(save_dir))
                Lnet_pose_est_eval = torch.stack(Lnet_pose_est_eval)
                Lnet_pose_est_eval = w2c_to_c2w(Lnet_pose_est_eval).detach().cpu().numpy() # change w2c pose to c2w
                np.save(save_dir, Lnet_pose_est_eval)
                print('eval Est_poses saved to {}'.format(save_dir))
                # draw traj.
                # Lnet_est_pose = np.load(save_dir)
                initial_pose = np.load(f"./pcd_save/{config['run_name']}/initial_pose.npy")
                gt_pose = np.load(f"./pcd_save/{config['run_name']}/gt_all_frames_c2w.npy")
                plot_global_pose(Lnet_pose_est_eval, output_dir=f"./results/{config['run_name']}/eval", Init_location=initial_pose, gt_location=gt_pose, epoch=epoch)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)