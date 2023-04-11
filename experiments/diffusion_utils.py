import math
from typing import List, Optional, Sequence, Union

import imageio
import logging
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.distributions import Normal
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pytorch3d.renderer import (
    AlphaCompositor, 
    NormWeightedCompositor, 
    OrthographicCameras,
    PointsRasterizationSettings, 
    PointsRasterizer,
    PointsRenderer, 
    look_at_view_transform)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch


# Disable unnecessary imageio logging
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:, [1, 2, 0]].dot(M).dot(N).dot(K), faces[:, [1, 2, 0]]
    return v, f


def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5

    return v, f


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


@torch.no_grad()
def visualize_distance_transform(
    path_stem: str,
    images: torch.Tensor,
) -> str:
    output_file_image = f'{path_stem}.png'
    if images.shape[3] in [1, 3]:  # convert to (B, C, H, W)
        images = images.permute(0, 3, 1, 2)
    images = images[:, -1:]  # (B, 1, H, W)  # get only distances (not vectors for now, for simplicity)
    image_grid = make_grid(images, nrow=int(math.sqrt(len(images))), pad_value=1, normalize=True)
    to_pil_image(image_grid).save(output_file_image)
    return output_file_image


@torch.no_grad()
def visualize_image(
    path_stem: str,
    images: torch.Tensor,
    mean: Union[torch.Tensor, float] = 0.5,
    std: Union[torch.Tensor, float] = 0.5,
) -> str:
    output_file_image = f'{path_stem}.png'
    if images.shape[3] in [1, 3, 4]:  # convert to (B, C, H, W)
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] in [3, 4]:  # normalize (single-channel images are not normalized)
        images[:, :3] = images[:, :3] * std + mean  # denormalize (color channels only, not alpha channel)
    if images.shape[1] == 4:  # normalize (single-channel images are not normalized)
        image_alpha = images[:, 3:]  # (B, 1, H, W)
        bg_color = torch.tensor([230, 220, 250], device=images.device).reshape(1, 3, 1, 1) / 255
        images = images[:, :3] * image_alpha + bg_color * (1 - image_alpha)  # (B, 3, H, W)
    image_grid = make_grid(images, nrow=int(math.sqrt(len(images))), pad_value=1)
    to_pil_image(image_grid).save(output_file_image)
    return output_file_image


def ensure_point_cloud_has_colors(pointcloud: Pointclouds):
    if pointcloud.features_padded() is None:
        pointcloud = type(pointcloud)(points=pointcloud.points_padded(), 
            normals=pointcloud.normals_padded(), features=torch.zeros_like(pointcloud.points_padded()))
    return pointcloud


@torch.no_grad()
def render_pointcloud_batch_pytorch3d(
    cameras: CamerasBase,
    pointclouds: Pointclouds,
    image_size: int = 224,
    radius: float = 0.01, 
    points_per_pixel: int = 10, 
    background_color: Sequence[float] = (0.78431373, 0.78431373, 0.78431373),
    compositor: str = 'norm_weighted'
):
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to rasterize_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
    )

    # Rasterizer
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # Compositor
    if compositor == 'alpha':
        compositor = AlphaCompositor(background_color=background_color)
    elif compositor == 'norm_weighted':
        compositor = NormWeightedCompositor(background_color=background_color)
    else:
        raise ValueError(compositor)

    # Create a points renderer by compositing points using an weighted compositor (3D points are
    # weighted according to their distance to a pixel and accumulated using a weighted sum)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # We cannot render a point cloud without colors, so add them if the pointcloud does 
    # not already have them
    pointclouds = ensure_point_cloud_has_colors(pointclouds)

    # Render batch of image
    images = renderer(pointclouds)

    return images


@torch.no_grad()
def visualize_pointcloud_batch_pytorch3d(
    pointclouds: Pointclouds,
    output_file_video: Optional[str] = None,
    output_file_image: Optional[str] = None,
    cameras: Optional[CamerasBase] = None,  # if None, we rotate
    scale_factor: float = 1.0,
    num_frames: int = 1,  # note that it takes a while with 30 * batch_size frames
    elev: int = 30,
):
    """Saves a video and a single image of a point cloud"""
    assert 360 % num_frames == 0, 'please select a better number of frames'
    
    # Sizes
    B, N, C, F = *(pointclouds.points_padded().shape), num_frames
    device = pointclouds.device

    # If a camera has not been provided, we render from a rotating view around an image
    if cameras is None:

        # Create view transforms - R is (F, 3, 3) and T is (F, 3)
        R, T = look_at_view_transform(dist=10.0, elev=elev, azim=list(range(0, 360, 360 // F)), degrees=True, device=device)

        # Repeat
        R = R.repeat_interleave(B, dim=0)  # (F * B, 3, 3)
        T = T.repeat_interleave(B, dim=0)  # (F * B, 3)
        points = pointclouds.points_padded().tile(F, 1, 1)  # (F * B, num_points, 3)
        colors = (torch.zeros_like(points) if pointclouds.features_padded() is None else 
                  pointclouds.features_padded().tile(F, 1, 1))  # (F * B, num_points, 3)
        
        # Initialize batch of cameras
        cameras = OrthographicCameras(focal_length=(0.25 * scale_factor), device=device, R=R, T=T)

        # Wrap in Pointclouds (with color, even if the original point cloud had no color)
        pointclouds = Pointclouds(points=points, features=colors).to(device)

    # Render image
    images = render_pointcloud_batch_pytorch3d(cameras, pointclouds)
        
    # Convert images into grid
    image_grids = []
    images_for_grids = images.reshape(F, B, *images.shape[1:]).permute(0, 1, 4, 2, 3)
    for image_for_grids in images_for_grids:
        image_grid = make_grid(image_for_grids, nrow=int(math.sqrt(B)), pad_value=1)
        image_grids.append(image_grid)
    image_grids = torch.stack(image_grids, dim=0)
    image_grids = image_grids.detach().cpu()

    # Save image
    if output_file_image is not None:
        to_pil_image(image_grids[0]).save(output_file_image)

    # Save video
    if output_file_video:
        video = (image_grids * 255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
        imageio.mimwrite(output_file_video, video, fps=10)


@torch.no_grad()
def visualize_pointcloud_evolution_pytorch3d(
    pointclouds: Pointclouds,
    output_file_video: str,
    camera: Optional[CamerasBase] = None,  # if None, we rotate
    scale_factor: float = 1.0,
):

    # Device
    B, device = len(pointclouds), pointclouds.device

    # Cameras
    if camera is None:
        R, T = look_at_view_transform(dist=10.0, elev=30, azim=0, device=device)
        camera = OrthographicCameras(focal_length=(0.25 * scale_factor), device=device, R=R, T=T)
    
    # Render
    frames = render_pointcloud_batch_pytorch3d(camera, pointclouds)

    # Save video
    video = (frames.detach().cpu() * 255).to(torch.uint8).numpy()
    imageio.mimwrite(output_file_video, video, fps=10)


def get_camera_index(cameras: CamerasBase, index: Optional[int] = None):
    if index is None:
        return cameras
    kwargs = dict(
        R=cameras.R[index].unsqueeze(0),
        T=cameras.T[index].unsqueeze(0),
        K=cameras.K[index].unsqueeze(0) if cameras.K is not None else None,
    )
    if hasattr(cameras, 'focal_length'):
        kwargs['focal_length'] = cameras.focal_length[index].unsqueeze(0)
    if hasattr(cameras, 'principal_point'):
        kwargs['principal_point'] = cameras.principal_point[index].unsqueeze(0)
    return type(cameras)(**kwargs).to(cameras.device)


def get_metadata(item) -> str:
    s = '-------------\n'
    for key in item.keys():
        value = item[key]
        if torch.is_tensor(value) and value.numel() < 25:
            value_str = value
        elif torch.is_tensor(value):
            value_str = value.shape
        elif isinstance(value, str):
            value_str = value
        elif isinstance(value, list) and 0 < len(value) and len(value) < 25 and isinstance(value[0], str):
            value_str = value
        elif isinstance(value, dict):
            value_str = str({k: type(v) for k, v in value.items()})
        else:
            value_str = type(value)
        s += f"{key:<30} {value_str}\n"
    return s
