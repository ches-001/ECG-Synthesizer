import os
import cv2
import cc3d
import qrcode
import heapq
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datetime import datetime, timedelta
from scipy.signal import fftconvolve
from pycocotools import mask as mask_utils
from typing import Tuple, Optional, List, Union, Dict, Any, Literal

SCALE               = 1
THICK_GRID_SPACE_MM = 5
THIN_GRID_SPACE_MM  = 1
THICK_GRID_SPACE_PX = 40 * SCALE
THIN_GRID_SPACE_PX  = 8 * SCALE
NUM_THICK_VLINES    = 56
NUM_THICK_HLINES    = 43
GRID_H              = THICK_GRID_SPACE_PX * NUM_THICK_HLINES
GRID_W              = THICK_GRID_SPACE_PX * NUM_THICK_VLINES
MM_PER_SEC          = 25
MM_PER_MV           = 10
SIGNAL_DURATION     = 10
R1_COLS             = ("I", "aVR", "V1", "V4")
R2_COLS             = ("II", "aVL", "V2", "V5")
R3_COLS             = ("III", "aVF", "V3", "V6")
R4_COLS             = ("II", )


def global_change_scale(scale: int) -> int:
    global SCALE
    
    if scale == SCALE:
        return scale
    
    global THICK_GRID_SPACE_PX
    global THIN_GRID_SPACE_PX
    global GRID_H
    global GRID_W

    SCALE               = scale
    THICK_GRID_SPACE_PX = 40 * SCALE
    THIN_GRID_SPACE_PX  = 8 * SCALE
    GRID_H              = THICK_GRID_SPACE_PX * NUM_THICK_HLINES
    GRID_W              = THICK_GRID_SPACE_PX * NUM_THICK_VLINES
    return SCALE


def apply_grid_padding(
        img: np.ndarray, 
        top: int=0, 
        bottom: int=0, 
        left: int=0, 
        right: int=0, 
        pad_val: Union[int, Tuple[int, int, int]]=255
    ) -> np.ndarray:
    h, w, c = img.shape
    if top:
        pad = np.zeros((top, w, c), dtype=img.dtype)
        pad[:, :, 3:] = 0
        pad[:, :, :3] = pad_val
        img = np.concatenate([pad, img], axis=0)
        h += top

    if bottom:
        pad = np.zeros((bottom, w, c), dtype=img.dtype)
        pad[:, :, 3:] = 0
        pad[:, :, :3] = pad_val
        img = np.concatenate([img, pad], axis=0)
        h += bottom

    if left:
        pad = np.zeros((h, left, c), dtype=img.dtype)
        pad[:, :, 3:] = 0
        pad[:, :, :3] = pad_val
        img = np.concatenate([pad, img], axis=1)
        w += left

    if right:
        pad = np.zeros((h, right, c), dtype=img.dtype)
        pad[:, :, 3:] = 0
        pad[:, :, :3] = pad_val
        img = np.concatenate([img, pad], axis=1)
    return img


def apply_background(
        img: np.ndarray, 
        bg_color: Tuple[int, int, int],
        bg_color_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]=None,
        size: Optional[Tuple[int, int]]=None,
        background_dir: Optional[str]=None,
        background_file: Optional[str]=None,
        background: Optional[np.ndarray]=None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add fancy background to image resize background to match size if size is specified else use background as is, 
    then pad image with 0s to match background size then overlay image on background. This program expects the 
    background size to be greater than the image size.

    NOTE: bg_color_range Is used to specify a range of color intensities to set to background pixels, normally
        bg_color would be enough, however, I noticed that due to the bilinear interpolation from grid
        sampling (during sinusoidal distortion and foreshortening), the "bg_color" values get interpolated and
        they are no more exactly equal to bg_color, as such the bg_color_range is used to handle such cases.
        It is also necessary that if bg_color_range is specified, then bg_color should be within the specified
        range incase of padding.
    """
    if background is None:
        if background_file:
            background = cv2.imread(background_file)
        else:
            texture_path = os.path.join(background_dir, random.choice(os.listdir(background_dir)))
            background = cv2.imread(texture_path)
        background = background[:, :, ::-1]

    H, W = img.shape[:2]

    if size is not None:
        bgH, bgW = size
        background = cv2.resize(background, dsize=(bgW, bgH))
    else:
        bgH, bgW = background.shape[:2]
    
    assert bgH >= H and bgW >= W

    dH = bgH - H
    dW = bgW - W
    if dH > 0 or dW > 0:
        img = apply_grid_padding(
            img, 
            top=dH//2+dH%2,
            bottom=dH//2, 
            left=dW//2+dW%2,
            right=dW//2, 
            pad_val=bg_color
        )
    bg_mask = (img[: ,:, :3] == bg_color).all(axis=2)[:, :, None]
    img[:, :, :3] = background * bg_mask + (1 - bg_mask) * img[:, :, :3]

    if bg_color_range is not None:
        rmask = (cv2.inRange(img[:, :, :3], *bg_color_range) > 0)[:, :, None]
        img[:, :, :3] = background * rmask + (1 - rmask) * img[:, :, :3]
        bg_mask |= rmask

    return img, bg_mask[:, :, 0]


def crop_grid(
        img: np.ndarray, 
        vrange: Optional[Tuple[int, int]]=None, 
        hrange: Optional[Tuple[int, int]]=None
    ) -> np.ndarray:
    
    return img[
        slice(*((vrange, ) if not vrange else vrange)), 
        slice(*((hrange, ) if not hrange else hrange)), 
    :]


def divide_unit_circle(n: int=12) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, num=n)
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return vectors


def get_perlin_gradients(coords: np.ndarray, perm: np.ndarray, grad_vecs: np.ndarray) -> np.ndarray:
    hashes = perm[(perm[coords[..., 0] & 255] + (coords[..., 1] & 255)) & 255]
    return grad_vecs[hashes % grad_vecs.shape[0]]


def generate_perlin_noise(
        img_size: Tuple[int, int], 
        grid_size: int,
        perm: Optional[np.ndarray]=None, 
        grad_vecs: Optional[np.ndarray]=None,
        normalize: bool=True,
    ) -> np.ndarray:
    """LINK: https://adrianb.io/2014/08/09/perlinnoise.html"""

    if perm is None:
        perm = np.arange(256)
        np.random.shuffle(perm)
    else:
        assert perm.shape[0] == 256

    if grad_vecs is None:
        grad_vecs = divide_unit_circle(12)
    else:
        assert grad_vecs.shape <= 256

    orig_h, orig_w = img_size
    nh, nw = orig_h // grid_size, orig_w // grid_size
    h, w = nh * grid_size, nw * grid_size

    xs = np.arange(0, w)
    ys = np.arange(0, h)
    xs, ys = np.meshgrid(xs, ys)
    xy = np.stack([xs, ys], axis=2)

    corners = np.stack([
        a := xy // grid_size * grid_size,
        a + [grid_size, 0],
        a + [0, grid_size],
        a + [grid_size, grid_size]
    ], axis=0)

    # fractional coordinates (range from 0 to 1)
    frac_xy = (xy - corners[0, ...]) / grid_size

    frac_corners = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])
    dis = frac_xy - frac_corners[:, None, None, :]
    corner_grads = get_perlin_gradients(corners, perm, grad_vecs)
    dot_prods = corner_grads[..., 0] * dis[..., 0] + corner_grads[..., 1] * dis[..., 1]

    # smoothen with a quintic function, to fade / smoothen edges and avoid grid-like artifacts
    frac_xy = 6 * frac_xy**5 - 15 * frac_xy**4 + 10 * frac_xy**3

    # interpolate relative x position of pixel from top left to top right to get top lerp value
    # interpolate relative x position of pixel from bottom left to bottom right to get bottom lerp value
    # interpolate relative y position of pixel from top lerp to bottom lerp to get perlin noise for pixel
    lerp_top = (dot_prods[0] * (1 - frac_xy[..., 0])) + (dot_prods[1] * frac_xy[..., 0])
    lerp_bottom = (dot_prods[2] * (1 - frac_xy[..., 0])) + (dot_prods[3] * frac_xy[..., 0])
    perlin = lerp_top * (1 - frac_xy[..., 1]) + (lerp_bottom * frac_xy[..., 1])

    if normalize:
        pmin = perlin.min()
        perlin = (perlin - pmin) / (perlin.max() - pmin)
        
    if h != orig_h or w != orig_w:
        perlin = cv2.resize(perlin, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return perlin


def apply_noise(
        img: np.ndarray, 
        noise_type: str="gaussian",
        mean: float=0.0,
        std: float=0.2,
        poisson_rate: float=15.0,
        n_snp: float=10_000,
        salt_val: Tuple[int, int, int]=(255, 255, 255),
        pepper_val: Tuple[int, int, int]=(0, 0, 0),
        perlin_grid_size: int=250,
        perlin_alpha: float=0.3,
        one_channel_noise: bool=False,
        ignore_mask: Optional[np.ndarray]=None
    ) -> np.ndarray:

    h, w, c = img.shape
    dtype = img.dtype
    img = img.astype(np.float32) / 255
    noise_c = [c, 1][one_channel_noise]
    
    if noise_type == "gaussian":
        noise = np.random.normal(loc=mean, scale=std, size=(h, w, noise_c))
        img = img + noise
    
    elif noise_type == "speckle":
        noise = np.random.normal(loc=mean, scale=std, size=(h, w, noise_c))
        img = img * (img + noise)
    
    elif noise_type == "poisson":
        noise = np.random.poisson(poisson_rate, size=(h, w, noise_c)) / 255
        img = img + noise
    
    elif noise_type == "snp":
        yidx = np.random.randint(0, h-1, size=(n_snp, ))
        xidx = np.random.randint(0, w-1, size=(n_snp, ))
        img[yidx, xidx, :] = np.asarray(salt_val) / 255

        yidx = np.random.randint(0, h-1, size=(n_snp, ))
        xidx = np.random.randint(0, w-1, size=(n_snp, ))
        img[yidx, xidx, :] = np.asarray(pepper_val) / 255

    elif noise_type == "perlin":
        # ignore_mask is only used for perlin noise
        noise = generate_perlin_noise((h, w), perlin_grid_size, normalize=True)

        if ignore_mask is not None:
            ignore_mask = np.broadcast_to(ignore_mask[:, :, None], (h, w, c))
            noise = np.tile(noise[:, :, None], (1, 1, c))
            noise[ignore_mask] = img[ignore_mask]
        else:
            noise = noise[:, :, None]

        img = img * (1 - perlin_alpha) + noise * perlin_alpha
    
    return np.clip(img * 255, a_min=0, a_max=255).astype(dtype)


def grid_sample(
        img: np.ndarray, 
        grid_xy: np.ndarray, 
        size: Tuple[int, int], 
        sample_mode: str="bilinear",
        padding_mode: str="zeros", 
        device: str="cpu",
        center_points: bool=False,
        align_corners: bool=True,
        oob_padvals: Optional[List[int]]=None,
        oob_ch_start: Optional[int]=None,
        oob_ch_stop: Optional[int]=None,
    ) -> np.ndarray:

    orig_dtype = img.dtype
    h, w, _ = img.shape

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    if not grid_xy.flags["C_CONTIGUOUS"]:
        grid_xy = np.ascontiguousarray(grid_xy)

    if center_points:
        xy_min = grid_xy.min(axis=(0, 1))
        grid_xy = (grid_xy - xy_min)
        offset = ([w, h] - grid_xy.max(axis=(0, 1))) / 2
        grid_xy = grid_xy + offset

    grid_xy = torch.from_numpy(2 * (grid_xy / [w-1, h-1]) - 1).to(device=device, dtype=torch.float32)
    
    grid_xy = F.interpolate(
        grid_xy[None].permute(0, 3, 1, 2), 
        size, 
        mode="bilinear", 
        align_corners=align_corners
    )[0].permute(1, 2, 0)

    img  = torch.from_numpy(img).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    img = F.grid_sample(
        img[None], 
        grid_xy[None], 
        mode=sample_mode, 
        padding_mode=padding_mode, 
        align_corners=align_corners
    )[0]

    # pad out of bound grid points for specified channel range
    if oob_padvals is not None and isinstance(oob_ch_start, int) and isinstance(oob_ch_stop, int):
        assert oob_ch_stop - oob_ch_start == len(oob_padvals)
        oob_mask = (
            (grid_xy[..., 0] < -1) |
            (grid_xy[..., 0] > 1)  |
            (grid_xy[..., 1] < -1) |
            (grid_xy[..., 1] > 1)
        )[None, :, :]
        img[oob_ch_start:oob_ch_stop, :, :] += (
            oob_mask * torch.tensor(oob_padvals, dtype=img.dtype).reshape(-1, 1, 1)
        )
    img = img.permute(1, 2, 0).contiguous().cpu().numpy().astype(orig_dtype)
    return img


def defuzzify(
        mask: np.ndarray, 
        threshold: Union[int, float]=70, 
        max_val: Union[int, float]=255,
    ) -> np.ndarray:
    """
    remove unnecessary values between 0s and max_vals to change from fuzzy to binary
    """
    mask[m := mask < threshold] = 0
    mask[~m] = max_val
    return mask


def fast_gaussian_blur(
        img: np.ndarray, 
        kernel_size: Tuple[int, int], 
        sigmas: Tuple[int, int]
    ) -> np.ndarray:
    # I made this specfically for scenarios where kernel_size is (0, 0), or kernel size is very large
    # like in smooth random distortion
    orig_dtype = img.dtype
    if kernel_size == (0, 0):
        kh = int(6 * sigmas[1] + 1)
        kw = int(6 * sigmas[0] + 1)
        kh += (kh % 2 == 0)
        kw += (kw % 2 == 0)
        kernel_size = (kw, kh)
    else:
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

    ky = cv2.getGaussianKernel(kernel_size[1], sigma=sigmas[1])
    kx = cv2.getGaussianKernel(kernel_size[0], sigma=sigmas[0])
    kernel = ky @ kx.T
    pad_y, pad_x = (kernel_size[1]-1)//2, (kernel_size[0]-1)//2

    if img.ndim == 2:
        img = np.pad(img, [(pad_y, pad_y), (pad_x, pad_x)], mode="constant", constant_values=0)
        out = fftconvolve(img, kernel, mode="valid")
    else:
        img = np.pad(img, [(pad_y, pad_y), (pad_x, pad_x), (0, 0)], mode="constant", constant_values=0)
        out = []
        for i in range(img.shape[2]):
            out += [fftconvolve(img[:, :, i], kernel, mode="valid")]
        out = np.stack(out, axis=2)

    out = out.astype(orig_dtype)
    return out


def generate_foreshorten_distortion(
        img: np.ndarray,
        *, 
        pos_xyz: Tuple[float, float, float]=(0, 0, 500),
        fx: float=512,
        fy: float=512,
        cx: float=0.0,
        cy: float=0.0,
        alpha: float=0,
        beta: float=0,
        gamma: float=0,
        grid_xy: Optional[np.ndarray]=None,
    ) -> np.ndarray:
    
    h, w, _ = img.shape
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)

    rot_X = np.asarray([1, 0, 0, 0, cos_alpha, -sin_alpha, 0, sin_alpha, cos_alpha]).reshape(3, 3)
    rot_Y = np.asarray([cos_beta, 0, sin_beta, 0, 1, 0, -sin_beta, 0, cos_beta]).reshape(3, 3)
    rot_Z = np.asarray([cos_gamma, -sin_gamma, 0, sin_gamma, cos_gamma, 0, 0, 0, 1]).reshape(3, 3)

    # K: Intrinsic matrix for mapping 3D camera space to 2D image space
    # R: Rotation matrix
    # T: Translation matrix (or vector)
    # M: Extrinsic matrix (combines rotation and translation) for mapping real world space to camera space
    K = np.asarray([[fx, 0, cx, 0, fy, cy, 0,  0,  1]]).reshape(3, 3)
    R = rot_Z @ rot_Y @ rot_X
    T = np.asarray(pos_xyz, dtype=R.dtype)
    M = np.zeros((3, 4), dtype=R.dtype)
    M[:3, :3] = R
    M[:, 3] = T

    if grid_xy is None:
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        grid_xy = np.stack([x, y], axis=2)[:, :, None, :]
    else:
        assert grid_xy.shape[:2] == (h, w)
        grid_xy = grid_xy[:, :, None, :]

    # include z and homogeneous coordinates (z here is different from z in translation T,
    # z in translation represents camera position relative to the world along the z axis)
    grid_zh = np.zeros(grid_xy.shape, dtype=grid_xy.dtype)
    grid_zh[..., 1] = 1
    grid_xyzh = np.concatenate([grid_xy, grid_zh], axis=3)
    grid_xyzh = np.matmul(grid_xyzh, M.T[None, None, :, :])
    grid_xyzh = np.matmul(grid_xyzh, K.T[None, None, :, :])
    grid_xy = (grid_xyzh / grid_xyzh[..., None, 2])[:, :, 0, :2]

    return grid_xy


def generate_sinusoidal_distortion(
        img: np.ndarray, 
        *,
        along_x: bool=True, 
        along_y: bool=True,
        x_amp: float=5, 
        y_amp: float=5,
        x_lambda: float=2000,
        y_lambda: float=2000,
        grid_xy: Optional[np.ndarray]=None
    ) -> np.ndarray:
    
    h, w, _ = img.shape
    if grid_xy is None:
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        grid_xy = np.stack([x, y], axis=2)
    else:
        assert grid_xy.shape[:2] == (h, w)

    if along_x:
        grid_xy[:, :, 0] = grid_xy[:, :, 0] + x_amp * np.sin(2 * np.pi * grid_xy[:, :, 1] / x_lambda)

    if along_y:
        grid_xy[:, :, 1] = grid_xy[:, :, 1] + y_amp * np.sin(2 * np.pi * grid_xy[:, :, 0] / y_lambda)

    return grid_xy


def generate_polynomial_distortion(
        img: np.ndarray, 
        *,
        kx: float=1e-7, 
        ky: float=1e-7,
        grid_xy: Optional[np.ndarray]=None
    ) -> np.ndarray:
    """
    Produces barrel (outward curve) or pin-cushion (inward curve) like distortion. This is the
    kind of distortion gottem from the polynomial expansion used in camera lens distortion models 
    """
    h, w = img.shape[:2]
    if grid_xy is None:
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid_xy = np.stack([xv, yv], axis=2)
    else:
        assert grid_xy.shape[:2] == (h, w)

    # compute distance from center and radius, then apply polynomial distortion
    dx = grid_xy[..., 0] - w / 2
    dy = grid_xy[..., 1] - h / 2
    r_sq = dx**2 + dy**2
    grid_xy[..., 0] = grid_xy[..., 0] + kx * r_sq * dx
    grid_xy[..., 1] = grid_xy[..., 1] + ky * r_sq * dy
    return grid_xy


def generate_smooth_random_distortion(
        img: np.ndarray, 
        *,
        kernel_size: Tuple[int, int]=(0, 0),
        x_amp: float=15,
        y_amp: float=15, 
        sigma: float=150,
        grid_xy: Optional[np.ndarray]=None
    ) -> np.ndarray:
    """
    curve / distorted grid randomly at random local regions with a gaussian function
    """

    h, w = img.shape[:2]
    if grid_xy is None:
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid_xy = np.stack([xv, yv], axis=2)
    else:
        assert grid_xy.shape[:2] == (h, w)

    # create random 2D displacement fields to make grid paper look like it was squeezed / wrinkled
    # and smooth the displacelemt field with a gaussian kernel
    dx_dy = np.random.randn(h, w, 2)
    dx_dy = fast_gaussian_blur(dx_dy, kernel_size=kernel_size, sigmas=(sigma, sigma))

    grid_xy[..., 0] = grid_xy[..., 0] + x_amp * dx_dy[..., 0] / np.max(np.abs(dx_dy[..., 0]))
    grid_xy[..., 1] = grid_xy[..., 1] + y_amp * dx_dy[..., 1] / np.max(np.abs(dx_dy[..., 1]))
    return grid_xy


def apply_distortions(
        img: np.ndarray,
        size: Tuple[int, int],
        types_list: List[Literal["foreshorten", "sinusoidal", "polynomial", "smooth_random"]],
        kwargs_list: Optional[List[Dict[str, Any]]]=None,
        padding_mode="zeros",
        device: str="cpu",
        **kwargs
    ) -> np.ndarray:

    n = len(types_list)
    if kwargs_list is not None:
        assert n == len(kwargs_list)
    else:
        kwargs_list = [None for _ in range(0, n)]
    
    h, w = img.shape[:2]
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    grid_xy = np.stack([xv, yv], axis=2)

    distortions_map = {
        "foreshorten": generate_foreshorten_distortion,
        "sinusoidal": generate_sinusoidal_distortion,
        "polynomial": generate_polynomial_distortion,
        "smooth_random": generate_smooth_random_distortion
    }
    for d_type, d_kwargs in zip(types_list, kwargs_list):
        d_kwargs = {} or d_kwargs
        p = d_kwargs.pop("p", 1)
        if p > np.random.uniform():
            grid_xy = distortions_map[d_type](img, grid_xy=grid_xy, **d_kwargs)

    return grid_sample(img, grid_xy, size=size, padding_mode=padding_mode, device=device, **kwargs)


def get_waveform_plot_rows(
        df: pd.DataFrame, 
        r_cols: List[str],
    ) -> pd.Series:

    row_series = []
    N = df.shape[0]
    for col in r_cols:
        mask = ~pd.isna(df[col])
        if col == "II" and len(r_cols)>1:
            mask[N-(sum((~pd.isna(df[c])).sum() for c in r_cols[1:])):] = False
        row_series.append(df[col][mask])
    row_series = pd.concat(row_series)
    return row_series


def project_to_grid(
        img: np.ndarray, 
        xs: np.ndarray,
        ys: np.ndarray, 
        pulser_n: int,
        n_leads: int,
        ignore_pulser: bool=False,
        color: Union[int, Tuple[int, int, int]]=(0, 0, 0),
        thickness: int=1
    ) -> np.ndarray:
    assert pulser_n >= 0
    pts = np.stack([xs, ys], axis=1)

    kwargs = dict(isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    if pulser_n > 0 and (not ignore_pulser):
        img = cv2.polylines(img, pts=[pts[:pulser_n]], **kwargs)

    pts = pts[pulser_n:]
    seg_n = pts.shape[0] // n_leads
    for i in range(n_leads):
        start = i * seg_n
        stop = (start + seg_n) if i < n_leads-1 else None
        img = cv2.polylines(img, pts=[pts[start:stop]], **kwargs)
    return img


def generate_random_timestamp() -> str:
    start = datetime(1990, 1, 1, 0, 0, 0)
    end = datetime.now()
    time_between_dates = end - start
    total_seconds = int(time_between_dates.total_seconds())
    random_seconds = random.randrange(total_seconds)
    random_datetime = start + timedelta(seconds=random_seconds)
    return random_datetime.strftime("%Y-%m-%d, %H:%M:%S")


def draw_signal(
        img: np.ndarray,
        signal_df: pd.DataFrame,
        sig_y_mm_offset: float=90,
        sig_x_mm_offset: float=10,
        sig_row_mm_space: float=36,
        sig_color: Tuple[int, int, int]=(0, 0, 0),
        tick_mm_height: float=7,
        text_y_mm_pos: float=7,
        sig_thickness: int=1,
        sig_mask_thickness: int=1,
        tick_thickness: int=5,
        text_thickness: int=2,
        font_scale: float=1.1,
        ignore_mask_sq_pulser: bool=False,
        font_face: int=cv2.FONT_HERSHEY_SIMPLEX,
        with_annotations: bool=False,
        num_points_scale: int=10,
    ) -> np.ndarray:
    
    assert num_points_scale > 0

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    amp2mm = lambda mv: mv * MM_PER_MV
    sec2mm = lambda sec: sec * MM_PER_SEC
    mm2px = lambda mm: mm * THIN_GRID_SPACE_PX

    sig_y_px_offset = int(mm2px(sig_y_mm_offset))
    sig_x_px_offset = int(mm2px(sig_x_mm_offset))
    sig_row_px_space = int(mm2px(sig_row_mm_space))
    tick_px_height = int(mm2px(tick_mm_height))
    text_y_px_pos = int(mm2px(text_y_mm_pos))

    annotations = None
    if with_annotations:
        h, w, _ = img.shape
        # the first four channels correspond to the four rows containing the 12 lead
        # signals and the fifth channel corresponds to text / labels of leads
        # the third channel corresponds to the short vertical lead markers
        annotations = np.zeros((6, h, w), dtype=img.dtype)

    for i, r_cols in enumerate([R1_COLS, R2_COLS, R3_COLS, R4_COLS]):
        plot_rows = get_waveform_plot_rows(signal_df, r_cols)
        xs, ys = plot_rows.index.values, plot_rows.values

        if num_points_scale != 1:
            xs *= num_points_scale
            num_points = int(xs.shape[0] * num_points_scale)
            # interpolate points for smoother curve on the ECG printout
            linspace = np.linspace(0, xs.shape[0] - 1, num=num_points)
            point_range = np.arange(0, xs.shape[0])
            xs = np.interp(linspace, point_range, xs)
            ys = np.interp(linspace, point_range, ys)

        n_samples = xs.shape[0]
        n_leads = len(r_cols)
        n_samples_per_lead = n_samples // n_leads
        sample_freq = n_samples / SIGNAL_DURATION

        xs = xs / (sample_freq - 1)
        xs, ys = sec2mm(xs), amp2mm(ys)

        # add initial square wave signal (square wave spans 5mm which is a duration is 
        # (THICK_GRID_SPACE_MM / mm_per_sec) seconds) sqwave has a height of 10 mm 
        # (akin to 10 / mm_per_mv) mv size of the sqwave. NOTE: This square wave
        # serves as the calibration pulse for vertical baseline
        sqwave_size = int((THICK_GRID_SPACE_MM / MM_PER_SEC) * sample_freq)
        sqwave_xs = np.linspace(0, THICK_GRID_SPACE_MM, num=sqwave_size)
        xs = THICK_GRID_SPACE_MM + xs
        sqwave_ys = np.zeros(sqwave_xs.shape)
        sqwave_ys[[0, -1]] = 0
        sqwave_ys[1:-1] = THICK_GRID_SPACE_MM * 2

        xs = np.concatenate([sqwave_xs, xs], axis=0)
        ys = np.concatenate([sqwave_ys, ys], axis=0)

        xs, ys = mm2px(xs).astype(np.int64), mm2px(ys).astype(np.int64)
        xs = sig_x_px_offset + xs
        vbaseline = sig_y_px_offset + i * sig_row_px_space - ((i == 3) * mm2px(3))
        ys = vbaseline - ys
        img = project_to_grid(img, xs, ys, pulser_n=sqwave_size, n_leads=n_leads, color=sig_color, thickness=sig_thickness)

        if len(r_cols) > 1:
            # since n_samples can be indivisible by 4 in some cases, xs[sqwave_size::n_samples_per_lead]
            # can have shape greater than 4, so we in such cases, it helps that we explicitly
            # get the first 4 elements.
            tick_xs = xs[sqwave_size::n_samples_per_lead][:4]
        else:
            # we only need the first element for the last row which corresponds to lead 11
            tick_xs = xs[sqwave_size::n_samples_per_lead][:1]
            
        for j, tx in enumerate(tick_xs):
            x1 = x2 = tx
            y1 = vbaseline - tick_px_height//2
            y2 = vbaseline + tick_px_height//2

            # draw thick short vertical lines between signal leads on image and on annotation mask (if need be)
            if i < 3 and j > 0:
                img = cv2.line(
                    img, pt1=(x1, y1), pt2=(x2, y2), color=sig_color, thickness=tick_thickness
                )
                if annotations is not None:
                    annotations[5] = cv2.line(
                        annotations[5], pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=tick_thickness
                    )
            
            text_kwargs = dict(
                text=r_cols[j],
                org=(x2, vbaseline + text_y_px_pos), 
                fontFace=font_face, 
                fontScale=font_scale, 
                thickness=text_thickness
            )

            # write lead text labels on image and annotations (if available)
            img = cv2.putText(img, color=sig_color, **text_kwargs)
            if annotations is not None:
                annotations[4] = cv2.putText(annotations[4], color=255, **text_kwargs)
            
        # draw lead signals on annotation masks
        if annotations is not None:
            annotations[i] = project_to_grid(
                annotations[i], 
                xs,
                ys,
                pulser_n=sqwave_size, 
                ignore_pulser=ignore_mask_sq_pulser, 
                n_leads=n_leads, 
                color=255, 
                thickness=sig_mask_thickness
            )

    sec_scale_label_px_pos = int(mm2px(50))
    mv_scale_label_px_pos = int(mm2px(100))
    scale_label_y_px_pos = int(mm2px(209.8))
    id_x_px_pos = int(mm2px(2.5))
    name_x_px_pos = int(mm2px(80))
    date_x_px_pos = int(mm2px(155))

    _id = str(np.random.randint(1, 10_000)).zfill(5)
    age = np.random.randint(10, 100)
    sex = np.random.choice(['Male', 'Female'])
    date = generate_random_timestamp()
    weight = np.random.randint(40, 120)

    for mat, c in zip([img, annotations[4]], [sig_color, 255]):
        text_kwargs = dict(
            fontFace=font_face, 
            fontScale=font_scale, 
            color=c,
            thickness=text_thickness
        )
        cv2.putText(mat, f"ID: {_id}_hr", org=(id_x_px_pos, int(mm2px(5))), **text_kwargs)
        cv2.putText(mat,f"Age: {age} yrs", org=(id_x_px_pos, int(mm2px(10))), **text_kwargs)
        cv2.putText(mat,f"Sex: {sex}", org=(id_x_px_pos, int(mm2px(15))), **text_kwargs)
        cv2.putText(mat, f"Name: ", org=(name_x_px_pos, int(mm2px(5))), **text_kwargs)
        cv2.putText(mat, f"Date: {date}", org=(date_x_px_pos, int(mm2px(5))), **text_kwargs)
        cv2.putText(mat,f"Weight: {weight} Kg", org=(date_x_px_pos, int(mm2px(10))), **text_kwargs)
        cv2.putText(mat, "25mm/s", org=(sec_scale_label_px_pos, scale_label_y_px_pos), **text_kwargs)
        cv2.putText(mat, "10mm/mv", org=(mv_scale_label_px_pos, scale_label_y_px_pos), **text_kwargs)

    if annotations is not None:
        annotations = np.transpose(annotations, (1, 2, 0))
        img = np.concatenate([img, annotations], axis=2)
    return img


def draw_h_or_v_lines(
        img: np.ndarray,
        line_indexes: np.ndarray,
        line_type: str,
        thickness: int=1,
        color: Union[int, Tuple[int, int, int]]=(0, 0, 0)
    ) -> np.ndarray:
    
    assert line_type in ["h", "v"]
    h, w = img.shape[:2]
    for line in line_indexes:
        if line_type == "h":
            x1, x2 = 0, w
            y1 = y2 = int(line)
        else:
            x1 = x2 = int(line)
            y1, y2 = 0, h
        img = cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def draw_grid(
        *,
        thickness: int=1,
        seg_thickness: int=2,
        point_radius: int=4,
        dtype="uint8",
        thick_color: Tuple[int, int, int]=(0, 0, 0),
        thin_color: Tuple[int, int, int]=(100, 100, 100),
        bg_color: Tuple[int, int, int]=(255, 255, 255),
        with_annotations: bool=False,
        qr_code: bool=True,
        qr_code_data: Optional[str]=None,
        qr_code_width: int=190,
    ) -> np.ndarray:
    
    # vdelta = vertical spacing between horizontal lines
    # hdelta = horizontal spacing between vertical lines
    thick_hlines = np.arange(THICK_GRID_SPACE_PX, GRID_H, THICK_GRID_SPACE_PX)
    thick_vlines = np.arange(THICK_GRID_SPACE_PX, GRID_W, THICK_GRID_SPACE_PX)

    thin_hlines = np.arange(THIN_GRID_SPACE_PX, GRID_H, THIN_GRID_SPACE_PX)
    thin_vlines = np.arange(THIN_GRID_SPACE_PX, GRID_W, THIN_GRID_SPACE_PX)
    
    grid = np.full((GRID_H, GRID_W, 3), fill_value=bg_color, dtype=getattr(np, dtype))

    grid = draw_h_or_v_lines(grid, thin_hlines, line_type="h", thickness=thickness, color=thin_color)
    grid = draw_h_or_v_lines(grid, thin_vlines, line_type="v", thickness=thickness, color=thin_color)
    
    grid = draw_h_or_v_lines(grid, thick_hlines, line_type="h", thickness=thickness, color=thick_color)
    grid = draw_h_or_v_lines(grid, thick_vlines, line_type="v", thickness=thickness, color=thick_color)

    masks = None
    ann_kwargs = None
    # if with_annotations is set, the first channel corresponds to only
    # vertical lines, the second channel to horizontal lines and the third channel
    # to point of intersection between horizontal and vertical lines
    if with_annotations:
        hlines_segments = np.zeros((GRID_H, GRID_W), dtype=getattr(np, dtype))
        vlines_segments = np.zeros((GRID_H, GRID_W), dtype=getattr(np, dtype))
        center_segments = np.zeros((GRID_H, GRID_W), dtype=getattr(np, dtype))

        # farthest edges of the grid should also be segmented
        thick_hlines = np.concatenate([[0], thick_hlines, [GRID_H-1]], axis=0)
        thick_vlines = np.concatenate([[0], thick_vlines, [GRID_W-1]], axis=0)

        ann_kwargs = dict(thickness=seg_thickness, color=255)

        # draw horizontal and vertical line segments
        hlines_segments = draw_h_or_v_lines(hlines_segments, thick_hlines, line_type="h", **ann_kwargs)
        vlines_segments = draw_h_or_v_lines(vlines_segments, thick_vlines, line_type="v", **ann_kwargs)

        # draw grid points
        y, x = np.meshgrid(thick_hlines, thick_vlines)
        intersect_xy = np.stack([x.ravel(), y.ravel()], axis=-1)
        for x, y in intersect_xy:
            center_segments = cv2.circle(center_segments, (x, y), radius=point_radius, color=255, thickness=-1)
        
        masks = np.concatenate([
            hlines_segments[..., None], 
            vlines_segments[..., None], 
            center_segments[..., None],
        ], axis=2)

    if qr_code:
        QRcode = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        QRcode.add_data(qr_code_data or "random text for QR code")
        QRcode.make(fit=True)
        qr_img = QRcode.make_image(fill_color="black", back_color="white").convert("RGB")
        wpercent = qr_code_width / float(qr_img.size[0])
        hsize = int((float(qr_img.size[1]) * float(wpercent)))
        qr_img = qr_img.resize((qr_code_width, hsize))
        qr_img = np.asarray(qr_img)
        qr_h, qr_w, _ = qr_img.shape

        w = grid.shape[1]
        xy_pos = ((w - qr_code_width) + qr_code_width//2, qr_code_width//2)
        a = xy_pos[1] - qr_h//2
        b = xy_pos[1] + qr_h//2
        c = xy_pos[0] - qr_w//2
        d = xy_pos[0] + qr_w//2
        grid[a:b, c:d, :] = qr_img[:b-a, :d-c, :]
        # set non-zero pixels in this region of the segmentation mask to 0 for consistency.
        if masks is not None:
            masks[a:b, c:d, :-1] = 0

    grid = np.concatenate([grid, masks], axis=2)
    return grid


def apply_crease(
    img: np.ndarray, 
    num_lines: int, 
    theta: float=np.pi/2,
    kernel_size: Tuple[int, int]=(3, 3), 
    sigmas: Tuple[float, float]=(1.0, 1.0), 
    line_thickness: int=5,
    alpha: float=1.0,
    ignore_mask: Optional[np.ndarray]=None
) -> np.ndarray:
    
    h, w, _ = img.shape

    spacing = (h + w) / num_lines

    x1 = np.arange(0, w, step=spacing)
    y1 = np.arange(0, h, step=spacing)

    x1 = np.concatenate([np.zeros((num_lines - x1.shape[0]), dtype=x1.dtype), x1], axis=0)
    y1 = np.concatenate([y1, np.zeros((num_lines - y1.shape[0]))], axis=0)    
   
    crease_map = np.ones((h, w), dtype=np.float32)
    d_offsets, cvals = [0, -5, +5, 10, -10], [1.25, 1.15, 1.15, 1.05, 1.05]

    for theta_ in[theta, np.pi/2-theta]:
        x2 = x1 + w * np.cos(theta_)
        y2 = y1 + h * np.sin(theta_)
        for i in range(num_lines):
            u1, v1 = int(x1[i]), int(y1[i])
            u2, v2 = int(x2[i]), int(y2[i])
            if(u1-10 < 0):
                for vd, a in zip(d_offsets, cvals):
                    crease_map = cv2.line(crease_map, (u1,v1+vd), (u2,v2+vd), a, line_thickness)
            else:
                for ud, a in zip(d_offsets, cvals):
                    crease_map = cv2.line(crease_map, (u1+ud,v1), (u2+ud,v2), a, line_thickness)

    img = img / 255
    crease_map = cv2.GaussianBlur(crease_map, kernel_size, sigmaX=sigmas[0], sigmaY=sigmas[1])
    
    if ignore_mask is not None:
        crease_map[ignore_mask] = 1
    img = np.clip((img * alpha * crease_map[:, :, None]) * 255, a_min=0, a_max=255).astype(np.uint8)

    return img


def compute_quilt_ssd(texture: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Formulation:
        Convolving one function (B = kernel) over another (A = signal / texture) in a sliding window
        manner and computing the SSD (Sum of Square Differences) between the kernel and each overlapping
        patches of the signal by sliding the kernel through the texture can be formulated as:
        SSD(x,y) = Σ (A_{x,y} - B)^2 = Σ A_{x,y}^2 - 2 Σ A_{x,y} B + Σ B^2
        
        The first term Σ A_{x,y}^2 is a cross correlation of A with a kernel filled with ones to produce
        the sum of squares of itself within a given patch

        The second term Σ A_{x,y} B is a cross correlation of A with a sliding fixed kernel B

        The third term Σ B^2 in this context is a constant, it is just the sum of square values of the kernel
        B, and B is constant for each overlapping patch in the convolution

        NOTE: The problem is a cross-correlation problem (not a convolution problem), but we can formulate it 
            as a convolution problem by spatially inverting the kernels B along both axes (x and y), by doing
            so we can exploit the compute efficency of the FFT (Fast Fourier Transform) for computing convolutions,
            this is due to the fact that convolution in spatial or temporal domain maps to multiplication in spectral
            domain (via Fourier or Laplace Transform), this is called the Convolution Theorem.
    """
    AB = fftconvolve(texture, kernel[::-1, ::-1], mode="valid")
    B_SQ = np.sum(kernel**2)
    ones = np.ones(kernel.shape[:2], dtype=texture.dtype)
    A_SQ = fftconvolve(texture**2, ones, mode="valid")

    # tiny negative values ocassionally occur due to numerical precision issues
    return np.clip(A_SQ - 2 * AB + B_SQ, a_min=0.0, a_max=np.inf)


def find_best_quilt_patch(
        texture: np.ndarray, 
        overlap_size: int,
        Lblock: Optional[np.ndarray]=None, 
        Tblock: Optional[np.ndarray]=None, 
        tolerance: float=0.1) -> np.ndarray:
    
    assert Lblock is not None or Tblock is not None

    H, W = texture.shape[:2]
    block_size = [Lblock, Tblock][Lblock is None].shape[0]
    error_map = np.zeros((H-block_size+1, W-block_size+1), dtype=texture.dtype)
    
    rtexture = texture.sum(axis=2)

    if Lblock is not None:
        Lblock = Lblock.sum(axis=2)
        Lblock = Lblock[:, block_size-overlap_size:]
        error_map += compute_quilt_ssd(rtexture, Lblock)[:, :W-block_size+1]

    if Tblock is not None:
        Tblock = Tblock.sum(axis=2)
        Tblock = Tblock[block_size-overlap_size:, :]
        error_map += compute_quilt_ssd(rtexture, Tblock)[:H-block_size+1, :]
        
    min_error = error_map.min()
    ys, xs = np.where(error_map <= (1 + tolerance) * min_error)
    idx = np.random.randint(0, ys.shape[0])
    y, x = ys[idx], xs[idx]
    return texture[y:y+block_size, x:x+block_size]


def get_quilt_seam_path(overlap_residue: np.ndarray) -> Tuple[List[int], np.ndarray]:
    opened_set = [(err, [i])for i, err in enumerate(overlap_residue[0])]
    closed_set = set()

    res_h, res_w = overlap_residue.shape[:2]
    block_size = max(res_h, res_w)
    heapq.heapify(opened_set)

    while opened_set:
        err, path = heapq.heappop(opened_set)
        curr_depth = len(path)
        curr_idx = path[-1]

        if (curr_depth, curr_idx) in closed_set:
            continue

        if curr_depth == res_h:
            break

        closed_set.add((curr_depth, curr_idx))

        for child_idx in range(curr_idx-1, curr_idx+2):
            if 0 <= child_idx < res_w:
                new_err = err + overlap_residue[curr_depth, child_idx]
                new_path = path + [child_idx]
                heapq.heappush(opened_set, (new_err, new_path))

    # using the generated paths of minimum cost, create a mask equal to block size, where all pixels
    # leftward of the path pixels and at this path index are 1s and all pixels rightward of the path
    # are 0s. The new patch will then be computed as follows:
    # patch = previous_patch * mask + selected_patch * (1 - mask)
    mask = np.zeros((block_size, block_size))
    path = np.asarray(path)
    mask[np.arange(mask.shape[1])[None, :] <= path[:, None]] = 1
    return path, mask


def quilt(
        texture: np.ndarray, 
        num_xpatches: int=1,
        num_ypatches: int=1,
        block_size: int=128,
        overlap_scale: int=6,
        tolerance: float=0.1
    ) -> np.ndarray:
    
    """LINK: https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf"""

    assert overlap_scale < block_size

    th, tw = texture.shape[:2]
    x0 = None
    y0 = None
    # while to half block size if block size is greater than or equal to texture image width or height
    while x0 is None or y0 is None:
        try:
            x0 = np.random.randint(0, tw-block_size)
            y0 = np.random.randint(0, th-block_size)
        except ValueError:
            block_size = block_size // 2

    overlap_size = block_size // overlap_scale
    out_h = (num_xpatches * block_size) - (num_xpatches - 1) * overlap_size
    out_w = (num_ypatches * block_size) - (num_ypatches - 1) * overlap_size

    output = np.zeros((out_h, out_w, texture.shape[2]), dtype=texture.dtype)
    output[:block_size, :block_size, :] = texture[y0:y0+block_size, x0:x0+block_size, :]

    hop_size = block_size - overlap_size

    for i in range(0, out_h-hop_size, hop_size):
        for j in range(0, out_w-hop_size, hop_size):
            a = i - block_size + overlap_size
            b = a + block_size
            c = j - block_size + overlap_size
            d = c + block_size

            if a < 0:
                a = 0
                b = block_size

            if c < 0:
                c = 0
                d = block_size

            x = i // hop_size
            y = j // hop_size
            if x + y == 0: 
                continue

            if x == 0:
                Tblock = None
            else:
                Tblock = output[a:b, c:d]

            if y == 0:
                Lblock = None
            else:
                Lblock = output[a:b, c:d]

            Hmask = None
            Vmask = None

            curr_patch = output[i:i+block_size, j:j+block_size]

            # find best quilt patch that minimizes overlap error of previous left and previous top blocks
            best_patch = find_best_quilt_patch(texture, overlap_size, Lblock, Tblock, tolerance=tolerance)

            # match overlapped area of best_patch and curr_patch (patch to edit) and get the best seam path
            if Tblock is not None:
                residue = np.pow(Tblock[block_size-overlap_size:, :] - best_patch[:overlap_size, :], 2)
                if residue.ndim == 3:
                    residue = residue.sum(axis=2).T
                _, Hmask = get_quilt_seam_path(residue)
                Hmask = Hmask.T[:, :, None]
                best_patch = curr_patch * Hmask + best_patch * (1 - Hmask)
            
            # match overlapped area of best_patch and curr_patch and get the best seam path
            if Lblock is not None:
                residue = np.pow(Lblock[:, block_size-overlap_size:] - best_patch[:, :overlap_size], 2)
                if residue.ndim == 3:
                    residue = residue.sum(axis=2)
                _, Vmask = get_quilt_seam_path(residue)
                Vmask = Vmask[:, :, None]
                best_patch = curr_patch * Vmask + best_patch * (1 - Vmask)
            
            curr_patch = output[i:i+block_size, j:j+block_size]            
            curr_patch[:] = best_patch[:curr_patch.shape[0], :curr_patch.shape[1]]
    return output


def apply_wrinkle_texture(
        img: np.ndarray,
        block_size: int=128,
        overlap_scale: int=6,
        tolerance: float=0.1,
        num_xpatches: int=1,
        num_ypatches: int=1,
        mean_shift: float=0.4,
        map_threshold: float=0.6,
        textures_dir: Optional[str]=None,
        texture_file: Optional[str]=None,
        texture: Optional[np.ndarray]=None,
        ignore_mask: Optional[np.ndarray]=None
    ) -> np.ndarray:
    """
    Although this function was originally designed to apply wrinkle textures via quilting, it can apply
    other textures as well like realistic stains (like drink on paper) and other textures as well, also 
    via quilting.
    """
    if texture is None:
        if texture_file:
            texture = cv2.imread(texture_file)
        else:
            assert textures_dir
            texture_path = os.path.join(textures_dir, random.choice(os.listdir(textures_dir)))
            texture = cv2.imread(texture_path)
        texture = texture[:, :, ::-1]
    
    img = (img / 255).astype(np.float32)
    texture = (texture / 255).astype(np.float32)
    wrinkle_map = quilt(texture, num_xpatches, num_ypatches, block_size=block_size, overlap_scale=overlap_scale, tolerance=tolerance)

    h, w = img.shape[:2]
    wrinkle_map = cv2.resize(wrinkle_map, (w, h))

    mean = wrinkle_map.mean() - mean_shift
    wrinkle_map -= mean

    _, thresh = cv2.threshold(wrinkle_map, map_threshold, 1.0, cv2.THRESH_BINARY)
    low = 2 * img * wrinkle_map
    high = 1 - 2 * (1 - img) * (1 - wrinkle_map)
    
    if ignore_mask is not None:
        ignore_mask = np.broadcast_to(ignore_mask[:, :, None], img.shape)
        low[ignore_mask, ...] = 0
        high[ignore_mask, ...] = 0
        orig_img = img.copy()
        orig_img[~ignore_mask] = 0
        img = orig_img + low * thresh + high * (1 - thresh)
    else:
        img = low * thresh + high * (1 - thresh)

    img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)
    return img


def apply_contrast(
        img: np.ndarray, 
        contrast: float=1.0, 
        brightness: float=0.0, 
    ) -> np.ndarray:
    # if statement to save compute
    assert contrast >= 0 and -1 <= brightness <= 1
    if contrast == 1 and brightness == 0: return img
    img = (img / 255).astype(np.float32)
    img = contrast * img + brightness
    return np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)


def apply_color_gradient(
        img: np.ndarray, 
        c1: Optional[Tuple[int, int, int]]=None, 
        c2: Optional[Tuple[int, int, int]]=None,
        alpha: float=0.1,
        along: str="vertical",
        grad_type: bool="linear"
    ) -> np.ndarray:

    assert along in ["horizontal", "vertical"]
    assert grad_type in ["linear", "sinusoidal"]

    is_c1 = c1 is not None
    is_c2 = c2 is not None

    assert is_c1 or is_c2

    h, w, _ = img.shape

    linear_grad = lambda c, idxs, max_idx: (c + 1)[None, :] - (c + 1)[None, :] * idxs[:, None] / max_idx
    cosine_grad = lambda c, idxs, max_idx: 1 / 2 * (c + 1)[None, :] * (1 + np.cos(np.pi * idxs[:, None] / max_idx))

    grad_func = linear_grad if grad_type == "linear" else cosine_grad

    if (is_c1 and not is_c2) or (is_c2 and not is_c1):
        c = np.asarray([c1, c2][is_c2])

        if along == "vertical":
            cmaps = grad_func(c, np.arange(h), h-1)
            cmaps = cmaps[:, None, :]
            if is_c2:
                cmaps = cmaps[::-1, :, :]

        elif along == "horizontal":
            cmaps = grad_func(c, np.arange(w), w-1)
            cmaps = cmaps[None, :, :]
            if is_c2:
                cmaps = cmaps[:, ::-1, :]

    if is_c1 and is_c2:
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)

        if along == "vertical":
            idxs = np.arange(h//2)
            cmaps1 = grad_func(c1, idxs, h//2-1)
            cmaps2 = grad_func(c2, idxs, h//2-1)
            cmaps = np.concatenate(
                [cmaps1, cmaps2[::-1, :]] if h//2 * 2 == h else [cmaps1, cmaps1[-1:, :], cmaps2[::-1, :]],
            axis=0)
            cmaps = cmaps[:, None, :]

        elif along == "horizontal":
            idxs = np.arange(w//2)
            cmaps1 = grad_func(c1, idxs, w//2-1)
            cmaps2 = grad_func(c2, idxs, w//2-1)
            cmaps = np.concatenate(
                [cmaps1, cmaps2[::-1, :]] if w//2 * 2 == w else [cmaps1, cmaps1[-1:, :], cmaps2[::-1, :]],
            axis=0)
            cmaps = cmaps[None, :, :]

    return np.clip((img * (1 - alpha)) + (cmaps * alpha), a_min=0, a_max=255).astype(np.uint8)


def apply_stains(
        img: np.ndarray, 
        num_circles: int=50,
        region: Tuple[Tuple[float, float], Tuple[float, float]]=((0.3, 0.4), (0.7, 0.7)),
        color: Tuple[int, int, int]=(0, 0, 0),
        kernel_size: Tuple[int, int]=(41, 41),
        sigmas: Tuple[float, float]=(15.0, 15.0),
        num_grains: int=5000,
        alpha: float=0.1,
        stain_type: str="multiplicative",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply additive, subtractive or multiplicative stain. Stains are generated by randomly drawing dark filled
    circles of varying radii within a specfic region and using the Gaussian filter blur the stained area, smearing 
    the circles across the image to create a seeminly realistic stain.
    If stain_type is multiplicative, the noise map is generated, scaled by 255 and multiplied with the original
    image. In such cases, while areas of interest might appear darker or a particular color different from the 
    rest of the image, it will not be completely occluded and can still be recovered, so the target mask for areas
    of interest can be left as is.
    If stain_type is additive or subtractive, then the generated noise map after the Gaussian filter is subtracted
    from 255 and added or subtracted to the original image. In this case the stains can cause full / heavy occlusion
    of a specific area of interest, if so the target mask should be adjusted accordily to prevent the model from
    from trying to predict segments it cannot see.
    """
    assert region and region[0][0] < region[1][0] and region[0][1] < region[1][1]
    assert all([0 <= u <= 1 for r in region for u in r])
    assert stain_type in ["additive", "subtractive", "multiplicative"]

    h, w, _ = img.shape
    radii = np.random.uniform(low=0.0002, high=0.01, size=(num_circles, ))
    x_centers = np.random.uniform(low=region[0][0], high=region[1][0], size=(num_circles, ))
    y_centers = np.random.uniform(low=region[0][1], high=region[1][1], size=(num_circles, ))

    stain = np.full_like(img, fill_value=255, dtype=np.float32)
    high = stain.copy()
    for i in range(radii.shape[0]):
        x, y = int(x_centers[i] * w), int(y_centers[i] * h)
        radius = max(1, int(radii[i] * max(h, w)))
        stain = cv2.circle(stain, center=(x, y), radius=radius, color=color, thickness=-1)
    
    x, y = int(w * region[0][0]),  int(h * region[0][1])
    u, v = int(w * region[1][0]),  int(h * region[1][1])
    stain[
        np.random.randint(y, v, size=(num_grains, )), 
        np.random.randint(x, u, size=(num_grains, ))
    ] = color

    stain = cv2.GaussianBlur(stain, ksize=kernel_size, sigmaX=sigmas[0], sigmaY=sigmas[1])
    stain = (stain * (1 - alpha)) + (high * alpha)

    if stain_type == "multiplicative":
        stain = stain / 255
        img = img * stain
    else:
        stain = 255 - stain
        if stain_type == "additive":
            img = img + stain
        else:
            img = img - stain

    img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
    
    return img


def to_grayscale(img: np.ndarray, with_three_channels: bool=True) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if with_three_channels:
        img = np.tile(img[:, :, None], (1, 1, 3))
    return img


def apply_blur(img: np.ndarray, kernel_size: Tuple[int, int], sigmas: Tuple[float, float]) -> np.ndarray:
    """
    kernel_size coresponds to kernel height and kernel width
    sigmas corresponds to sigma along the vertical axis and along the horizontal axis
    """
    img = cv2.GaussianBlur(img, ksize=kernel_size[::-1], sigmaY=sigmas[0], sigmaX=sigmas[1])
    return img


def get_sobel_gradient(
        img: np.ndarray, 
        xkernel_size: int=3, 
        ykernel_size: int=3,
        dx: int=1,
        dy: int=1,
        compute_direction: bool=False,
        atan2_direction: bool=False,
        eps: float=1e-6
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    """compute the edge gradient (contour) map with the sobel gradient filter"""
    x_grad = cv2.Sobel(img, cv2.CV_32F, dx=dx, dy=0, ksize=xkernel_size)
    y_grad = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=dy, ksize=ykernel_size)
    grad_magnitude = np.sqrt(x_grad**2 + y_grad**2)
    
    if not compute_direction:
        return grad_magnitude
    
    if atan2_direction:
        grad_direction = np.arctan2(y_grad, (x_grad + eps))
    else:
        grad_direction = np.arctan(y_grad / (x_grad + eps))
    return grad_magnitude, grad_direction


def apply_occlusion_on_target_masks(
        final_img: np.ndarray, 
        target_masks: np.ndarray, 
        threshold: float=2.0,
        blf_d: int=5,
        blf_sigma_color: int=90,
        blf_sigma_space: int=75,
        inplace: bool=False,
        **kwargs
    ) -> np.ndarray:
    """
    This function tries to estimate occluded areas in the image and apply to the target masks accordingly
    It converts the final image to grayscale, computes the contour map with two sobel filters for each axis,
    normalizes the contour map, scales the target mask by the contour mask, thresholds the mask and multiplies
    it by 255. It then applies a Gaussian filter to fill in the gaps within thicker lines then thresholds the
    final mask and multiplies by 255.
    The hope here is to occlude only regions where we are certain that the signal region has been completely (100%)
    occluded by stain.
    """
    assert final_img.ndim == 2 or (final_img.ndim == 3 and final_img.shape[2] == 3)
    
    if final_img.ndim == 3:
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
        
    img_contour = get_sobel_gradient(final_img, compute_direction=False, **kwargs)
    img_contour /= (img_contour.max() + 1e-5)
    
    if not inplace:
        target_masks = target_masks.copy()
    
    if target_masks.ndim == 2:
        target_masks = target_masks[:, :, None]

    for i in range(target_masks.shape[2]):
        mask = target_masks[:, :, i]
        occluded_mask = (img_contour * mask) > threshold
        occluded_mask = (occluded_mask * 255).astype(np.uint8)
        occluded_mask = cv2.bilateralFilter(occluded_mask, d=blf_d, sigmaColor=blf_sigma_color, sigmaSpace=blf_sigma_space)
        occluded_mask = ((occluded_mask > 0) * 255).astype(np.uint8)
        target_masks[:, :, i] = occluded_mask

    return np.squeeze(target_masks)


def generate_random_kernel_size(k_min: int, k_max: int) -> int:
    # kernel size (for gaussian filters) needs to be odd number
    k = np.random.randint(k_min, k_max)
    if k % 2 == 0:
        k += 1
    if k >= k_max:
        k -= 2
    return k


def generate_random_color(c_range: Tuple[int, int]=(0, 255)) -> Tuple[int, int, int]:
    return (*np.random.randint(*c_range, size=(3, ), dtype=np.uint8).tolist(), )


def generate_random_color_contrasts(
        color: Tuple[int, int, int],
        contrast_range: Optional[Tuple[float, float]]=None,
        brightness_range: Optional[Tuple[float, float]]=None,
        add_contrast: bool=True,
        add_brightness: bool=False
    ) -> Tuple[int, int, int]:

    color = np.asarray(color) / 255

    if add_contrast:
        if not contrast_range:
            contrast_range = (0.8, 1.2)
        contrast = np.random.uniform(*contrast_range)
        color = contrast * color

    if add_brightness:
        if not brightness_range:
            brightness_range = (-0.3, 0.3)
        brightness = np.random.uniform(*brightness_range)
        color = color + brightness

    color = np.clip(color * 255, a_min=0, a_max=255).astype(np.uint8)
    return (*color.tolist(),)


def invert_channels(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]


def rl_encode(arr: np.ndarray) -> np.ndarray:
    starts = np.where(arr[:-1] != arr[1:])[0] + 1
    starts = np.concatenate([[0], starts], axis=0)
    ends = np.concatenate([starts[1:], [arr.shape[0]]], axis=0)
    counts = ends - starts
    mask = arr[starts] > 0
    starts = starts[mask]
    counts = counts[mask]
    return np.stack([starts, counts], axis=1)


def rl_decode(enc: np.ndarray, arr_size: int, arr_value: int=255, dtype="uint8") -> np.ndarray:
    assert enc.ndim == 2 and enc.shape[1] == 2
    arr = np.zeros((arr_size, ), dtype=np.float32)
    arr[enc[:, 0]] = arr_value
    arr[enc[:, 0] + enc[:, 1]] = -arr_value
    return np.cumsum(arr).astype(getattr(np, dtype))


def coco_rl_encode(arr: np.ndarray) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if not arr.flags["F_CONTIGUOUS"]:
        arr = arr.astype(arr.dtype, order="F")
    
    non_zeros = arr[non_zeros_mask := arr > 0].copy()
    rle = mask_utils.encode(non_zeros_mask)
    encoded = dict(non_zeros=non_zeros, rle=rle)
    return encoded


def coco_rl_decode(
        enc: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]], 
        normalize: bool=False, 
        dtype="uint8",
        norm_dtype="float32"
    ) -> np.ndarray:

    assert "rle" in enc and "non_zeros" in enc
    rle = enc["rle"]
    non_zeros = enc["non_zeros"]

    dec_arr = mask_utils.decode(rle)
    if not dec_arr.flags["C_CONTIGUOUS"]:
        dec_arr = dec_arr.astype(getattr(np, dtype), order="C")
    dec_arr[dec_arr > 0] = non_zeros

    if normalize:
        dec_arr = (dec_arr / 255).astype(getattr(np, norm_dtype))
    return dec_arr


def save_coco_rl_encode(
        path: str, 
        enc: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]],
        compressed: bool=False
    ):
    assert isinstance(enc, dict)
    rle = enc["rle"]
    non_zeros = enc["non_zeros"]

    assert isinstance(rle, (list, dict))
    if isinstance(rle, list):
        counts = [e["counts"] for e in rle]
        sizes = [e["size"] for e in rle]
    else:
        counts = rle["counts"]
        sizes = rle["size"]
    if compressed:
        np.savez_compressed(path, non_zeros=non_zeros, counts=counts, sizes=sizes)
    else:
        np.savez(path, non_zeros=non_zeros, counts=counts, sizes=sizes)


def load_coco_rl_encode(
        path: str, 
        decode: bool=False, 
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:

    with np.load(path) as f:
        assert "non_zeros" in f and "counts" in f and "sizes" in f
        non_zeros = f["non_zeros"]
        sizes = f["sizes"]
        counts = f["counts"]

    if sizes.ndim == 2:
        rle = [{"size": sizes[i], "counts": bytes(counts[i])} for i in range(len(sizes))]
    else:
        rle = {"size": sizes, "counts": counts}

    enc = dict(non_zeros=non_zeros, rle=rle)

    if decode:
        return coco_rl_decode(enc, **kwargs)
    return enc


def get_2d_poly_features(x: np.ndarray, y: np.ndarray, order: int=5) -> np.ndarray:
    """
    get 2d polynomial features up to n order to fit a polynomial
    """
    pterms = []
    for i in range(order):
        for j in range(order + 1 - i):
            pterms += [x**i * y**j]
    return np.stack(pterms, axis=1)


def extrapolate_grid_nan(grid_coords: np.ndarray, order: int=5, inplace: bool=False) -> np.ndarray:
    """
    Fit a polynomial function to the grid coordinates to impute NaN values
    F(x, y) = a_0 + a_1 + a_2 x + a_3 y + a_4 x^2 + a_5 xy + a_6 y^2 + ...
    The coefficients of the polynomial are computed via Least Squares optimisation
    """
    H, W, D = grid_coords.shape
    nan_mask = np.isnan(grid_coords)
    known_mask =(~nan_mask).any(axis=2)

    if not nan_mask.any():
        return grid_coords

    f_yy, f_xx = np.mgrid[:H, :W]
    yy = f_yy[known_mask]
    xx = f_xx[known_mask]
    f_yy = f_yy.ravel()
    f_xx = f_xx.ravel()

    exter_coords = []
    for d in range(D):
        features = get_2d_poly_features(xx, yy, order=order)
        labels = grid_coords[:, :, d][known_mask][:, None]
        coeffs, *_ = np.linalg.lstsq(features, labels, rcond=None)
        exter_coords += [get_2d_poly_features(f_xx, f_yy, order=order) @ coeffs]
        
    exter_coords = np.stack(exter_coords, axis=1).reshape(H, W, D)

    if not inplace:
        grid_coords = grid_coords.copy()
    grid_coords[nan_mask] = exter_coords[nan_mask]
    
    return grid_coords


def generate_rectifier(
		hlines: np.ndarray, 
		vlines: np.ndarray, 
		points: np.ndarray,
		dust_threshold: int=1000,
		dust_connectivity: int=8,
		impute_nan: bool=True,
		poly_order: int=5,
        pad_grid: bool=False,
	) -> Dict[str, np.ndarray]:

    """
    Using the segmented horizontal and vertical lines, as well as the segmented grid points, this function
    generates a grid of M x N 2D points corresponding to the (x, y) centroids of each grid point which can be
    used to  undo any of the distortions introduced by the augmentation on the ECG printout and corresponding
    masks. Where M and N are number of segmented horizontal and vertical lines respectively.
    The function also returns relabeled segments for both horizontal and vertical line segments (necessary
    to compute the rectification grid)
    """
    assert hlines.ndim == vlines.ndim == points.ndim == 2
    assert hlines.shape == vlines.shape == points.shape
    
    H, W = hlines.shape
    
    hlines = cc3d.dust(
		hlines, 
		threshold=dust_threshold, 
		connectivity=dust_connectivity, 
		in_place=False
	)
    cc = cc3d.connected_components(hlines)
    stats = cc3d.statistics(cc, no_slice_conversion=True)
    hcenter = stats["centroids"][1:]

    hcc = np.zeros((H, W), dtype=np.uint8)
    for j, a in enumerate(np.argsort(hcenter[:, 0]) + 1, start=1):
        hcc[cc == a] = j
    
    vlines = cc3d.dust(
		vlines, 
		threshold=dust_threshold, 
		connectivity=dust_connectivity, 
		in_place=False
	)
    cc = cc3d.connected_components(vlines) 
    stats = cc3d.statistics(cc, no_slice_conversion=True)
    vcenter = stats["centroids"][1:]
    
    vcc = np.zeros((H, W), dtype=np.uint8)
    for j, a in enumerate(np.argsort(vcenter[:, 1]) + 1, start=1):
        vcc[cc == a] = j
    
    cc = cc3d.connected_components(points)
    stats = cc3d.statistics(cc, no_slice_conversion=True)
    pcenter = stats["centroids"][1:]
    grid_xy = np.full(
		(hcenter.shape[0], vcenter.shape[0], 2), 
		fill_value=np.nan, 
		dtype=np.float32
	)
    for y, x in pcenter:
        uy = round(y)
        ux = round(x)
        a = int(hcc[uy, ux]) - 1
        b = int(vcc[uy, ux]) - 1
        if 0<=a<grid_xy.shape[0] and 0<=b<grid_xy.shape[1]:
            grid_xy[a, b] = [x, y]

    if impute_nan:
        grid_xy = extrapolate_grid_nan(grid_xy, order=poly_order, inplace=True)

    if pad_grid:
        # pad grid with out of bound values so the grid sampler pads the edges of the rectified image
        grid_xy = np.pad(grid_xy, pad_width=[(1, 1),(1, 1),(0, 0)], mode="constant", constant_values=-9_999_999)
		
    return {"grid_xy": grid_xy, "hcc": hcc, "vcc": vcc}


def rectify_img(
        img: np.ndarray, 
        hlines: np.ndarray, 
        vlines: np.ndarray, 
        points: np.ndarray, 
        output_size: Optional[Tuple[int, int]]=None, 
        center_points: bool = False,
        sample_mode: str="bilinear",
        device: str="cpu",
        align_corners: bool=True,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    """Rectify image with horizontal lines, vertical lines and grid point segments"""
    rect_dict = generate_rectifier(hlines, vlines, points, **kwargs)
    rectified = grid_sample(
        img, 
        rect_dict["grid_xy"], 
        size=output_size or img.shape[:2], 
        center_points=center_points,
        sample_mode=sample_mode, 
        device=device, 
        align_corners=align_corners
    )
    return rectified, rect_dict