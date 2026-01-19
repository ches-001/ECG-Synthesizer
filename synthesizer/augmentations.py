import os
import random
import numpy as np
from .utils import(
    apply_grid_padding, 
    crop_grid,
    apply_noise,
    apply_distortions,
    apply_background,
    apply_crease,
    apply_wrinkle_texture,
    apply_contrast,
    apply_color_gradient,
    apply_stains,
    to_grayscale,
    apply_blur,
    generate_random_kernel_size,
    generate_random_color,
    generate_random_color_contrasts,
    invert_channels,
)
from typing import Tuple, Dict, Optional, Any


BG_COLOR       = (0, 255, 0)
BG_COLOR_RANGE = ((0, 100, 0), (100, 255, 100))


def stain_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p,
        "num_circles_range": (10, 700),
        "region_x_range": (0, 2016),
        "region_y_range": (0, 1548),
        "region_w_range": (112, 1120),
        "region_h_range": (86, 860),
        "kernel_size_range": (20, 202),
        "sigmas_range": (10, 60),
        "num_grains_range": (2000, 70000),
        "alpha_range": (0.0, 0.5),
    }
    H, W = img.shape[:2]
    if kwargs["p"] > np.random.uniform():
        img = apply_stains(
            img,
            num_circles=np.random.randint(*kwargs["num_circles_range"]),
            region=(
                (
                    x:=np.random.randint(*kwargs["region_x_range"]) / W,
                    y:=np.random.randint(*kwargs["region_y_range"]) / H,
                ),
                (
                    np.clip(x + np.random.randint(*kwargs["region_w_range"]) / W, a_min=0, a_max=1),
                    np.clip(y + np.random.randint(*kwargs["region_h_range"]) / H, a_min=0, a_max=1)
                )
            ),
            color=random.choice([(0, 0, 0), generate_random_color(c_range=(0, 5))]),
            kernel_size=(generate_random_kernel_size(*kwargs["kernel_size_range"]), )*2,
            sigmas=(np.random.uniform(*kwargs["sigmas_range"]), )*2,
            num_grains=np.random.randint(*kwargs["num_grains_range"]),
            alpha=np.random.uniform(*kwargs["alpha_range"]),
            stain_type=random.choice(["additive", "subtractive", "multiplicative"]),
        )
    return img


def contrast_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p,
        "contrast_range": (0.9, 1.1),
        "brightness_range": (-0.2, 0.2),
    }
    if kwargs["p"] > np.random.uniform():
        img = apply_contrast(
            img,
            contrast=np.random.uniform(*kwargs["contrast_range"]),
            brightness=np.random.uniform(*kwargs["brightness_range"])
        )
    return img


def color_gradient_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p,
        "alpha_range": (0.05, 0.3),
    }
    if kwargs["p"] > np.random.uniform():
        blue = (100, 120, 255)
        yellow = (255, 205, 80)
        c1, c2 = None, None
        while c1 is None and c2 is None:
            c1 = random.choice([None, blue, yellow])
            c2 = random.choice([None, blue, yellow])
            if c1:
                c1 = generate_random_color_contrasts(c1)
            if c2:
                c2 = generate_random_color_contrasts(c2)
            
        img = apply_color_gradient(
            img,
            c1=c1,
            c2=c2,
            alpha=np.random.uniform(*kwargs["alpha_range"]),
            along=random.choice(["vertical", "horizontal"]),
            grad_type=random.choice(["linear", "sinusoidal"])
        )
    return img


def crease_augmentation(
        img: np.ndarray, 
        kwargs: Dict[str, Any], 
        default_p: float=0.5,
        ignore_mask: Optional[np.ndarray]=None
    ) -> np.ndarray:

    kwargs = kwargs or {
        "p": default_p,
        "num_lines_range": (1, 15),
        "theta_range": (np.pi/2 - np.pi/60, np.pi/2 + np.pi/60),
        "kernel_size_range": (3, 9),
        "sigmas_range": (0.5, 3.0),
        "line_thickness_range": (5, 7)
    }
    if kwargs["p"] > np.random.uniform():
        img = apply_crease(
            img,
            num_lines=np.random.randint(*kwargs["num_lines_range"]),
            theta=np.random.uniform(*kwargs["theta_range"]),
            kernel_size=(generate_random_kernel_size(*kwargs["kernel_size_range"]), )*2,
            sigmas=(np.random.uniform(*kwargs["sigmas_range"]), )*2,
            line_thickness=np.random.randint(*kwargs["line_thickness_range"]),
            ignore_mask=ignore_mask
        )
    return img


def wrinkle_augmentation(
        img: np.ndarray, 
        texture_dir: str, 
        kwargs: Dict[str, Any], 
        default_p: float=0.5,
        ignore_mask: Optional[np.ndarray]=None
    ) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p,
        "block_size_range": (128, 200),
        "num_xpatches_range": (1, 4),
        "num_ypatches_range": (1, 4),
        "mean_shift_range": (0.1, 0.5),
        "map_threshold_range": (0.4, 0.7)
    }

    if kwargs["p"] > np.random.uniform() and texture_dir and os.listdir(texture_dir):
        img = apply_wrinkle_texture(
            img,
            block_size=np.random.randint(*kwargs["block_size_range"]),
            num_xpatches=np.random.randint(*kwargs["num_xpatches_range"]),
            num_ypatches=np.random.randint(*kwargs["num_ypatches_range"]),
            mean_shift=np.random.uniform(*kwargs["mean_shift_range"]),
            map_threshold=np.random.uniform(*kwargs["map_threshold_range"]),
            textures_dir=texture_dir,
            ignore_mask=ignore_mask
        )
    return img


def noise_augmentation(
        img: np.ndarray, 
        kwargs: Dict[str, Any], 
        default_p: float, 
        ignore_mask: Optional[np.ndarray]=None
    ) -> np.ndarray:

    kwargs = kwargs or {
        "p": default_p,
        "mean_range": (0, 0.1),
        "std_range": (0.001, 0.2),
        "poisson_rate_range": (1, 30),
        "n_snp_range": (4_000, 50_000),
        "perlin_grid_size": (32, 513),
        "perlin_alpha": (0.1, 0.5)
    }

    if kwargs["p"] > np.random.uniform():
        noise_opts = ["snp", "gaussian", "speckle", "poisson", "perlin"]
        _N = len(noise_opts)
        for _ in range(0, _N):
            noise_type = random.choice(noise_opts)
            # ensure that no two noise types is selected more than once
            noise_opts.remove(noise_type)

            # perlin noise takes sometime to compute compared to the rest, so leave it to.
            # chance to save compute and time
            if noise_type == "perlin" and (not random.choice([True, False])):
                continue
            img = apply_noise(
                img,
                noise_type=noise_type,
                mean=np.random.uniform(*kwargs["mean_range"]),
                std=np.random.uniform(*kwargs["std_range"]),
                poisson_rate=np.random.uniform(*map(int, kwargs["poisson_rate_range"])),
                n_snp=np.random.randint(*map(int, kwargs["n_snp_range"])),
                salt_val=np.random.randint(200, 255, size=(3, )),
                pepper_val=np.random.randint(0, 20, size=(3, )),
                perlin_grid_size=np.random.randint(*kwargs["perlin_grid_size"]),
                perlin_alpha=np.random.uniform(*kwargs["perlin_alpha"]),
                one_channel_noise=random.choice([True, False]),
                ignore_mask=ignore_mask
            )
    return img


def invert_color_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p
    }
    if kwargs["p"] > np.random.uniform():
        img = invert_channels(img)
    return img


def grayscale_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p
    }
    if kwargs["p"] > np.random.uniform():
        img = to_grayscale(img, with_three_channels=True)
    return img


def blur_augmentation(img: np.ndarray, kwargs: Dict[str, Any], default_p: float=0.5) -> np.ndarray:
    kwargs = kwargs or {
        "p": default_p,
        "kernel_size_range": (3, 15),
        "sigmas_range": (0.5, 5.0)
    }
    if kwargs["p"] > np.random.uniform():
        img = apply_blur(
            img, 
            kernel_size=(
                generate_random_kernel_size(*kwargs["kernel_size_range"]),
                generate_random_kernel_size(*kwargs["kernel_size_range"]),
            ),
            sigmas=(
                np.random.uniform(*kwargs["sigmas_range"]), 
                np.random.uniform(*kwargs["sigmas_range"])
            )
        )
    return img


def crop_augmentation(
        img: np.ndarray, 
        kwargs: Dict[str, Any], 
        default_p: float=0.5, 
        ignore_mask: Optional[np.ndarray]=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    kwargs = kwargs or {
        "p": default_p,
        "crop_top": (0, 172),
        "crop_bottom": (0, 86),
        "crop_left": (0, 112),
        "crop_right": (0, 112)
    }
    H, W = img.shape[:2]
    if kwargs["p"] > np.random.uniform():
        vrange = (
            np.random.randint(*kwargs["crop_top"]), 
            H-np.random.randint(*kwargs["crop_bottom"])
        )
        hrange = (
            np.random.randint(*kwargs["crop_left"]), 
            W-np.random.randint(*kwargs["crop_right"])
        )
        img = crop_grid(img, vrange=vrange, hrange=hrange)
        if ignore_mask is not None:
            ignore_mask = crop_grid(ignore_mask[:, :, None], vrange=vrange, hrange=hrange)[:, :, 0]
    return img, ignore_mask


def pad_augmentation(
        img: np.ndarray, 
        kwargs: Dict[str, Any], 
        default_p: float=0.5,
        set_green_screen: bool=False,
    ) -> np.ndarray:

    kwargs = kwargs or {
        "p": default_p,
        "vpad_range": (30, 200),
        "hpad_range": (30, 200),
    }
    if kwargs["p"] > np.random.uniform():
        pad_val = random.choice([0, 255])
        if pad_val == 0 and set_green_screen:
            pad_val = BG_COLOR

        v_pad = np.random.randint(*kwargs["vpad_range"])
        h_pad = np.random.randint(*kwargs["hpad_range"])
        img = apply_grid_padding(
            img, 
            top=v_pad,
            bottom=h_pad, 
            left=h_pad,
            right=h_pad,
            pad_val=pad_val
        )
    return img


def distortion_augmentation(
    img: np.ndarray, 
    forshorten_distortion_kwargs: Optional[Dict[str, Any]]=None,
    sinusoidal_distortion_kwargs: Optional[Dict[str, Any]]=None,
    polynomial_distortion_kwargs: Optional[Dict[str, Any]]=None,
    smooth_random_distortion_kwargs: Optional[Dict[str, Any]]=None,
    default_p: float=0.5,
    set_green_screen: bool=False,
    device: str="cpu"
) -> np.ndarray:
    
    forshorten_distortion_kwargs = forshorten_distortion_kwargs or {
        "p": default_p,
        "alpha_range": (-np.pi/90, np.pi/90),
        "beta_range": (-np.pi/90, np.pi/90),
        "gamma_range": (-np.pi/60, np.pi/60),
        "fx_range": (2000, 2100),
        "fy_range": (2000, 2100),
        "pos_x_range": (-30, 30),
        "pos_y_range": (-30, 30),
        "pos_z_range": (1900, 2100),
        "cx_range": (-20, 20),
        "cy_range": (-20, 20),
    }
    sinusoidal_distortion_kwargs = sinusoidal_distortion_kwargs or {
        "p": default_p,
        "x_amp_range": (2, 25),
        "y_amp_range": (2, 25),
        "x_lambda_range": (2000, 3500),
        "y_lambda_range": (2000, 3500),
    }
    polynomial_distortion_kwargs = polynomial_distortion_kwargs or {
        "p": default_p,
        "kx_range": (1e-8, 5e-8),
        "ky_range": (1e-8, 5e-8)
    }
    smooth_random_distortion_kwargs = smooth_random_distortion_kwargs or {
        "p": default_p,
        "x_amp_range": (10, 30),
        "y_amp_range": (10, 30),
        "sigma_range": (80, 200),
    }
    
    H, W = img.shape[:2]

    distortion_types = []
    distortions_types_kwargs_list = []
    forshorten_kwargs = dict(
        p=forshorten_distortion_kwargs["p"],
        fx=np.random.uniform(*forshorten_distortion_kwargs["fx_range"]),
        fy=np.random.uniform(*forshorten_distortion_kwargs["fy_range"]), 
        pos_xyz=(
            np.random.uniform(*forshorten_distortion_kwargs["pos_x_range"]),
            np.random.uniform(*forshorten_distortion_kwargs["pos_y_range"]),
            np.random.uniform(*forshorten_distortion_kwargs["pos_z_range"]),
        ),
        alpha=np.random.uniform(*forshorten_distortion_kwargs["alpha_range"]),
        beta=np.random.uniform(*forshorten_distortion_kwargs["beta_range"]), 
        gamma=np.random.uniform(*forshorten_distortion_kwargs["gamma_range"]),
    )
    distortion_types.append("foreshorten")
    distortions_types_kwargs_list.append(forshorten_kwargs)

    sinusoidal_kwargs = dict(
        p=sinusoidal_distortion_kwargs["p"],
        along_x=random.choice([True, False]),
        along_y=random.choice([True, False]),
        x_amp=np.random.uniform(*sinusoidal_distortion_kwargs["x_amp_range"]),
        y_amp=np.random.uniform(*sinusoidal_distortion_kwargs["y_amp_range"]),
        y_lambda=np.random.uniform(*sinusoidal_distortion_kwargs["x_lambda_range"]),
        x_lambda=np.random.uniform(*sinusoidal_distortion_kwargs["y_lambda_range"]),
    )
    distortion_types.append("sinusoidal")
    distortions_types_kwargs_list.append(sinusoidal_kwargs)

    polynomial_kwargs = dict(
        p=polynomial_distortion_kwargs["p"],
        kx=np.random.choice([-1, 1]) * np.random.uniform(*polynomial_distortion_kwargs["kx_range"]),
        ky=np.random.choice([-1, 1]) * np.random.uniform(*polynomial_distortion_kwargs["ky_range"])
    )
    distortion_types.append("polynomial")
    distortions_types_kwargs_list.append(polynomial_kwargs)

    smooth_random_kwargs = dict(
        p=smooth_random_distortion_kwargs["p"],
        x_amp=np.random.uniform(*smooth_random_distortion_kwargs["x_amp_range"]),
        y_amp=np.random.uniform(*smooth_random_distortion_kwargs["y_amp_range"]),
        sigma=np.random.uniform(*smooth_random_distortion_kwargs["sigma_range"])
    )
    distortion_types.append("smooth_random")
    distortions_types_kwargs_list.append(smooth_random_kwargs)
    n_dtypes = len(distortion_types)

    if n_dtypes > 0:
        distortion_indexes = np.arange(n_dtypes)
        np.random.shuffle(distortion_indexes)
        
        distortion_types = [distortion_types[i] for i in distortion_indexes]
        distortions_types_kwargs_list = [distortions_types_kwargs_list[i] for i in distortion_indexes]
        img = apply_distortions(
            img, 
            (H, W), 
            types_list=distortion_types,
            kwargs_list=distortions_types_kwargs_list,
            center_points=True,
            oob_padvals=BG_COLOR if set_green_screen else None, 
            oob_ch_start=0, 
            oob_ch_stop=3,
            device=device
        )
    return img


def background_augmentation(
        img: np.ndarray,
        apply: bool,
        texture_dir: str,
        kwargs: Dict[str, Any], 
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    kwargs = kwargs or {
        "h_range": (10, 1000),
        "w_range": (10, 1000),
    }
    H, W = img.shape[:2]
    bg_mask = None
    if apply:
        img, bg_mask = apply_background(
            img, 
            background_dir=texture_dir,
            bg_color=BG_COLOR,
            bg_color_range=BG_COLOR_RANGE,
            size=(
                H + np.random.randint(*kwargs["h_range"]),
                W + np.random.randint(*kwargs["w_range"])
            )
        )
    return img, bg_mask