import os
import logging
import pandas as pd
import numpy as np
from .utils import(
    global_change_scale,
    draw_signal,
    draw_grid,
    generate_random_color_contrasts,
    coco_rl_encode,
)
from .augmentations import *
from typing import Tuple, Dict, Optional, Any, List, Union

LOGGER = logging.getLogger(__name__)


def generate_ecg_sample(
        ecg_df_path: str,
        wrinkles_dir: Optional[str]=None,
        background_dir: Optional[str]=None,
        signal_color_brange: Tuple[float, float]=(0.0, 0.3),
        padding_kwargs: Optional[Dict[str, Any]]=None,
        forshorten_distortion_kwargs: Optional[Dict[str, Any]]=None,
        sinusoidal_distortion_kwargs: Optional[Dict[str, Any]]=None,
        polynomial_distortion_kwargs: Optional[Dict[str, Any]]=None,
        smooth_random_distortion_kwargs: Optional[Dict[str, Any]]=None,
        background_kwargs: Optional[Dict[str, Any]]=None,
        crease_kwargs: Optional[Dict[str, Any]]=None,
        wrinkle_kwargs: Optional[Dict[str, Any]]=None,
        contrast_kwargs: Optional[Dict[str, Any]]=None,
        noise_kwargs: Optional[Dict[str, Any]]=None,
        color_gradient_kwargs: Optional[Dict[str, Any]]=None,
        stains_kwargs: Optional[Dict[str, Any]]=None,
        crop_kwargs: Optional[Dict[str, Any]]=None,
        grayscale_kwargs: Optional[Dict[str, float]]=None,
        blur_kwargs: Optional[Dict[str, float]]=None,
        invert_channels_kwargs: Optional[Dict[str, float]]=None,
        rle_masks: bool=True,
        default_p: float=0.5,
        return_rectified: bool=False,
        scale: int=1,
        device: str="cpu"
    ) -> Tuple[np.ndarray, Union[Dict[str, Any], List[Dict[str, Any]], np.ndarray]]:
    """
    The returned segmentation mask of this function has 9 channels:

    channel 0: Horizontal lines segment

    channel 1: Vertical lines segment

    channel 2: Center points segment, where horizontal and vertical lines intersect

    channel 3: Lead channel segments for first row (I, aVR, V1, V4)

    channel 4: Lead channel segments for second row (II, aVL, V2, V5)

    channel 5: Lead channel segments for third row (III, aVF, V3, V6)

    channel 6: Lead channel segments for fourth row (II)

    channel 7: Lead text labels segment

    channel 8: Short verical lines dividing lead signals
    """
    scale = global_change_scale(scale)

    if not os.path.isfile(ecg_df_path):
        LOGGER.error(f"ERROR: {ecg_df_path} file is not found")
        return
    
    QR_BASEWIDTH = np.random.randint(150*scale, 301*scale)

    thick_lines_color = np.asarray([255, 70, 70])
    thick_lines_color[1:] = (thick_lines_color[1:] / np.random.randint(1, 4)).astype(np.uint8)
    thin_lines_color = np.clip(thick_lines_color + 140, a_min=0, a_max=255)
    thick_lines_color = (*thick_lines_color.tolist(), )
    thin_lines_color = (*thin_lines_color.tolist(), )
    
    line_thickness = np.random.randint(1*scale, 3*scale)
    point_radius = line_thickness

    img = draw_grid(
        thickness=line_thickness,
        seg_thickness=line_thickness,
        point_radius=point_radius,
        thick_color=thick_lines_color,
        thin_color=thin_lines_color,
        with_annotations=True,
        bg_color=(255, 255, int(np.random.randint(225, 256))),
        qr_code=True,
        qr_code_data="random text for QR code",
        qr_code_width=QR_BASEWIDTH,
    )
    lines_segments = img[:, :, 3:]
    img = img[:, :, :3]
    
    signal_df = pd.read_csv(ecg_df_path)

    sig_thickness = np.random.randint(1*scale, 3*scale)
    # given how sensitive signal segments are, it makes sense to make it this thin
    sig_mask_thickness = 1
    font_scale = np.random.uniform(1.0*scale, 1.2*scale)
    text_thickness = np.random.randint(2*scale, 4*scale)
    tick_thickness = sig_thickness + (4 * scale)
    num_points_scale = 20 * scale

    img = draw_signal(
        img,
        signal_df,
        sig_color=generate_random_color_contrasts(
            (0, 0, 0),
            add_contrast=False, 
            add_brightness=True, 
            brightness_range=signal_color_brange
        ),
        sig_thickness=sig_thickness,
        sig_mask_thickness=sig_mask_thickness,
        tick_thickness=tick_thickness,
        text_thickness=text_thickness,
        font_scale=font_scale,
        ignore_mask_sq_pulser=True,
        with_annotations=True,
        num_points_scale=num_points_scale,
    )
    signal_segments = img[:, :, 3:]
    img = img[:, :, :3]
    img = stain_augmentation(img, stains_kwargs, default_p)
    segments = np.concatenate([lines_segments, signal_segments], axis=2)
    bg_mask = None

    if not return_rectified:
        img = np.concatenate([img, segments], axis=2)

        background_kwargs = background_kwargs or {
            "p": default_p,
            "h_range": (10, 1000),
            "w_range": (10, 1000),
        }
        # if set_green_screen is True, all padded areas are filled with pure
        # greens pixels to make the applied background texture seem natural
        set_green_screen = (
            background_kwargs["p"] > np.random.uniform() 
            and background_dir 
            and os.listdir(background_dir)
        )
        img = pad_augmentation(img, padding_kwargs, default_p, set_green_screen)

        img = distortion_augmentation(
            img,
            forshorten_distortion_kwargs, 
            sinusoidal_distortion_kwargs, 
            polynomial_distortion_kwargs, 
            smooth_random_distortion_kwargs,
            default_p,
            set_green_screen,
            device
        )

        img, bg_mask = background_augmentation(img, set_green_screen, background_dir, background_kwargs)
        img, bg_mask = crop_augmentation(img, crop_kwargs, default_p, bg_mask)

        segments = img[:, :, 3:]
        img = img[:, :, :3]

    img = contrast_augmentation(img, contrast_kwargs, default_p)
    img = color_gradient_augmentation(img, color_gradient_kwargs, default_p)
    img = crease_augmentation(img, crease_kwargs, default_p, bg_mask)
    img = wrinkle_augmentation(img, wrinkles_dir, wrinkle_kwargs, default_p, bg_mask)
    img = noise_augmentation(img, noise_kwargs, default_p, bg_mask)
    img = invert_color_augmentation(img, invert_channels_kwargs, default_p)
    img = grayscale_augmentation(img, grayscale_kwargs, default_p)
    img = blur_augmentation(img, blur_kwargs, default_p)

    if not rle_masks:
        return img, segments
    
    segments = coco_rl_encode(segments)
    return img, segments