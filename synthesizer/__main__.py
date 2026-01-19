import os
import cv2
import sys
import tqdm
import glob
import yaml
import shutil
import random
import logging
import argparse
from .generate import generate_ecg_sample
from .utils import save_coco_rl_encode
from typing import List, Dict, Any


LOGGER = logging.getLogger(__name__)
SAMPLE_STRAT = ["circle", "random"]
DEVICES = ["cpu", "cuda"]
DEFAULT_CONFIG_PATH = "config/synthesizer_s1_config.yaml"


def validate_and_get_files(args: argparse.Namespace) -> List[str]:
    if not os.path.isfile(args.config_path):
        LOGGER.error(f"Config file {args.config_path} is not found")
        sys.exit(1)
    
    if args.device not in DEVICES:
        LOGGER.error(f"device is expected to be one of {DEVICES}, got {args.device}")
        sys.exit(1) 
    
    if not os.path.isdir(args.input_dir):
        LOGGER.error(f"The Directory {args.input_dir} cannot be found!")
        sys.exit(1)
    
    input_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    num_files = len(input_files)

    if num_files == 0:
        LOGGER.error(f"Number of ECG CSV files in {args.input_dir} is 0")
        sys.exit(1)
    
    if args.sample_strat not in SAMPLE_STRAT:
        LOGGER.error(f"sample_strat can only be one of {SAMPLE_STRAT}")
        sys.exit(1)

    return input_files


def generate_single(args: argparse.Namespace, config: Dict[str, Any], ecg_path: str, output_dir: str):
    try:
        img, rle_mask = generate_ecg_sample(
            ecg_path, 
            wrinkles_dir=args.wrinkles_dir,
            background_dir=args.background_dir,
            device=args.device,
            return_rectified=args.rectify,
            rle_masks=True,
            scale=config["scale"],
            **config["generator"]
        )
    except Exception as e:
        LOGGER.error(f"Error generating ECG image for file {ecg_path}")
        LOGGER.error(e, stack_info=True, exc_info=True)
        return
    os.makedirs(output_dir, exist_ok=True)
    # cv2 saves files in BGR format instead of RGB
    cv2.imwrite(os.path.join(output_dir, f"ecg_img.{args.ext}"), img[:, :, ::-1])
    save_coco_rl_encode(os.path.join(output_dir, "segments.npz"), rle_mask, compressed=args.compress_rle)
    shutil.copy(ecg_path, output_dir)


def generate_batch(args: argparse.Namespace):
    input_files = validate_and_get_files(args)
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    LOGGER.info(f"Generating {args.nsamples} synthetic ECG samples")
    for i in tqdm.tqdm(range(args.nsamples), total=args.nsamples):
        if args.sample_strat == SAMPLE_STRAT[0]:
            file = input_files[i % len(input_files)]
        else:
            file = random.choice(input_files)
        output_dir = os.path.join(args.output_dir, f"{str(i).zfill(5)}")
        generate_single(args, config=config, ecg_path=file, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ECG Synthesizer")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config")
    parser.add_argument("--input_dir", type=str, default="data/ecgs", help="Directory of CSV files to generate ECG images from")
    parser.add_argument("--ext", type=str, default="jpg", help="Extension of image data")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Directory to store generated ECG samples")
    parser.add_argument("--wrinkles_dir", type=str, default="data/wrinkle_textures", help="Directory where wrinkle textures are stored")
    parser.add_argument("--background_dir", type=str, default="data/background_textures", help="Directory where background textures are stored")
    parser.add_argument("--device", type=str, default="cpu", choices=DEVICES, help="Device to perform any grid sampling related transforms")
    parser.add_argument("--nsamples", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--sample_strat", type=str, default=SAMPLE_STRAT[1], choices=SAMPLE_STRAT, help="Data Sampling strategy")
    parser.add_argument("--rectify", action="store_true", help="Used to rectify augmented distorted images")
    parser.add_argument("--compress_rle", action="store_true", help="Used to compress the RLE")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parser.parse_args()

    generate_batch(args)