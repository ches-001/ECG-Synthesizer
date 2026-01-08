import os
import tqdm
import yaml
import logging
import random
import logging
import argparse
from multiprocessing import Semaphore
from concurrent.futures import ProcessPoolExecutor
from .__main__ import validate_and_get_files, generate_single, SAMPLE_STRAT, DEVICES, DEFAULT_CONFIG_PATH


LOGGER = logging.getLogger(__name__)
MAX_CPU_CORES = os.cpu_count()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ECG Synthesizer")
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config")
    parser.add_argument("--input_dir", type=str, default="data/ecgs", help="Directory of CSV files to generate ECG images from")
    parser.add_argument("--ext", type=str, default="jpg", help="Extension of image data")
    parser.add_argument("--max_workers", type=int, default=MAX_CPU_CORES, help="Maximum number of CPU workers")
    parser.add_argument("--num_concurrent_subs", type=int, default=100, help="Number of submissions before semaphore can no more be acquired")
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
    ecg_files = validate_and_get_files(args)
    nfiles = len(ecg_files)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    LOGGER.info(f"Generating {args.nsamples} synthetic ECG samples with {args.max_workers} CPU workers")
    pbar = tqdm.tqdm(total=args.nsamples)
    semaphore = Semaphore(args.num_concurrent_subs)
    
    with ProcessPoolExecutor(args.max_workers) as pool:
        for i in range(args.nsamples):
            if args.sample_strat == SAMPLE_STRAT[0]:
                ecg_path = ecg_files[i % nfiles]
            else:
                ecg_path = random.choice(ecg_files)
            output_dir = os.path.join(args.output_dir, str(i).zfill(5))
            semaphore.acquire()
            fut = pool.submit(generate_single, args, config, ecg_path, output_dir)
            fut.add_done_callback(lambda fut : [semaphore.release(), pbar.update(1)])
    pbar.close()
