# Synthetic ECG Generator

This codebase was created for the sole purpose of generating synthetic ECG printout images meant to resemble realistic ones. It was designed specifically for the [2025 PhysioNet - Digitization of ECG Images Challenge](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/overview) and as such can only produce ECG images with the lead format and layout used in this competition (the code can be adjusted to accommodate for other layouts).

With this you can generate ECG images as well as segmentation masks for grid lines, signals, texts and markers. All you need are a bunch of CSV files containing the ECG digitized readings.

## HOW TO USE:

There are several augmentation techniques implemented in this codebase to replicate realistic distortions and artifacts and you can combine all these augmentations to generate random samples automatically with a single function. Each augmentation has a probability value associated with it and the value indicates the chances of the augmentation being applied to the image to produce the final image.

The function returns the image and all corresponding masks. The masks are in the following order:

The returned segmentation mask of this function has 9 channels:

    channel 0: Horizontal lines segment

    channel 1: Vertical lines segment

    channel 2: Center points segment, where horizontal and vertical lines intersect

    channel 3: Lead channels segments for first row (I, aVR, V1, V4)

    channel 4: Lead channels segments for second row (II, aVL, V2, V5)

    channel 5: Lead channels segments for third row (III, aVF, V3, V6)

    channel 6: Lead channels segments for fourth row (II)

    channel 7: Lead text labels segment

    channel 8: Short verical lines dividing lead signals


### Example Code:
```python

    from synthesizer.generate import generate_ecg_sample

    signal_df_path = "data/train/112870634/112870634.csv"

    wrinkle_dir = "data/wrinkle_textures"
    bg_dir = "data/background_textures"

    img, masks = generate_ecg_sample(
        signal_df_path, 
        wrinkles_dir=wrinkle_dir, 
        background_dir=bg_dir, 
        rle_masks=True,
    )
```

You can also generate a lot of synthetic samples automatically over the command line by running this:

Generate in parallel with multiple CPU cores:

```sh
python -m synthesizer.parallel \
        --config_path = config/synthesizer_config.yaml \
        --input_dir = {csv_input_dir} \
        --ext = jpg \
        --output_dir = {output_dir} \
        --wrinkles_dir=data/wrinkle_textures \
        --background_dir = data/background_textures \
        --sample_strat = random \
        --max_workers = {max_workers} \
        --nsamples = {nsamples} \
        --compress_rle
```

Generate with single CPU core:

```sh
python -m synthesizer \
        --config_path = config/synthesizer_config.yaml \
        --input_dir = {csv_input_dir} \
        --ext = jpg \
        --output_dir = {output_dir} \
        --wrinkles_dir=data/wrinkle_textures \
        --background_dir = data/background_textures \
        --sample_strat = random \
        --max_workers = {max_workers} \
        --nsamples = {nsamples} \
        --compress_rle
```

You can also generate samples with all the augmentations except the distortions that effect grid space orientation with the `--rectify` flag