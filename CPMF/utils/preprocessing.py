import os
import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image
import mvtec3d_util as mvt_util
import argparse


def preprocess_pc(tiff_path):
    # READ FILES
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    organized_rgb = np.array(Image.open(rgb_path))[:, :, :3]

    organized_gt = None
    gt_exists = os.path.isfile(gt_path)
    if gt_exists:
        organized_gt = np.array(Image.open(gt_path))
        organized_gt = organized_gt[:, :, :3]
        organized_gt = np.dot(organized_gt[..., :3], [0.299, 0.587, 0.114])
        organized_gt = np.where(organized_gt > 128, 255, 0).astype(np.uint8)

    # SAVE PREPROCESSED FILES
    tiff.imsave(tiff_path, organized_pc)
    Image.fromarray(organized_rgb).save(rgb_path)
    if gt_exists:
        Image.fromarray(organized_gt).save(gt_path)
    print("updated: ", tiff_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('--dataset_path', type=str,required=True , help='The root path of the MVTec 3D-AD. The preprocessing is done inplace (i.e. the preprocessed dataset overrides the existing one)')
    args = parser.parse_args()

    # Original method to get root path (when externally executed in terminal)
    root_path = args.dataset_path

    paths = Path(root_path).rglob('*.tiff')
    print(f"Found {len(list(paths))} tiff files in {root_path}")
    processed_files = 0
    for path in Path(root_path).rglob('*.tiff'):
        try:
            preprocess_pc(path)
        except:
            print(f'error while processing {path}, maybe have processed')
        processed_files += 1
        if processed_files % 50 == 0:
            print(f"Processed {processed_files} tiff files...")
