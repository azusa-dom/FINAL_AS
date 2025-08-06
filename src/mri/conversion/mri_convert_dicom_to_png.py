#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_dicom_to_png.py

Batch-convert a directory of .dcm files to .png for SimCLR pretraining.
"""
import os
import argparse
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, fn)
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            # Normalize to [0,255]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
            img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
            outname = os.path.splitext(fn)[0] + '.png'
            img.save(os.path.join(output_dir, outname))
            count += 1
    print(f"Converted {count} DICOM → PNG into: {output_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',  required=True, help='Path to folder with .dcm files')
    p.add_argument('--output-dir', required=True, help='Where to save .png files')
    args = p.parse_args()
    convert_folder(args.input_dir, args.output_dir)