import os 
import pandas as pd
import numpy as np
import random
import h5py
import tqdm
import io
from PIL import Image
import argparse


def TCGADataset(data_dir, outdir, crop_size=256, token_num=75):

    os.makedirs(outdir, exist_ok=True)

    data_file = h5py.File(os.path.join(data_dir, "TCGA_BRCA_10x_448_tumor.hdf5"), "r")

    train = np.load(os.path.join(data_dir, f"train_test_brca_tumor_{token_num}/train.npz"), allow_pickle=True)
    test = np.load(os.path.join(data_dir, f"train_test_brca_tumor_{token_num}/test.npz"), allow_pickle=True)
    indices_train = train["indices"]
    indices_test = test["indices"]

    indices = indices_test

    print(data_file.keys())

    # for idx in tqdm.tqdm(indices):
        
    #     tile = data_file["X"][idx]
    #     folder_name = data_file["folder_name"][idx].decode("utf-8")
    #     wsi = data_file["wsi"][idx].decode("utf-8")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create image list for training and testing')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to the data directory')
    parser.add_argument('--outdir', type=str, default="data", help='Path to the output directory')
    parser.add_argument('--crop_size', type=int, default=256, help='Size of the cropped image')
    parser.add_argument('--token_num', type=int, default=75, help='Number of tokens')
    args = parser.parse_args()

    TCGADataset(args.data_dir, args.outdir, args.crop_size, args.token_num)