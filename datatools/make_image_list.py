import os 
import pandas as pd
import numpy as np
import random
import h5py
import tqdm
import io
from PIL import Image


def TCGADataset(data_dir, token_num, outdir, crop_size=256):

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

    