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
    print(train.files)
    test_til = test["prob_til"]

    test_tumor = test["prob_tumor"]

    df = pd.DataFrame(columns=["file_name", "til_score", "tumor_score"])
    for idx in tqdm.tqdm(indices):
        tile = data_file["X"][idx]
        folder_name = data_file["folder_name"][idx].decode("utf-8")
        wsi = data_file["wsi"][idx].decode("utf-8")
        file_name = wsi + "_" + folder_name
        til_score = test_til[folder_name]
        tumor_score = test_tumor[folder_name]
        df = pd.concat([df, pd.DataFrame({"file_name": [file_name], "til_score": [til_score], "tumor_score": [tumor_score]})])
    
    train_til = train["prob_til"]
    train_tumor = train["prob_tumor"]
    
    for idx in tqdm.tqdm(indices):
        tile = data_file["X"][idx]
        folder_name = data_file["folder_name"][idx].decode("utf-8")
        wsi = data_file["wsi"][idx].decode("utf-8")
        file_name = wsi + "_" + folder_name
        til_score = train_til[folder_name]
        tumor_score = train_tumor[folder_name]
        df = pd.concat([df, pd.DataFrame({"file_name": [file_name], "til_score": [til_score], "tumor_score": [tumor_score]})])
    
    
    df.to_csv(os.path.join(outdir, "set.csv"), index=False)

def split(DF, outdir, train_size=0.8, folds=3):
    outdir = os.path.join(outdir, "folds")
    train_outdir = os.path.join(outdir, "train")
    test_outdir = os.path.join(outdir, "test")

    indices = np.array(DF.index)
    np.random.seed(43)
    np.random.shuffle(indices)
    train_size = int(len(indices) * train_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    for i in range(folds):
        fold_train = train_indices[i*train_size//folds:(i+1)*train_size//folds]
        fold_test = test_indices[i*len(test_indices)//folds:(i+1)*len(test_indices)//folds]

        train_df = DF.iloc[fold_train]
        test_df = DF.iloc[fold_test]

        train_df.to_csv(os.path.join(train_outdir, f"train_{i}.csv"), index=False)
        test_df.to_csv(os.path.join(test_outdir, f"test_{i}.csv"), index=False)





    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create image list for training and testing')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to the data directory')
    parser.add_argument('--outdir', type=str, default="data", help='Path to the output directory')
    args = parser.parse_args()


    set_path = os.path.join(args.outdir, "set.csv")
    data_df = TCGADataset(args.data_dir, args.outdir) if not os.exists(set_path) else pd.read_csv(set_path)

