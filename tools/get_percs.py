import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
import cv2
import argparse
import tqdm

classes_inset = ['AMBIGUOUS' 'nonTIL_stromal' 'other_nucleus' 'sTIL' 'tumor_any']

mapping_class = {
    'nonTIL_stromal': 0,
    'sTIL': 1,
    'tumor_any': 2,
    'AMBIGUOUS': 3,
    'other_nucleus': 3,
}


# outpath = "/Users/mraoaakash/Documents/research/research-personal/threshfinder/data/nucls/ind_percs/"


def get_percs(base_path):
    outpath = os.path.join(base_path, 'data/nucls/ind_percs')
    os.makedirs(outpath, exist_ok=True)

    csv = os.path.join(base_path, 'data/nucls/processed', 'nucls_processed.csv')

    try:
        csv = pd.read_csv(csv).drop(columns=['Unnamed: 0'])
    except:
        raise Exception("CSV not found")

    perc_df = pd.DataFrame(columns=['image_name', 'image_path', 'mask_path', 'perc_0', 'perc_1', 'perc_2', 'perc_3', 'perc_bg'])
    img_names = []
    img_paths = []
    mask_paths = []
    perc_0 = []
    perc_1 = []
    perc_2 = []
    perc_3 = []
    perc_bg = []

    print("Calculating percentages for {} images".format(len(csv)))
    for index, row in tqdm.tqdm(csv.iterrows(), total=len(csv)):
        image_name = row['image_path'].split('/')[-1].split('.png')[0]
        # image = cv2.imread(row['image_path'])
        mask = cv2.imread(row['mask_path'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)-1

        # print(mask)
        # plt.figure()
        # plt.imshow(mask)
        # plt.show()

        img_names.append(image_name)
        img_paths.append(row['image_path'])
        mask_paths.append(row['mask_path'])
        mask = np.array(mask).flatten()
        # print(mask)
        perc_0.append(np.sum(mask==0)/len(mask))
        perc_1.append(np.sum(mask==1)/len(mask))
        perc_2.append(np.sum(mask==2)/len(mask))
        perc_3.append(np.sum(mask==3)/len(mask))
        perc_bg.append(np.sum(mask==255)/len(mask))


        # break

    # print("perc_0 = {}, \nperc_1 = {}, \nperc_2 = {}, \nperc_3 = {}, \nperc_bg = {}".format(perc_0, perc_1, perc_2, perc_3, perc_bg))
    perc_df['image_name'] = img_names
    perc_df['image_path'] = img_paths
    perc_df['mask_path'] = mask_paths
    perc_df['perc_0'] = [i*100 for i in perc_0]
    perc_df['perc_1'] = [i*100 for i in perc_1]
    perc_df['perc_2'] = [i*100 for i in perc_2]
    perc_df['perc_3'] = [i*100 for i in perc_3]
    perc_df['perc_bg'] = [i*100 for i in perc_bg]

    # round to 3 decimal places
    perc_df = perc_df.round(3)
    perc_df.to_csv(os.path.join(outpath, 'perc_df.csv'))
    pass

def label_threshold(base_path):
    outpath = os.path.join(base_path, 'data/nucls/ind_percs/thresholds')
    os.makedirs(outpath, exist_ok=True)

    csv = os.path.join(base_path, 'data/nucls/ind_percs', 'perc_df.csv')
    try:
        csv = pd.read_csv(csv).drop(columns=['Unnamed: 0'])
    except:
        get_percs(base_path)
        csv = pd.read_csv(csv).drop(columns=['Unnamed: 0'])
    


    print("Creating thresholded labels for {} images".format(len(csv)))
    for threshold in tqdm.tqdm(range(1, 101)):
        # file name with leading zeros 
        filename = os.path.join(outpath, f"perc_{str(threshold).zfill(3)}.csv")
        df_copy = csv.copy()
        df_copy.drop(columns=['perc_bg'], inplace=True)
    
        # replace all values below threshold with 0 and above with 1
        for i in range (0, 4):
            df_copy[f"perc_{i}"] = df_copy[f"perc_{i}"].apply(lambda x: 0 if x < threshold else 1)

        df_copy.drop(columns=['mask_path'], inplace=True)
        df_copy.to_csv(filename)

        # Printing and breaking for debugging purposes
        # print(df_copy.head())
        # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--base_path', type=str, required=True)
    args = parser.parse_args()
    # get_percs(args.base_path)
    label_threshold(args.base_path)
