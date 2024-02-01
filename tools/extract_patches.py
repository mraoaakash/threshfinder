import os 
import cv2 
import pandas as pd 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tqdm

classes_inset = ['AMBIGUOUS' 'nonTIL_stromal' 'other_nucleus' 'sTIL' 'tumor_any']

mapping_class = {
    'nonTIL_stromal': 0,
    'sTIL': 1,
    'tumor_any': 2,
    'AMBIGUOUS': 3,
    'other_nucleus': 3,
}


def extract_patches(base_path, size=314):
    try:
        data_df = os.path.join(base_path, 'data/nucls', 'nucls.csv')
        data_df = pd.read_csv(data_df)
        # print(data_df.head())
    except:
        raise ValueError('Dataframe not found at {}'.format(os.path.join(base_path, 'data/nucls', 'nucls.csv')))


    if len(data_df) < 0:
        raise ValueError('No data found in dataframe')
    
    data_out_path = os.path.join(base_path, 'data/nucls', 'processed')
    im_out_path = os.path.join(data_out_path, 'images')
    mask_out_path = os.path.join(data_out_path, 'masks')
    df = pd.DataFrame(columns=['image_path', 'mask_path'])


    im_arr = []
    mask_arr = []
    os.makedirs(im_out_path, exist_ok=True)
    os.makedirs(mask_out_path, exist_ok=True)
    print('Extracting patches from {} images'.format(len(data_df)))
    for index, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
        image_path = row['image_path']
        mask_path = row['mask_path']
        image = cv2.imread(row['image_path'])
        mask = pd.read_csv(row['mask_path']).rename(columns={'Unnamed: 0': 'index'})
        mask = mask.drop(columns=['index'])
        mask = mask.replace({'super_classification': mapping_class})

        imzeros = np.zeros((image.shape[0], image.shape[1]))
        for index, row in mask.iterrows():
            xmin,  ymin,  xmax,  ymax, = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            imzeros[ymin:ymax, xmin:xmax] = int(row['super_classification'])+1

        image = image[0:size, 0:size]
        mask = imzeros[0:size, 0:size]

        # Visualization for debugging purposes:
        # plt.figure()
        # plt.imshow(image)
        # plt.imshow(mask, alpha=0.2)
        # plt.show()

        im_arr.append(os.path.join(im_out_path, os.path.basename(image_path)))
        mask_arr.append(os.path.join(mask_out_path,os.path.basename(image_path)))

        cv2.imwrite(os.path.join(im_out_path, os.path.basename(image_path)), image)
        cv2.imwrite(os.path.join(mask_out_path,os.path.basename(image_path)), mask)




        # break
    df['image_path'] = im_arr
    df['mask_path'] = mask_arr
    df.to_csv(os.path.join(data_out_path, 'nucls_processed.csv'))
    pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-b', '--base_path', type=str, help='path to base directory')
    args = argparser.parse_args()
    extract_patches(args.base_path)