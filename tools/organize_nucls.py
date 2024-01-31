import os 
import pandas as pd 
import numpy as np 
import os 
import sys
import argparse
import tqdm


def organize_nucls(nucl_path):
    image_path = os.path.join(nucl_path, 'rgb')
    mask_path = os.path.join(nucl_path, 'csv')

    # get current working directory
    cwd = os.getcwd()
    split_cwd = cwd.split('threshfinder')[0]
    out_path = os.path.join(split_cwd, 'threshfinder', 'data', 'nucls')
    os.makedirs(out_path, exist_ok=True)

    image_list = os.listdir(image_path)

    # remove .DS_Store
    try:
        image_list.remove('.DS_Store')
    except:
        pass

    df = pd.DataFrame(columns=['image_path', 'mask_path'])
    imarr = []
    maskarr = []
    for image in tqdm.tqdm(image_list, total=len(image_list)):
        img_path = os.path.join(image_path, image)
        msk_path = os.path.join(mask_path, image.split('.png')[0] + '.csv')
        # print(mask_path)
        if os.path.isfile(msk_path):
            imarr.append(img_path)
            maskarr.append(msk_path)
        else:
            # print('No mask found for image: {}'.format(image))
            pass
        
    df['image_path'] = imarr
    df['mask_path'] = maskarr
    df.to_csv(os.path.join(out_path, 'nucls.csv'), index=False)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-i','--nucl_path', type=str, help='path to nucl folder')
    args = argparse.parse_args()

    organize_nucls(args.nucl_path)