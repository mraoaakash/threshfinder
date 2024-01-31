import os 
import pandas as pd 
import numpy as np 
import os 
import sys


def organize_nucls(nucl_path, out_path):
    image_path = os.path.join(nucl_path, 'rgb')
    mask_path = os.path.join(nucl_path, 'mask')