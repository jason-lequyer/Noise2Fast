import os
from tifffile import imread, imwrite
import numpy as np
import sys
from pathlib import Path

if __name__ == "__main__":
    
    folder = sys.argv[1]
    sigma = np.int(sys.argv[2]) 

    mask_file_list = [f for f in os.listdir(folder+'/')]
    outfolder = folder+'_gaussian'+str(sigma)
    Path(outfolder).mkdir(exist_ok=True)
    for v in range(len(mask_file_list)):
        file_name =  mask_file_list[v]
        img = imread(folder + '/' + file_name)
        noise = np.random.normal(0, sigma, img.shape)
        img = img + noise
        imwrite(outfolder + '/' + file_name, img)