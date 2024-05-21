# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch

import sys
sys.path.append('./')


from torchvision import transforms
import torch.nn as nn
import pandas as pd
from PIL import Image




import torch.nn as nn
import random

from LIQE import LIQE
from torchvision.transforms import ToTensor

seed = 20200626
num_patch = 15

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class LIQE_feature(torch.nn.Module):
    def __init__(self):
        super(LIQE_feature, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = 'LIQE.pt'
        model_LIQE = LIQE(model_path, device)
        self.feature_extraction = model_LIQE

        

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            
        return x





def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_LIQE = LIQE_feature()
    model_LIQE = model_LIQE.to(device)



    transformations_test = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])


    videos_dir_test = config.videos_dir_test
    datainfo_test = config.datainfo_test

    feature_save_folder = config.feature_save_folder


    with torch.no_grad():

        column_names = ['filename','score']
        dataInfo = pd.read_csv(datainfo_test, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        video = dataInfo['filename'].tolist()  
        n = len(video)              

        for i in range(n):
            video_name_str = video[i][:-4]
            print(video[i])
            if not os.path.exists(feature_save_folder + video_name_str):
                os.makedirs(feature_save_folder + video_name_str)
            

            for j in range(8):
                imge_name = os.path.join(videos_dir_test, video_name_str, '{:03d}'.format(j) + '.png')
                print(imge_name)
                image = Image.open(imge_name)
                image = ToTensor()(image).unsqueeze(0)
                feature = model_LIQE(image)

                np.save(feature_save_folder + video_name_str + '/' + 'feature_' + str(j) + '_LIQE_feature', feature.to('cpu').numpy())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--videos_dir_test', type=str)
    parser.add_argument('--feature_save_folder', type=str)
    parser.add_argument('--datainfo_test', type=str)

    config = parser.parse_args()
    main(config)