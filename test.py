# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import torch

import random

from data_loader import BVQA_VideoDataset

import models

from torchvision import transforms


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_file = ['ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_9_SRCC_0.885692.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_22_SRCC_0.894115.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_25_SRCC_0.913571.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_16_SRCC_0.901800.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_6_SRCC_0.905095.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v0_epoch_4_SRCC_0.905999.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v1_epoch_19_SRCC_0.923127.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v2_epoch_21_SRCC_0.924423.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v3_epoch_8_SRCC_0.896798.pth',\
                  'ckpts/Swin_b_384_in22k_SlowFast_Fast_LLM_LIQE_FASTVQA_BoT_NTIREVideo_plcc_NR_v4_epoch_14_SRCC_0.904949.pth']


    
    y_output_all = []
    for i_exp in range(config.n_exp):



        model = models.RQ_VQA_base_model(None)

        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

        # load model
        print('load the model ' + os.path.join(config.pretrained_path, model_file[i_exp]))
        model.load_state_dict(torch.load(os.path.join(config.pretrained_path, model_file[i_exp])))
        
        transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
        datainfo = 'data/test_data.csv'
        videos_dir = 'test_image_384p/' # the path of video frames
        feature_3D_dir = 'features/SlowFast/'

        feature_LLM = 'features/QAlign/'
        feature_LIQE_dir = 'features/LIQE/'
        feature_FASTVQA_dir = 'featurs/FASTVQA/'
        testset = BVQA_VideoDataset(videos_dir, feature_3D_dir, feature_LLM, \
                            feature_LIQE_dir, feature_FASTVQA_dir, datainfo, transformations_test, config.crop_size)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
            shuffle=False, num_workers=config.num_workers)


        # # Test
        with torch.no_grad():
            model.eval()
            y_output = []
            video_names = []

            for i, (video, feature_3D, feature_LLM, feature_LIQE, feature_FASTVQA, video_name) in enumerate(test_loader):
                
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                feature_LLM = feature_LLM.to(device)
                feature_LIQE = feature_LIQE.to(device)
                feature_FASTVQA = feature_FASTVQA.to(device)

                outputs = model(video, feature_3D, feature_LLM, feature_LIQE, feature_FASTVQA)
                y_output.append(outputs.item())
                video_names.append(video_name[0])
                print(video_name[0])

            
            y_output = np.array(y_output)
        
        
        
        y_output_all.append(y_output)
    
    y_output_ensemble = np.zeros(y_output.shape)

    for i in range(len(y_output_all)):
        y_output_ensemble += y_output_all[i]
    
    y_output_ensemble /= len(y_output_all)

        

    # overall 
    test_data = []
    for i in range(len(video_names)):
        test_data.append([video_names[i], y_output_ensemble[i]])

    column_names = ['filename','score']
    test_data_df = pd.DataFrame(test_data, columns = column_names)
    test_data_df.to_csv(config.save_file, index = False)


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_exp', type=int)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--random_seed', type=int, default=8)
    
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    main(config)