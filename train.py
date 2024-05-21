# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import random
import torch.nn as nn
import scipy.io as scio

from data_loader import BVQA_VideoDataset_RQ_VQA
from utils import performance_fit
from utils import plcc_loss

import models

from torchvision import transforms
import time

def main(config):

    all_val_SRCC, all_val_KRCC, all_val_PLCC, all_val_RMSE = [], [], [], []
    save_model_name_all = []

    for i in range(config.n_exp):
        config.exp_version = i
        print('%d round training starts here' % i)
        if config.random_seed != 0:
            seed = (i+1) * config.random_seed
        else:
            seed = i * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.model_name == 'RQ_VQA':
            print('The current model is ' + config.model_name)
            model = models.RQ_VQA(config.pretrained_path)


        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)
        

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

        if config.loss_type == 'plcc':
            criterion = plcc_loss

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))


        transformations_train = transforms.Compose([transforms.Resize(config.resize), transforms.RandomCrop(config.crop_size), transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        ## training data
        
        if config.database == 'NTIREVideo':
            datainfo = 'data/train_data.csv'
            videos_dir = '/data/sunwei_data/ntire_video/image_384p'

            feature_dir = '/data/sunwei_data/ntire_video/NTIREVideo_Train_SlowFast_feature/'

            feature_visual_dir = '/data/sunwei_data/ntire_video/qalign_features_frames/'
            feature_LIQE_dir = '/data/sunwei_data/ntire_video/LIQE_feature/'
            feature_FASTVQA_dir = '/data/sunwei_data/ntire_video/FASTVQA/sampled/'
            trainset = BVQA_VideoDataset_RQ_VQA(videos_dir, feature_dir, feature_visual_dir, feature_LIQE_dir, feature_FASTVQA_dir, datainfo, transformations_train, 'NTIREVideo_train', config.crop_size, seed=seed)
            valset = BVQA_VideoDataset_RQ_VQA(videos_dir, feature_dir, feature_visual_dir, feature_LIQE_dir, feature_FASTVQA_dir, datainfo, transformations_test, 'NTIREVideo_val', config.crop_size, seed=seed)



        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
            shuffle=True, num_workers=config.num_workers)



        best_val_criterion = -1  # SROCC min
        best_val = []

        print('Starting training:')

        old_save_name = None
        old_mat_name = None

        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()

            for i, (video, feature_3D, feature_llm, feature_liqe, feature_fastvqa, mos, _) in enumerate(train_loader):
                

                video = video.to(device)
                feature_3D = feature_3D.to(device)
                feature_llm = feature_llm.to(device)
                feature_liqe = feature_liqe.to(device)
                feature_fastvqa = feature_fastvqa.to(device)
                labels = mos.to(device).float()
                

                outputs = model(video, feature_3D, feature_llm, feature_liqe, feature_fastvqa)

                optimizer.zero_grad()
                
                loss = criterion(labels, outputs)

                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                loss.backward()
                
                optimizer.step()

                if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                        (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                            avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))
            

            # Val
            with torch.no_grad():
                model.eval()
                label = np.zeros([len(valset)])
                y_output = np.zeros([len(valset)])

                for i, (video, feature_3D, feature_llm, feature_liqe, feature_fastvqa, mos, _) in enumerate(val_loader):
                    
                    video = video.to(device)
                    feature_3D = feature_3D.to(device)
                    feature_llm = feature_llm.to(device)
                    feature_liqe = feature_liqe.to(device)
                    feature_fastvqa = feature_fastvqa.to(device)

                    label[i] = mos.item()

                    outputs = model(video, feature_3D, feature_llm, feature_liqe, feature_fastvqa)

                    y_output[i] = outputs.item()

                
                val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
                
                print('Epoch {} completed. The result on the val databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                    val_SRCC, val_KRCC, val_PLCC, val_RMSE))
                
                

                    
                if val_SRCC > best_val_criterion:
                    print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = val_SRCC
                    best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]

                    print('Saving model...')
                    if not os.path.exists(config.ckpt_path):
                        os.makedirs(config.ckpt_path)

                    if epoch > 0:
                        if os.path.exists(old_save_name):
                            os.remove(old_save_name)
                        if os.path.exists(old_mat_name):
                            os.remove(old_mat_name)

                    save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                        config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
                            + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, val_SRCC))

                    save_mat_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                        config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
                            + '_epoch_%d_SRCC_%f.mat' % (epoch + 1, val_SRCC))
                    torch.save(model.state_dict(), save_model_name)
                    old_save_name = save_model_name
                    old_mat_name = save_mat_name

            save_model_name_all.append(save_model_name)

            print('Training completed.')
            print('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val[0], best_val[1], best_val[2], best_val[3]))
            print('*************************************************************************************************************************')

        all_val_SRCC.append(best_val[0])
        all_val_KRCC.append(best_val[1])
        all_val_PLCC.append(best_val[2])
        all_val_RMSE.append(best_val[3])
   
    
    save_mat_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
        config.database + '_' + config.loss_type + '.mat')    
    scio.savemat(save_mat_name, {'save_model_name_all':save_model_name_all})
    
    print('*************************************************************************************************************************')
    print('SRCC:')
    print(all_val_SRCC)
    print('KRCC:')
    print(all_val_KRCC)
    print('PLCC:')
    print(all_val_PLCC)
    print('RMSE:')
    print(all_val_RMSE)
    print(
        'The avg results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(all_val_SRCC), np.mean(all_val_KRCC), np.mean(all_val_PLCC), np.mean(all_val_RMSE)))

    print(
        'The std results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(all_val_SRCC), np.std(all_val_KRCC), np.std(all_val_PLCC), np.std(all_val_RMSE)))

    print(
        'The median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_val_SRCC), np.median(all_val_KRCC), np.median(all_val_PLCC), np.median(all_val_RMSE)))
    print('*************************************************************************************************************************')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--print_samples', type=int, default = 0)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--n_exp', type=int)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='plcc')

    
    config = parser.parse_args()


    torch.manual_seed(config.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    main(config)







