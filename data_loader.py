import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import cv2
import random


class BVQA_VideoDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, data_dir_LLM, data_dir_LIQE, data_dir_FASTVQA, filename_path, transform, crop_size):
        super(BVQA_VideoDataset, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        video = dataInfo['filename'].tolist()
        score = None
        n = len(video)
        video_names = []
        for i in range(n):
            video_names.append(video[i])
        self.video_names = video_names
        self.score = score

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.data_dir_LLM = data_dir_LLM
        self.data_dir_LIQE = data_dir_LIQE
        self.data_dir_FASTVQA = data_dir_FASTVQA
        self.transform = transform
        self.length = len(self.video_names)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]        

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       


        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        

        # read 3D features
        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D
        

        # read LLM features
        feature_folder_name = os.path.join(self.data_dir_LLM, video_name_str)
        transformed_LLM_feature = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            feature_LLM = np.load(os.path.join(feature_folder_name, '{:03d}'.format(i) + '.npy'))
            feature_LLM = torch.from_numpy(feature_LLM)
            feature_LLM = feature_LLM.squeeze()
            transformed_LLM_feature[i] = feature_LLM

        # read LIQE features
        feature_folder_name = os.path.join(self.data_dir_LIQE, video_name_str)
        transformed_LIQE_feature = torch.zeros([video_length_read, 495])
        for i in range(video_length_read):
            feature_LIQE = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_LIQE_feature.npy'))
            feature_LIQE = torch.from_numpy(feature_LIQE)
            feature_LIQE = feature_LIQE.squeeze()
            transformed_LIQE_feature[i] = feature_LIQE


        # read FAST-VQA features
        feature_folder_name = os.path.join(self.data_dir_FASTVQA, video_name_str+'.npy')
        feature_FASTVQA = np.load(feature_folder_name)
        feature_FASTVQA = torch.from_numpy(feature_FASTVQA)
        transformed_FASTVQA_feature = torch.zeros([video_length_read, 768])
        for i in range(video_length_read):
            transformed_FASTVQA_feature[i] = feature_FASTVQA


       



        return transformed_video, transformed_feature, transformed_LLM_feature, transformed_LIQE_feature, transformed_FASTVQA_feature, video_name





class BVQA_VideoDataset_RQ_VQA(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, data_dir_LLM, data_dir_LIQE, data_dir_FASTVQA, filename_path, transform, database_name, crop_size, seed=0):
        super(BVQA_VideoDataset_RQ_VQA, self).__init__()


        column_names = ['filename','score']
        dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        video = dataInfo['filename'].tolist()
        score = dataInfo['score'].tolist()
        n = len(video)
        video_names = []
        for i in range(n):
            video_names.append(video[i])
        if database_name == 'NTIREVideo':
            self.video_names = video_names
            self.score = score
        else:
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            random.seed(seed)
            np.random.seed(seed)
            length = 418
            index_rd = np.random.permutation(length)
            train_index_ref = index_rd[0:int(length * 0.8)]
            # do not use the validation set
            val_index_ref = index_rd[int(length * 0.8):]
            train_index = []
            for i_ref in train_index_ref:
                for i_dis in range(7):
                    train_index.append(i_ref*7+i_dis)
            val_index = []
            for i_ref in val_index_ref:
                for i_dis in range(7):
                    val_index.append(i_ref*7+i_dis)

            print('train_index')
            print(train_index)
            print('val_index')
            print(val_index)
            if database_name == 'NTIREVideo_train':
                self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                self.score = dataInfo.iloc[train_index]['MOS'].tolist()
            elif database_name == 'NTIREVideo_val':
                self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                self.score = dataInfo.iloc[val_index]['MOS'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.data_dir_LLM = data_dir_LLM
        self.data_dir_LIQE = data_dir_LIQE
        self.data_dir_FASTVQA = data_dir_FASTVQA
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]        
        

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       


        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        

        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D
        

        feature_folder_name = os.path.join(self.data_dir_LLM, video_name_str)
        transformed_LLM_feature = torch.zeros([video_length_read, 4096])
        for i in range(video_length_read):
            feature_LLM = np.load(os.path.join(feature_folder_name, '{:03d}'.format(i) + '.npy'))
            feature_LLM = torch.from_numpy(feature_LLM)
            feature_LLM = feature_LLM.squeeze()
            transformed_LLM_feature[i] = feature_LLM

        feature_folder_name = os.path.join(self.data_dir_LIQE, video_name_str)
        transformed_LIQE_feature = torch.zeros([video_length_read, 495])
        for i in range(video_length_read):
            feature_LIQE = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_LIQE_feature.npy'))
            feature_LIQE = torch.from_numpy(feature_LIQE)
            feature_LIQE = feature_LIQE.squeeze()
            transformed_LIQE_feature[i] = feature_LIQE


        feature_folder_name = os.path.join(self.data_dir_FASTVQA, video_name_str+'.npy')
        feature_FASTVQA = np.load(feature_folder_name)
        feature_FASTVQA = torch.from_numpy(feature_FASTVQA)
        transformed_FASTVQA_feature = torch.zeros([video_length_read, 768])
        for i in range(video_length_read):
            transformed_FASTVQA_feature[i] = feature_FASTVQA


        return transformed_video, transformed_feature, transformed_LLM_feature, transformed_LIQE_feature, transformed_FASTVQA_feature, video_score, video_name



class BVQA_VideoDataset_RQ_VQA_base_model(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, seed=0):
        super(BVQA_VideoDataset_RQ_VQA_base_model, self).__init__()


        column_names = ['filename','score']
        dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        video = dataInfo['filename'].tolist()
        score = dataInfo['score'].tolist()
        n = len(video)
        video_names = []
        for i in range(n):
            video_names.append(video[i])
        if database_name == 'NTIREVideo':
            self.video_names = video_names
            self.score = score
        else:
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            random.seed(seed)
            np.random.seed(seed)
            length = 418
            index_rd = np.random.permutation(length)
            train_index_ref = index_rd[0:int(length * 0.8)]
            # do not use the validation set
            val_index_ref = index_rd[int(length * 0.8):]
            train_index = []
            for i_ref in train_index_ref:
                for i_dis in range(7):
                    train_index.append(i_ref*7+i_dis)
            val_index = []
            for i_ref in val_index_ref:
                for i_dis in range(7):
                    val_index.append(i_ref*7+i_dis)

            print('train_index')
            print(train_index)
            print('val_index')
            print(val_index)
            if database_name == 'NTIREVideo_train':
                self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                self.score = dataInfo.iloc[train_index]['MOS'].tolist()
            elif database_name == 'NTIREVideo_val':
                self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                self.score = dataInfo.iloc[val_index]['MOS'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]        
        

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       


        video_length_read = 8


        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        

        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D


        return transformed_video, transformed_feature, video_score, video_name


class BVQA_VideoDataset_SlowFast(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize):
        super(BVQA_VideoDataset_SlowFast, self).__init__()


        dataInfo = pd.read_csv(filename_path)
        video = dataInfo['filename'].tolist()

        video_names = []
        for video_i in video:
            video_names.append(video_i)
        self.video_names = video_names



        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]

        filename=os.path.join(self.videos_dir, 'test', video_name) # for test
        print(filename)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
           video_clip = 10
        else:
            video_clip = int(video_length/video_frame_rate)


        video_clip_min = 8

        video_length_clip = 32             

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name_str