# RQ-VQA
This is a repository for the models proposed in the paper "Enhancing Blind Video Quality Assessment with Rich Quality-aware Features". [Arxiv Version](https://arxiv.org/abs/2405.08745)

RQ-VQA won **first place ** in [NTIRE 2024 Short-form UGC Video Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/17638).

### Environments
- Base model: timm==0.6.13 (higer version will cause error), pytorch>=1.13 (test on 1.13), torchvision, cv2, pandas
- For FAST-VQA feature extraction: the same requirement in https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
- For LIQE feature extraction: ftfy, regex, tqdm, clip (pip install git+https://github.com/openai/CLIP.git)
- For Q-Align feature extraction: the same requirement in https://github.com/Q-Future/Q-Align

### Dataset
Download the [KVQ dataset](https://drive.google.com/drive/folders/1dkC4NsxMrd6Rxm1IogKe71U8bYy76ojV)

### Train RQ-VQA
- Frame extraction
```
python frame_extraction/extract_frame_NTIREVideo_384p.py --filename_path data/train_data.csv --videos_dir /data/sunwei_data/ntire_video --save_folder /data/sunwei_data/ntire_video/test_image_384p
python frame_extraction/extract_frame_NTIREVideo_original.py --filename_path data/train_data.csv --videos_dir /data/sunwei_data/ntire_video --save_folder /data/sunwei_data/ntire_video/test_image_original # for extracting LIQE and Q-Align features
```

- SlowFast feature extraction
```
CUDA_VISIBLE_DEVICES=0 python -u feature_extraction/extract_SlowFast_feature_VQA.py \
--database NTIREVideoTest \
--resize 224 \
--feature_save_folder  /data/sunwei_data/ntire_video/NTIREVideo_Train_SlowFast_feature/ \
--datainfo_test data/train_data.csv \
--videos_dir /data/sunwei_data/ntire_video
```

- LIQE features extraction

Weight download: https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view?usp=sharing
```
CUDA_VISIBLE_DEVICES=0 python -u feature_extraction/extract_LIQE_feature_KVQ.py --videos_dir_test /data/sunwei_data/ntire_video/test_image_original --feature_save_folder /data/sunwei_data/ntire_video/LIQE_feature/ --datainfo_test data/train_data.csv
```























- FASTVQA features extraction

You should put the path of data csvfile in Line 18 (data/train_data.csv), data path in Line 19, and pretrained FAST_VQA_B_1*4.pth (https://1drv.ms/u/s!AsQt2I-RXJHQjoN0b_-BsMm-VHSGNw?e=PMKNTh) path in Line 50 in options/fast-b_NTIRE_UGC.yml
```
cd features/FastVQA_feature
CUDA_VISIBLE_DEVICES=0 python extract_fastvqa_feature.py \
--opt options/fast-b_NTIRE_UGC.yml \
--save_path /data/sunwei_data/ntire_video/FASTVQA/sampled/
```
- Q-Align features extraction
```
cd feature_extraction/Q-Align
read the readme.txt for feature extraction 
```






To facilitate the reproduction of the experiments, we provide SlowFast, FASTVQA, LIQE and Q-Align features for the [KVQ training, validation and test sets](https://www.dropbox.com/scl/fi/sp80tb9se3jxj8f0cptlx/features.tar?rlkey=ea7n5m4us1064c6wi7gbwgyvd&st=1198mttn&dl=0).

- Train the model

Download the pre-trained [model](https://drive.google.com/file/d/1jgzVV0sil0kGhhHIV0RLr6YoDZNp7LNi/view?usp=sharing) on LSVQ
```
  CUDA_VISIBLE_DEVICES=0,1 python -u train.py \
 --database NTIREVideo \
 --model_name RQ_VQA \
 --pretrained_path /home/sunwei/code/VQA/SimpleVQA/ckpts/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
 --multi_gpu \
 --motion \
 --conv_base_lr 0.00001 \
 --epochs 30 \
 --train_batch_size 6 \
 --print_samples 400 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --loss_type plcc \
 --random_seed 10 \
 --n_exp 10 \
 --resize 384 \
 --crop_size 384 \
 >> logs/train.log
```

For computational efficiency, you can simply train the base model, which does not require extracting FASTVQA, LIQE, and Q-Align features.
```
  CUDA_VISIBLE_DEVICES=0,1 python -u train_base_model.py \
 --database NTIREVideo \
 --model_name RQ_VQA_base_model \
 --pretrained_path /home/sunwei/code/VQA/SimpleVQA/ckpts/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
 --multi_gpu \
 --motion \
 --conv_base_lr 0.00001 \
 --epochs 30 \
 --train_batch_size 6 \
 --print_samples 400 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --loss_type plcc \
 --random_seed 10 \
 --n_exp 10 \
 --resize 384 \
 --crop_size 384 \
 >> logs/train.log
```

### Test RQ-VQA
- Download the [model weighs](https://drive.google.com/file/d/1mcJdgYZPybUvLfWTUtZOhktsSsPlekgv/view?usp=sharing) trained on KVQ.
- Extract video frames, SlowFast features, FASTVQA features, LIQE features, and Q-Align features of KVQ validation and test sets.
- Run the code
```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
--save_file results.csv \
--n_exp 10 \
--resize 384 \
--crop_size 384 \
--pretrained_path  \ # put the folder of trained model here
>> test.log
```


## Citation
**If you find this code is useful for  your research, please cite**:

```latex
@inproceedings{sun2024Enhancing,
  title={Enhancing Blind Video Quality Assessment with Rich Quality-aware Features},
  author={Sun, Wei and Wu, Haoning and Zhang, Zicheng and Jia, Jun and Zhang, Zhichao and Cao, Linhan and Chen, Qiubo and Min, Xiongkuo and Lin, Weisi and Zhai Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2024}
}
```