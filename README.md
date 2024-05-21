# RQ-VQA
This is a repository for the models proposed in the paper "Enhancing Blind Video Quality Assessment with Rich Quality-aware Features ".

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
````

- Train the model
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