1. You should put the path of data csvfile in Line 18, data path in Line 19,
and pretrained FAST_VQA_B_1*4.pth path in Line 50 in options/fast-b_NTIRE_UGC.yml

2. Run this command:
CUDA_VISIBLE_DEVICES=0 python extract_fastvqa_feature.py
--opt options/fast-b_NTIRE_UGC.yml
--save_path "your save path"
