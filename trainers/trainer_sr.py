# trainer_sr.py
# Dual Super-Resolution Learning
# 단일 Encoder & 개별 Decoder로 SS & SR 동시에 시행하는 구조
# [memo] --------------
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# AMP(Automatic Mixed Precision) 사용됨

# now data-loader supprt multi workers
"""
# [기본 라이브러리]----------------------
import os
import numpy as np
import random
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time

# [py 파일]--------------------------
from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *

from mps.mp_sr_plt_saver import plts_saver

# https://github.com/KyungBong-Ryu/Codes_implementation/blob/main/BasicSR_NIQE.py
# from DLCs.BasicSR_NIQE import calc_niqe_with_pil
from DLCs.BasicSR_NIQE import calc_niqe as _calc_niqe

#<<< @@@ trainer_sr


def trainer_sr(**kargs):
    '''#========================================#
    trainer_sr  (# <patch를 통해 학습 & RGB 이미지를 생성하는 모델>#
                 # 코드 실행 장소 (colab 여부 확인용, colab == -1)
                 RUN_WHERE = 
                 # 버퍼 크기 (int) -> 버퍼 사용시, valid 기준으로 PSNR Best 를 확인함에 주의 (사용안함 = -1)
                ,BUFFER_SIZE = 
                 # 초기화 기록 dict 이어받기
                ,dict_log_init = 
                 # 랜덤 시드 고정
                ,HP_SEED = HP_SEED
                
                
                 # 학습 관련 기본 정보(epoch 수, batch 크기(train은 생성할 patch 수), 학습 시 dataset 루프 횟수)
                ,HP_EPOCH = HP_EPOCH
                ,HP_BATCH_TRAIN = HP_BATCH_TRAIN_SR
                ,HP_DATASET_LOOP = HP_DATASET_LOOP_SR
                ,HP_BATCH_VAL = HP_BATCH_VAL
                ,HP_BATCH_TEST = HP_BATCH_TEST
                
                
                 # 데이터 입출력 경로, 폴더명
                ,PATH_BASE_IN = PATH_BASE_IN
                ,NAME_FOLDER_TRAIN = NAME_FOLDER_TRAIN
                ,NAME_FOLDER_VAL = NAME_FOLDER_VAL
                ,NAME_FOLDER_TEST = NAME_FOLDER_TEST
                ,NAME_FOLDER_IMAGES = NAME_FOLDER_IMAGES
                ,NAME_FOLDER_LABELS = NAME_FOLDER_LABELS
                
                 # (선택) degraded image 불러올 경로
                ,PATH_BASE_IN_SUB = PATH_BASE_IN_SUB
                
                ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
                ,PATH_OUT_MODEL = PATH_OUT_MODEL
                ,PATH_OUT_LOG = PATH_OUT_LOG
                
                
                 # 데이터(이미지) 입출력 크기 (원본 이미지, 모델 입력 이미지), 이미지 채널 수(이미지)
                ,HP_ORIGIN_IMG_W = HP_ORIGIN_IMG_W
                ,HP_ORIGIN_IMG_H = HP_ORIGIN_IMG_H
                ,HP_MODEL_IMG_W = HP_MODEL_SR_IMG_W
                ,HP_MODEL_IMG_H = HP_MODEL_SR_IMG_H
                ,HP_CHANNEL_RGB = HP_CHANNEL_RGB
                
                 #Patch 생성 관련
                ,is_use_patch = True
                ,HP_PATCH_STRIDES = HP_PATCH_STRIDES_SR
                ,HP_PATCH_CROP_INIT_COOR_RANGE = HP_PATCH_CROP_INIT_COOR_RANGE_SR
                
                 # 모델 이름 -> 모델에 따라 예측결과 형태가 다르기에 모델입출력물을 조정하는 역할
                 # 지원 리스트 = "MPRNet"
                ,model_name = model_sr_name
                
                 # model, optimizer, scheduler, loss
                ,model = model_sr
                ,optimizer = optimizer
                ,scheduler = scheduler
                ,criterion = criterion_sr
                 # 스케쥴러 업데이트 간격("epoch" 또는 "batch")
                ,HP_SCHEDULER_UPDATE_INTERVAL = HP_SCHEDULER_UPDATE_INTERVAL_SR
                
                
                 # DataAugm- 관련 (colorJitter 포함)
                ,HP_AUGM_RANGE_CROP_INIT = HP_AUGM_RANGE_CROP_INIT
                ,HP_AUGM_ROTATION_MAX = HP_AUGM_ROTATION_MAX
                ,HP_AUGM_PROB_FLIP = HP_AUGM_PROB_FLIP
                ,HP_AUGM_PROB_CROP = HP_AUGM_PROB_CROP
                ,HP_AUGM_PROB_ROTATE = HP_AUGM_PROB_ROTATE
                ,HP_CJ_BRIGHTNESS = HP_CJ_BRIGHTNESS
                ,HP_CJ_CONTRAST = HP_CJ_CONTRAST
                ,HP_CJ_SATURATION = HP_CJ_SATURATION
                ,HP_CJ_HUE = HP_CJ_HUE
                
                
                 # 이미지 -> 텐서 시 norm 관련 (정규화 시행여부, 평균, 표준편차)
                ,is_norm_in_transform_to_tensor = is_norm_in_transform_to_tensor
                ,HP_TS_NORM_MEAN = HP_TS_NORM_MEAN_SR
                ,HP_TS_NORM_STD = HP_TS_NORM_STD_SR
                
                
                 # Degradation 관련 설정값
                ,HP_DG_CSV_NAME = HP_DG_CSV_NAME
                ,HP_DG_SCALE_FACTOR = HP_DG_SCALE_FACTOR
                ,HP_DG_RESIZE_OPTION = HP_DG_RESIZE_OPTION
                ,HP_DG_RANGE_NOISE_SIGMA = HP_DG_RANGE_NOISE_SIGMA
                ,HP_DG_NOISE_GRAY_PROB = HP_DG_NOISE_GRAY_PROB
                )
    
    '''#========================================#
    
    
    # [최우선 초기화요소 시행]------------------------
    # colab 여부 판별용 (colab == -1)
    try:
        RUN_WHERE = kargs['RUN_WHERE']
    except:
        RUN_WHERE = 1
    
    try:
        BUFFER_SIZE = kargs['BUFFER_SIZE']
        if BUFFER_SIZE < 1:
            BUFFER_SIZE = -1
    except:
        BUFFER_SIZE = -1    # minus value = BUFFER not used
    
    if BUFFER_SIZE == -1:
        print("Buffer not used")
    else:
        print("BUFFER_SIZE set to", BUFFER_SIZE)
    
    # log dict 이어받기
    dict_log_init = kargs['dict_log_init']
    
    # 사용 decive 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 랜덤 시드(seed) 적용
    HP_SEED = kargs['HP_SEED']
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    # pytorch 랜덤시드 고정 (CPU)
    torch.manual_seed(HP_SEED)
    
    
    update_dict_v2("", ""
                  ,"", "랜덤 시드값 (random numpy pytorch)"
                  ,"", "HP_SEED: " + str(HP_SEED)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    if device == 'cuda':
        # pytorch 랜덤시드 고정 (GPU & multi-GPU)
        torch.cuda.manual_seed(HP_SEED)
        torch.cuda.manual_seed_all(HP_SEED)
    
    # epoch 수
    HP_EPOCH = kargs['HP_EPOCH']
    # batch 크기 & (train) 데이터셋 루프 횟수
    HP_BATCH_TRAIN = kargs['HP_BATCH_TRAIN']
    HP_DATASET_LOOP = kargs['HP_DATASET_LOOP']
    HP_BATCH_VAL = kargs['HP_BATCH_VAL']
    HP_BATCH_TEST = kargs['HP_BATCH_TEST']
    
    update_dict_v2("", ""
                  ,"", "최대 epoch 설정: " + str(HP_EPOCH)
                  ,"", "batch 크기"
                  ,"", "HP_BATCH_TRAIN: " + str(HP_BATCH_TRAIN)
                  ,"", "학습 시 데이터셋 반복횟수"
                  ,"", "HP_DATASET_LOOP: " + str(HP_DATASET_LOOP)
                  ,"", "그래디언트 축적(Gradient Accumulation) 사용 안함"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [입출력 Data 관련]-----------------------------
    # 경로: 입력
    PATH_BASE_IN = kargs['PATH_BASE_IN']
    NAME_FOLDER_TRAIN = kargs['NAME_FOLDER_TRAIN']
    NAME_FOLDER_VAL = kargs['NAME_FOLDER_VAL']
    NAME_FOLDER_TEST = kargs['NAME_FOLDER_TEST']
    NAME_FOLDER_IMAGES = kargs['NAME_FOLDER_IMAGES']
    NAME_FOLDER_LABELS = kargs['NAME_FOLDER_LABELS']
    
    # 경로: 출력
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    PATH_OUT_MODEL = kargs['PATH_OUT_MODEL']
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    
    # 원본 이미지 크기
    HP_ORIGIN_IMG_W = kargs['HP_ORIGIN_IMG_W']
    HP_ORIGIN_IMG_H = kargs['HP_ORIGIN_IMG_H']
    # 이미지 모델입력 크기 (train & val & test)
    HP_MODEL_IMG_W = kargs['HP_MODEL_IMG_W']
    HP_MODEL_IMG_H = kargs['HP_MODEL_IMG_H']
    # 이미지 채널 수
    HP_CHANNEL_RGB = kargs['HP_CHANNEL_RGB']
    
    #Patch 생성 관련
    is_use_patch = kargs['is_use_patch']
    if is_use_patch:
        HP_PATCH_STRIDES = kargs['HP_PATCH_STRIDES']
        HP_PATCH_CROP_INIT_COOR_RANGE = kargs['HP_PATCH_CROP_INIT_COOR_RANGE']
        update_dict_v2("", ""
                      ,"", "원본 Dataset 이미지 크기"
                      ,"", "HP_ORIGIN_IMG_(W H): (" + str(HP_ORIGIN_IMG_W) + " " + str(HP_ORIGIN_IMG_H) + ")"
                      ,"", "모델 입출력 이미지 크기 (train val test)"
                      ,"", "HP_MODEL_IMG_(W H): (" + str(HP_MODEL_IMG_W) + " " + str(HP_MODEL_IMG_H) + ")"
                      ,"", "이미지 채널 수 (이미지): " + str(HP_CHANNEL_RGB)
                      ,"", "Patch 생성함"
                      ,"", "stride (w and h): " + str(HP_PATCH_STRIDES[0]) + " and " + str(HP_PATCH_STRIDES[1])
                      ,"", "crop 시작: " + str(HP_PATCH_CROP_INIT_COOR_RANGE[0]) + " ~ " + str(HP_PATCH_CROP_INIT_COOR_RANGE[1])
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
    else:
        HP_PATCH_STRIDES = (1,1)
        HP_PATCH_CROP_INIT_COOR_RANGE = (0,0)
    
        update_dict_v2("", ""
                      ,"", "원본 Dataset 이미지 크기"
                      ,"", "HP_ORIGIN_IMG_(W H): (" + str(HP_ORIGIN_IMG_W) + " " + str(HP_ORIGIN_IMG_H) + ")"
                      ,"", "모델 입출력 이미지 크기 (train val test)"
                      ,"", "HP_MODEL_IMG_(W H): (" + str(HP_MODEL_IMG_W) + " " + str(HP_MODEL_IMG_H) + ")"
                      ,"", "이미지 채널 수 (이미지): " + str(HP_CHANNEL_RGB)
                      ,"", "Patch 생성 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    # [model, 모델 추가정보, optimizer, scheduler, loss]--------------------
    #<<< 
    #   현재 지원되는 모델 리스트
    #   1. MPRNet
    #   2. ESRT
    #
    model_name = kargs['model_name']
    #>>>
    
    model = kargs['model']
    
    update_dict_v2("", ""
                  ,"", "모델 종류: " + str(model_name)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    optimizer = kargs['optimizer']
    scheduler = kargs['scheduler']
    HP_SCHEDULER_UPDATE_INTERVAL = kargs['HP_SCHEDULER_UPDATE_INTERVAL']
    # loss
    criterion = kargs['criterion']
    
    # [Automatic Mixed Precision 선언] ---
    amp_scaler = torch.cuda.amp.GradScaler(enabled = True)
    update_dict_v2("", ""
                  ,"", "Automatic Mixed Precision 사용됨"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [data augmentation 관련]--------------------
    
    HP_AUGM_RANGE_CROP_INIT = kargs['HP_AUGM_RANGE_CROP_INIT']
    HP_AUGM_ROTATION_MAX = kargs['HP_AUGM_ROTATION_MAX']
    HP_AUGM_PROB_FLIP = kargs['HP_AUGM_PROB_FLIP']
    HP_AUGM_PROB_CROP = kargs['HP_AUGM_PROB_CROP']
    HP_AUGM_PROB_ROTATE  = kargs['HP_AUGM_PROB_ROTATE']
    # colorJitter 관련
    # https://pytorch.org/vision/master/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
    HP_CJ_BRIGHTNESS = kargs['HP_CJ_BRIGHTNESS']
    HP_CJ_CONTRAST   = kargs['HP_CJ_CONTRAST']
    HP_CJ_SATURATION = kargs['HP_CJ_SATURATION']
    HP_CJ_HUE        = kargs['HP_CJ_HUE']
    
    transform_cj = transforms.ColorJitter(brightness = HP_CJ_BRIGHTNESS
                                         ,contrast   = HP_CJ_CONTRAST
                                         ,saturation = HP_CJ_SATURATION
                                         ,hue        = HP_CJ_HUE
                                         )
    
    update_dict_v2("", ""
                  ,"", "ColorJitter 설정"
                  ,"", "brightness: ( " + " ".join([str(t_element) for t_element in HP_CJ_BRIGHTNESS]) +" )"
                  ,"", "contrast:   ( " + " ".join([str(t_element) for t_element in HP_CJ_CONTRAST])   +" )"
                  ,"", "saturation: ( " + " ".join([str(t_element) for t_element in HP_CJ_SATURATION]) +" )"
                  ,"", "hue:        ( " + " ".join([str(t_element) for t_element in HP_CJ_HUE])        +" )"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    
    # [이미지 변수 -> 텐서 변수 변환]-------------------
    # 정규화 여부
    is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
    
    if is_norm_in_transform_to_tensor:
        # 평균
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN']
        # 표준편차
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD']
        # 입력 이미지 텐서 변환 후 정규화 시행
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  transforms.ToTensor()
                                                  # 평균, 표준편차를 활용해 정규화
                                                 ,transforms.Normalize(mean = HP_TS_NORM_MEAN, std = HP_TS_NORM_STD)
                                                 ,
                                                 ])
        
        # 역정규화 변환
        transform_ts_inv_norm = transforms.Compose([# 평균, 표준편차를 역으로 활용해 역정규화
                                                    transforms.Normalize(mean = [ 0., 0., 0. ]
                                                                        ,std = [ 1/HP_TS_NORM_STD[0], 1/HP_TS_NORM_STD[1], 1/HP_TS_NORM_STD[2] ])
                                                     
                                                   ,transforms.Normalize(mean = [ -HP_TS_NORM_MEAN[0], -HP_TS_NORM_MEAN[1], -HP_TS_NORM_MEAN[2] ]
                                                                        ,std = [ 1., 1., 1. ])
                                                                        
                                                   ,
                                                   ])
        
        update_dict_v2("", ""
                      ,"", "입력 이미지(in_x) 정규화 시행됨"
                      ,"", "mean=[ " + str(HP_TS_NORM_MEAN[0]) + " " + str(HP_TS_NORM_MEAN[1]) + " "+ str(HP_TS_NORM_MEAN[2]) + " ]"
                      ,"", "std=[ " + str(HP_TS_NORM_STD[0]) + " " + str(HP_TS_NORM_STD[1]) + " "+ str(HP_TS_NORM_STD[2]) + " ]"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        # 정규화 없이 이미지를 텐서형으로 변환
        transform_to_ts_img = transforms.Compose([# PIL 이미지 or npArray -> pytorch 텐서
                                                  # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
                                                  # 일반적인 경우 (PIL mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1 또는 numpy.ndarray), 
                                                  # (H x W x C) in the range [0, 255] 입력 데이터를
                                                  # (C x H x W) in the range [0.0, 1.0] 출력 데이터로 변환함 (scaled)
                                                  transforms.ToTensor()
                                                 ])
        
        update_dict_v2("", ""
                      ,"", "입력 이미지(in_x) 정규화 시행 안함"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    # [Degradation 관련]-------------------------------
    # (bool) 학습 & 평가 시 Degradaded Input 사용 여부
    option_apply_degradation = True
    
    if option_apply_degradation:
        # "Train & Test 과정에 Degradation 시행 됨"
        
        try:
            PATH_BASE_IN_SUB = kargs['PATH_BASE_IN_SUB']
            HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
            if not PATH_BASE_IN_SUB[-1] == "/":
                PATH_BASE_IN_SUB += "/"
            HP_DG_CSV_PATH = PATH_BASE_IN_SUB + HP_DG_CSV_NAME
            
            dict_loaded_pils = load_pils_2_dict(# 경로 내 pil 이미지를 전부 불러와서 dict 형으로 묶어버림
                                                # (str) 파일 경로
                                                in_path = PATH_BASE_IN_SUB
                                                # (선택, str) 파일 경로 - 하위폴더명
                                               ,in_path_sub = NAME_FOLDER_IMAGES
                                               )
            print("Pre-Degraded images loaded from:", PATH_BASE_IN_SUB + NAME_FOLDER_IMAGES)
            
            dict_dg_csv = csv_2_dict(path_csv = PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
            print("Pre-Degrade option csv re-loaded from:", PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
            
            flag_pre_degraded_images_loaded = True
            tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러왔습니다."
        except:
            print("(exc) Pre-Degraded images load FAIL")
            flag_pre_degraded_images_loaded = False
            tmp_log_pre_degraded_images_load = "Degraded 이미지를 불러오지 않습니다."
        
        update_dict_v2("", ""
                      ,"", "Degraded 이미지 옵션: " + tmp_log_pre_degraded_images_load
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        # 고정옵션 dict
        #HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
        #HP_DG_CSV_PATH = PATH_BASE_IN_SUB + HP_DG_CSV_NAME
        #dict_dg_csv = csv_2_dict(path_csv = HP_DG_CSV_PATH)
        
        # scale_factor 고정값
        HP_DG_SCALE_FACTOR = kargs['HP_DG_SCALE_FACTOR']
        # resize (downscale) 옵션
        HP_DG_RESIZE_OPTION = kargs['HP_DG_RESIZE_OPTION']
        
        # Gaussian 노이즈 시그마 범위
        HP_DG_RANGE_NOISE_SIGMA = kargs['HP_DG_RANGE_NOISE_SIGMA']
        # Gray 노이즈 확률 (%)
        HP_DG_NOISE_GRAY_PROB = kargs['HP_DG_NOISE_GRAY_PROB']
        
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Test 과정에 Degradation 시행 됨"
                      ,"", "DG 지정값 파일 경로: " + HP_DG_CSV_PATH
                      ,"", "Scale Factor 고정값 = x" + str(HP_DG_SCALE_FACTOR)
                      ,"", "Resize 옵션 = " + HP_DG_RESIZE_OPTION
                      ,"", "Gaussian 노이즈 시그마 범위 = [ " + str(HP_DG_RANGE_NOISE_SIGMA[0]) + " " + str(HP_DG_RANGE_NOISE_SIGMA[-1]) + " ]"
                      ,"", "노이즈 종류 (Color or Gray 중 Gray 노이즈 확률 = " + str(HP_DG_NOISE_GRAY_PROB)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
        
        
    else:
        # "Train & Test 과정에 Degradation 시행 안됨"
        update_dict_v2("", ""
                      ,"", "Degradation 관련"
                      ,"", "시행여부: " + "Train & Test 과정에 Degradation 시행 안됨"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    
    
    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = "log_init.csv"
                 ,in_dict = dict_log_init
                 )
    
    
    # [data & model load]--------------------------
    
    dataset_train = Custom_Dataset_V3(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr', 'info_augm'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'train'
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TRAIN
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = True
                                      # below options can be skipped when above option is False
                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                     
                                      #--- options for HR label
                                     ,is_return_label               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = True
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                      # sub-option a : don't use with sub-option b
                                      #,in_path_dlc                 = not used
                                      #,in_name_dlc_csv             = not used
                                      # sub-option b : don't use with sub-option a
                                     ,opt_dg_blur                   = "Gaussian"
                                     ,opt_dg_interpolation          = HP_DG_RESIZE_OPTION
                                     ,opt_dg_noise                  = "Gaussian"
                                     ,opt_dg_noise_sigma_range      = HP_DG_RANGE_NOISE_SIGMA
                                     ,opt_dg_noise_gray_prob        = HP_DG_NOISE_GRAY_PROB
                                     
                                      #--- increase dataset length
                                     ,in_dataset_loop               = HP_DATASET_LOOP                               #@@@ check required
                                     
                                      #--- options for generate patch
                                     ,is_patch                      = is_use_patch
                                     ,patch_stride                  = HP_PATCH_STRIDES
                                     ,patch_crop_init_range         = HP_PATCH_CROP_INIT_COOR_RANGE
                                     
                                      #--- options for generate tensor
                                     ,model_input_size              = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)              #@@@ check required
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    dataset_val   = Custom_Dataset_V3(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = ' val '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_VAL
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR label
                                     ,is_return_label               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = True
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                      # sub-option a : don't use with sub-option b
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate patch
                                     ,is_patch                      = is_use_patch
                                     
                                      #--- options for generate tensor
                                     ,model_input_size              = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)              #@@@ check required
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    
    dataset_test  = Custom_Dataset_V3(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'test '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TEST
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                     
                                      #--- options for train 
                                     ,is_train                      = False
                                     
                                      #--- options for HR label
                                     ,is_return_label               = False
                                     
                                      #--- options for LR image
                                     ,is_return_image_lr            = True
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                      # sub-option a : don't use with sub-option b
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME
                                     
                                      #--- options for generate patch
                                     ,is_patch                      = False
                                     
                                      #--- options for generate tensor
                                     ,model_input_size              = (HP_ORIGIN_IMG_W//HP_DG_SCALE_FACTOR
                                                                      ,HP_ORIGIN_IMG_H//HP_DG_SCALE_FACTOR
                                                                      )                                             #@@@ check required
                                     ,transform_img                 = transform_to_ts_img                           #@@@ check required
                                     )
    
    
    #https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    
    dataloader_train = torch.utils.data.DataLoader(dataset     = dataset_train
                                                  ,batch_size  = HP_BATCH_TRAIN
                                                  ,shuffle     = True
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )
    
    dataloader_val   = torch.utils.data.DataLoader(dataset     = dataset_val
                                                  ,batch_size  = HP_BATCH_VAL
                                                  ,shuffle     = False
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  )
    
    dataloader_test  = torch.utils.data.DataLoader(dataset     = dataset_test
                                                  ,batch_size  = HP_BATCH_TEST
                                                  ,shuffle     = False
                                                  ,num_workers = 0
                                                  ,prefetch_factor = 2
                                                  )
    
    
    # [Train & Val & Test]-----------------------------
    print("pause before init trainer")
    time.sleep(3)
        
    # 1 epoch 마다 시행할 mode list
    list_mode = ["train", "val", "test"]
    
    #<<< new_record_system
    # 학습 전체 기록
    d_log_total_train = {}
    d_log_total_val   = {}
    d_log_total_test  = {}
    
    # total log dict의 dict
    d_d_log_total = {list_mode[0]: d_log_total_train
                    ,list_mode[1]: d_log_total_val
                    ,list_mode[2]: d_log_total_test
                    }
    
    for i_key in list_mode:
        tmp_str_front = "loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),NIQE_(" + i_key + ")"
        
        update_dict_v2("epoch", tmp_str_front
                      ,in_dict_dict = d_d_log_total
                      ,in_dict_key = i_key
                      ,in_print_head = "d_log_total_" + i_key
                      )
    
    #at .update_epoch(), set is_print_sub = True to see epoch info
    
    rb_train_loss = RecordBox(name = "train_loss", print_interval = 10, is_print = False)
    rb_train_psnr = RecordBox(name = "train_psnr", print_interval = 10, is_print = False)
    rb_train_ssim = RecordBox(name = "train_ssim", print_interval = 10, is_print = False)
    rb_train_niqe = RecordBox(name = "train_niqe", print_interval = 10, is_print = False)
    
    rb_val_loss = RecordBox(name = "val_loss", is_print = False)
    rb_val_psnr = RecordBox(name = "val_psnr", is_print = False)
    rb_val_ssim = RecordBox(name = "val_ssim", is_print = False)
    rb_val_niqe = RecordBox(name = "val_niqe", is_print = False)
    
    rb_test_loss = RecordBox(name = "test_loss", is_print = False)
    rb_test_psnr = RecordBox(name = "test_psnr", is_print = False)
    rb_test_ssim = RecordBox(name = "test_ssim", is_print = False)
    rb_test_niqe = RecordBox(name = "test_niqe", is_print = False)
    
    
    calc_niqe = _calc_niqe()         # new niqe method
    #>>> new_record_system
    
    
    
    for i_epoch in range(HP_EPOCH):
        # train -> val -> test -> train ... 순환
        
        #<<< new_record_system
        # epoch 단위 기록
        d_log_epoch_train = {}
        d_log_epoch_val   = {}
        d_log_epoch_test  = {}
        
        # epoch log dict의 dict
        d_d_log_epoch = {list_mode[0]: d_log_epoch_train
                        ,list_mode[1]: d_log_epoch_val
                        ,list_mode[2]: d_log_epoch_test
                        }
        
        #>>> new_record_system
        
        
        for i_mode in list_mode:
            print("--- init", i_mode, "---")
            # [공용 변수 초기화] ---
            # 오류 기록용 dict
            dict_log_error = {}
            update_dict_v2("", "오류 기록용 dict"
                          ,in_dict = dict_log_error
                          ,in_print_head = "dict_log_error"
                          ,is_print = False
                          )
            # 오류 발생여부 flag
            flag_error = 0
            
            #<<< #i_mode in list_mode
            # GPU 캐시 메모리 비우기
            torch.cuda.empty_cache()
            
            # 이번 epoch 첫 batch 여부 플래그
            flag_init_epoch = 0
            
            # 현재 batch 번호 (이미지 묶음 단위)
            i_batch = 0
            
            
            # 이번 epoch loss 총 합
            epoch_loss_sum = 0
            # 이번 epoch miou 총 합
            epoch_miou_sum = 0
            
            # epoch log dict 들의 머리글(표 최상단) 설정
            for i_key in list_mode:
                #<<< new_record_system
                #epoch 번호 - batch 번호, 파일 이름, Loss PSRN SSIM NIQE mIoU IoUs
                update_dict_v2(i_key + "_"+ str(i_epoch + 1), "batch_num,file_name,Loss,PSNR,SSIM,NIQE"
                              ,in_dict_dict = d_d_log_epoch
                              ,in_dict_key = i_key
                              ,in_print_head = "d_log_epoch_" + i_key
                              ,is_print = False
                              )
                
                #>>> new_record_system
                
            
            
            #<<<
            #[모드별 변수 초기화] ---
            if i_mode == "train":
                torch.cuda.empty_cache()
                #현재 모드 batch size 재설정 (생성할 patch 개수를 의미)
                current_batch_size = HP_BATCH_TRAIN
                #dataloader 설정
                if i_epoch != 0 and i_epoch % 10 == 0:
                    #re generate train dataset
                    dataset_train = Custom_Dataset_V3(# Return: dict key order -> 'file_name'
                                                      #                         , 'pil_img_hr', 'ts_img_hr', 'info_augm'
                                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                                      name_memo                     = 'train'
                                                     ,in_path_dataset               = PATH_BASE_IN
                                                     ,in_category                   = NAME_FOLDER_TRAIN
                                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES
                                                     
                                                      #--- options for train 
                                                     ,is_train                      = True
                                                      # below options can be skipped when above option is False
                                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                                     
                                                      #--- options for HR label
                                                     ,is_return_label               = False
                                                     
                                                      #--- options for LR image
                                                     ,is_return_image_lr            = True
                                                      # below options can be skipped when above option is False
                                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                                      # sub-option a : don't use with sub-option b
                                                      #,in_path_dlc                 = not used
                                                      #,in_name_dlc_csv             = not used
                                                      # sub-option b : don't use with sub-option a
                                                     ,opt_dg_blur                   = "Gaussian"
                                                     ,opt_dg_interpolation          = HP_DG_RESIZE_OPTION
                                                     ,opt_dg_noise                  = "Gaussian"
                                                     ,opt_dg_noise_sigma_range      = HP_DG_RANGE_NOISE_SIGMA
                                                     ,opt_dg_noise_gray_prob        = HP_DG_NOISE_GRAY_PROB
                                                     
                                                      #--- increase dataset length
                                                     ,in_dataset_loop               = HP_DATASET_LOOP               
                                                     
                                                      #--- options for generate patch
                                                     ,is_patch                      = True
                                                     ,patch_stride                  = HP_PATCH_STRIDES
                                                     ,patch_crop_init_range         = HP_PATCH_CROP_INIT_COOR_RANGE
                                                     
                                                      #--- options for generate tensor
                                                     ,model_input_size              = (HP_MODEL_IMG_W, HP_MODEL_IMG_H)
                                                     ,transform_img                 = transform_to_ts_img             
                                                     )
                    
                    dataloader_train = torch.utils.data.DataLoader(dataset     = dataset_train
                                                                  ,batch_size  = HP_BATCH_TRAIN
                                                                  ,shuffle     = True
                                                                  ,num_workers = 0
                                                                  ,prefetch_factor = 2
                                                                  ,drop_last = True
                                                                  )
                
                dataloader_input = dataloader_train
                #모델 모드 설정 (train / eval)
                model.train()
                torch.cuda.empty_cache()
                time.sleep(3)
            elif i_mode == "val":
                torch.cuda.empty_cache()
                #현재 모드 batch size 재설정
                current_batch_size = HP_BATCH_VAL
                dataloader_input = dataloader_val
                model.eval()
                torch.cuda.empty_cache()
                time.sleep(3)
            elif i_mode == "test":
                torch.cuda.empty_cache()
                #현재 모드 batch size 재설정 (patch 생성 없이 원본 이미지 입력 시행)
                current_batch_size = HP_BATCH_TEST
                dataloader_input = dataloader_test
                model.eval()
                torch.cuda.empty_cache()
                time.sleep(3)
            #>>>
            
            #전체 batch 개수
            i_batch_max = len(dataloader_input)
            
            print("\ncurrent_batch_size & total_batch_numbers", current_batch_size, i_batch_max)
            
            
            count_dataloader = 0
            
            # MP 함수용 버퍼 (초기화)
            try:
                del list_mp_buffer
            except:
                pass
            list_mp_buffer = []
            
            
            
            # [train val test 공용 반복 구간] ---
            for dataloader_items in dataloader_input:
                count_dataloader += 1
                # 이제 콘솔 출력 epoch 값과 실제 epoch 값이 동일함
                #print("", end = '\r')
                print("\rin", i_mode, (i_epoch + 1), count_dataloader, "/", i_batch_max, end = '')
                
                #print("number of items from dataloader:", len(dataloader_items), type(dataloader_items))
                
                #for i in range(len(dataloader_items)):
                #    print(i, type(dataloader_items[i]))
                
                #item distribute (length == batch size)
                if i_mode == "train":
                    dl_str_file_name    = dataloader_items[0]
                    dl_pil_img_hr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[2])
                    dl_ts_img_hr        = dataloader_items[2].float()
                    dl_str_info_augm    = dataloader_items[3]
                    
                    dl_pil_img_lr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_lr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[5])
                    dl_ts_img_lr        = dataloader_items[5].float().requires_grad_(True)
                    dl_str_info_deg     = dataloader_items[6]
                    
                elif i_mode == "val":
                    dl_str_file_name    = dataloader_items[0]
                    dl_pil_img_hr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[2])
                    dl_ts_img_hr        = dataloader_items[2].float()
                    dl_str_info_augm    = ["Not Train: no augmentation applied"]
                    
                    dl_pil_img_lr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[3])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_lr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
                    dl_ts_img_lr        = dataloader_items[4].float()
                    dl_str_info_deg     = dataloader_items[5]
                    
                elif i_mode == "test":
                    dl_str_file_name    = dataloader_items[0]
                    dl_pil_img_hr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[2])
                    dl_ts_img_hr        = dataloader_items[2].float()
                    dl_str_info_augm    = ["Not Train: no augmentation applied"]
                    
                    dl_pil_img_lr_raw   = tensor_2_list_pils_v1(in_tensor = dataloader_items[3])
                    #resized or patch ver of RAW -> Test should same with RAW
                    dl_pil_img_lr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
                    dl_ts_img_lr        = dataloader_items[4].float()
                    dl_str_info_deg     = dataloader_items[5]
                    
                
                '''
                print("\nDataloader 형태 확인")
                print("dl_str_file_name", type(dl_str_file_name), dl_str_file_name)
                print("dl_str_info_augm", type(dl_str_info_augm), dl_str_info_augm)
                print("dl_str_info_deg",  type(dl_str_info_deg),  dl_str_info_deg)
                
                print("dl_pil_img_hr", type(dl_pil_img_hr), len(dl_pil_img_hr))
                print("dl_pil_img_lr", type(dl_pil_img_lr), len(dl_pil_img_lr))
                
                print("dl_ts_img_hr", type(dl_ts_img_hr), dl_ts_img_hr.shape)
                #initial input = (in_b, in_c, in_h, in_w) -> 
                print("dl_ts_img_lr", type(dl_ts_img_lr), dl_ts_img_lr.shape)
                
                imshow_pil(dl_pil_img_hr[0])
                imshow_pil(dl_pil_img_lr[0])
                '''
                
                
                
                dl_ts_img_hr = dl_ts_img_hr.to(device)
                dl_ts_img_lr = dl_ts_img_lr.to(device)
                
                
                
                if i_mode == "train":
                    if i_batch == 0:
                        # 기울기 초기화
                        optimizer.zero_grad()
                        print("optimizer.zero_grad()")
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        #<<< AMP
                        in_batch_size, _, _, _ = dl_ts_img_lr.shape
                        if i_mode == "train" and in_batch_size != current_batch_size:
                            print("Batch size is not same with HyperParameter:", in_batch_size, current_batch_size)
                            sys.exit(-1)
                        
                        # Model 예측결과 생성 & Loss 계산
                        if model_name == "MPRNet":
                            # (tensor) SR 결과 [stage 3, 2, 1]
                            tensor_out_sr_set = model(dl_ts_img_lr)
                            tensor_out_sr = tensor_out_sr_set[0]
                            loss = criterion(tensor_out_sr_set, dl_ts_img_hr)
                            
                        elif model_name == 'ESRT':
                            tensor_out_sr = model(dl_ts_img_lr)
                            loss = criterion(tensor_out_sr, dl_ts_img_hr)
                            
                        batch_loss = loss.item()
                        
                        # SR 이미지텐서 역 정규화
                        if is_norm_in_transform_to_tensor:
                            #MPRNet 사용 안함
                            tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                    
                        #>>> AMP
                    
                    #<<< new_record_system
                    rb_train_loss.add_item(loss.item())
                    #>>> new_record_system
                    
                    try:
                        # loss overflow, underflow 오류 방지
                        amp_scaler.scale(loss).backward()
                    except:
                        flag_error = 1
                        update_dict_v2("", "in " + str(i_epoch) + " " + str(i_batch)
                                      ,"", "loss.backward 실패():" + str(loss.item())
                                      ,in_dict = dict_log_error
                                      ,in_print_head = "dict_log_error"
                                      )
                    
                    # 가중치 갱신 (batch 마다)
                    #optimizer.step()
                    amp_scaler.step(optimizer)
                    #print("optimizer.step()", i_batch)
                    amp_scaler.update()
                    # 기울기 초기화
                    optimizer.zero_grad()
                    #print("optimizer.zero_grad()")
                    if HP_SCHEDULER_UPDATE_INTERVAL == "batch":
                        # 스케쥴러 갱신
                        scheduler.step()
                        print("scheduler.step()")
                    
                
                else: # val or test
                    with torch.no_grad():
                        #<<< @@@
                        # Model 예측결과 생성 & Loss 계산
                        if model_name == "MPRNet":
                            # (tensor) SR 결과 [stage 3, 2, 1]
                            tensor_out_sr_set = model(dl_ts_img_lr)
                            tensor_out_sr = tensor_out_sr_set[0]
                            loss = criterion(tensor_out_sr_set, dl_ts_img_hr)
                            
                        elif model_name == 'ESRT':
                            tensor_out_sr = model(dl_ts_img_lr)
                            loss = criterion(tensor_out_sr, dl_ts_img_hr)
                        
                        batch_loss = loss.item()
                        
                        # SR 이미지텐서 역 정규화
                        if is_norm_in_transform_to_tensor:
                            #MPRNet 사용 안함
                            tensor_out_sr = transform_ts_inv_norm(tensor_out_sr)
                        
                        #>>> @@@
                        
                        
                        #<<< new_record_system
                        if i_mode == "val":
                            rb_val_loss.add_item(loss.item())
                        elif i_mode == "test":
                            rb_test_loss.add_item(loss.item())
                        #>>> new_record_system
                
                
                
                #VVV [Tensor -> model 예측결과 생성] ----------------------- 
                
                
                list_out_pil_sr = tensor_2_list_pils_v1(# 텐서 -> pil 이미지 리스트
                                                       # (tensor) 변환할 텐서, 모델에서 다중 결과물이 생성되는 경우, 단일 출력물 묶음만 지정해서 입력 
                                                       # (예: MPRNet -> in_tensor = tensor_sr_hypo[0])
                                                       in_tensor = tensor_out_sr
                                                       
                                                       # (bool) 라벨 여부 (출력 pil 이미지 = 3ch, 라벨 = 1ch 고정) (default: False)
                                                      ,is_label = False
                                                       
                                                       # (bool) pil 이미지 크기 변환 시행여부 (default: False)
                                                      ,is_resized = False
                                                      )
                
                
                #AAA [batch 단위 이미지 평가] --------------------------
                
                for i_image in range(current_batch_size):
                    # 입력 x LR 이미지
                    # list_patch_pil_x[i_image] -> dl_pil_img_lr[i_image]
                    
                    # 입력 y HR 라벨 
                    # list_patch_pil_y[i_image] -> dl_pil_lab_hr[i_image]
                    
                    #라벨 예측결과를 원본 라벨 크기로 변환 ->  크기변환 없음
                    #pil_hypo_resized = list_out_pil_label[i_image]
                    
                    #--- PSNR SSIM NIQE
                    try:
                        out_psnr, out_ssim = calc_psnr_ssim(pil_original = dl_pil_img_hr[i_image]
                                                           ,pil_contrast = list_out_pil_sr[i_image]
                                                           )
                    except:
                        print("(exc) PSRN SSIM calc FAIL")
                        out_psnr, out_ssim = -999, -999
                    
                    try:
                        #out_niqe = calc_niqe_with_pil(list_out_pil_sr[i_image])
                        out_niqe = calc_niqe.with_pil(list_out_pil_sr[i_image])
                    except:
                        print("(exc) NIQE calc FAIL")
                        out_niqe = -999
                    
                    #<<< new_record_system
                    if i_mode == "train":
                        rb_train_psnr.add_item(out_psnr)
                        rb_train_ssim.add_item(out_ssim)
                        rb_train_niqe.add_item(out_niqe)
                    elif i_mode == "val":
                        rb_val_psnr.add_item(out_psnr)
                        rb_val_ssim.add_item(out_ssim)
                        rb_val_niqe.add_item(out_niqe)
                    elif i_mode == "test":
                        rb_test_psnr.add_item(out_psnr)
                        rb_test_ssim.add_item(out_ssim)
                        rb_test_niqe.add_item(out_niqe)
                    
                    #>>> new_record_system
                    
                    # 이미지 단위 로그 갱신
                    
                    #<<< new_record_system
                    #epoch 번호 - batch 번호, 파일 이름, Loss PSRN SSIM NIQE  #<<< @@@작성중 20220705
                    
                    tmp_str_contents = (str(count_dataloader) + "," + dl_str_file_name[i_image] + "," + str(batch_loss) 
                                       +"," + str(out_psnr) + "," + str(out_ssim) + "," + str(out_niqe)
                                       )
                    
                    update_dict_v2("", tmp_str_contents
                                  ,in_dict_dict = d_d_log_epoch
                                  ,in_dict_key = i_mode
                                  ,in_print_head = "d_log_epoch_" + i_mode
                                  ,is_print = False
                                  )
                    
                    #>>> new_record_system
                    
                    #<<< 예측결과를 이미지로 생성
                    
                    if i_mode == "train":
                        if i_batch % (i_batch_max//20 + 1) == 0:
                            # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                            plt_title = "File name: " + dl_str_file_name[i_image]
                            plt_title += "\n" + dl_str_info_augm[i_image]
                            if option_apply_degradation:
                                # 현재 patch의 degrad- 옵션 불러오기
                                plt_title += "\n" + dl_str_info_deg[i_image]
                            plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                            plt_title += "  SSIM: " + str(round(out_ssim, 4))
                            plt_title += "  NIQE: " + str(round(out_niqe, 4))
                            
                            tmp_bool = True
                        else:
                            tmp_bool = False
                    
                    elif i_mode == "test" and RUN_WHERE == -1:  #Test phase on colab
                        if i_batch % (i_batch_max//20 + 1) == 0 or dl_str_file_name[i_image] == "0016E5_08123.png":
                            # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                            plt_title = "File name: " + dl_str_file_name[i_image]
                            if option_apply_degradation:
                                # 현재 patch의 degrad- 옵션 불러오기
                                plt_title += "\n" + dl_str_info_deg[i_image]
                            
                            plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                            plt_title += "  SSIM: " + str(round(out_ssim, 4))
                            plt_title += "  NIQE: " + str(round(out_niqe, 4))
                            tmp_bool = True
                        else:
                            tmp_bool = False
                    
                    else: # val or test
                        # 모든 이미지를 저장함
                        plt_title = "File name: " + dl_str_file_name[i_image]
                        if option_apply_degradation:
                            # 현재 patch의 degrad- 옵션 불러오기
                            plt_title += "\n" + dl_str_info_deg[i_image]
                        
                        plt_title += "\nPSNR: " + str(round(out_psnr, 4))
                        plt_title += "  SSIM: " + str(round(out_ssim, 4))
                        plt_title += "  NIQE: " + str(round(out_niqe, 4))
                        tmp_bool = True
                    
                    if tmp_bool:
                        tmp_file_name = (i_mode + "_" + str(i_epoch + 1) + "_" + str(i_batch + 1) + "_"
                                        +dl_str_file_name[i_image]
                                        )
                        
                        if model_name == "MPRNet" or model_name == "ESRT":
                            # SR model train does not use mp_buffer
                            # RAM 할당량 보고 버퍼 사용여부 결정하기
                            # 버퍼 사용시, pil 저장을 위한 is_best 처리방식 수정해야됨
                            
                            if len(list_mp_buffer) >= BUFFER_SIZE and BUFFER_SIZE > 0:
                                # chunk full -> toss mp_buffer -> empty mp_buffer
                                if i_mode == 'test':
                                    tmp_is_best = rb_val_psnr.is_best_max   #chunk 단위 buffer 구조상 valid 기준으로 best여부 검사
                                else:
                                    tmp_is_best = False
                                
                                plts_saver(list_mp_buffer, is_best = tmp_is_best)
                                
                                try:
                                    del list_mp_buffer
                                except:
                                    pass
                                list_mp_buffer = []
                            
                            
                            list_mp_buffer.append((# 0 (model name)
                                                   model_name
                                                   # 1 ~ 3 (pils)
                                                  ,dl_pil_img_hr[i_image]
                                                  ,dl_pil_img_lr[i_image]
                                                  ,list_out_pil_sr[i_image]
                                                   # 4 ~ 6 (sub title)
                                                  ,"HR Image", "LR Image", "SR Image"
                                                   # 7 (path for plt)
                                                  ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                                   # 8 (path for SR pil)
                                                  ,PATH_OUT_IMAGE + i_mode + "/_SR_Images/" + str(i_epoch + 1)
                                                   # 9 (plt title)
                                                  ,plt_title
                                                   # 10 (file name)
                                                  ,tmp_file_name
                                                  )
                                                 )
                            
                            
                    #>>> 예측결과를 이미지로 생성
                    
                
                #VVV [batch 단위 이미지 평가] --------------------------
                
                
                #<<< new_record_system
                if i_mode == "train":
                    rb_train_loss.update_batch()
                    rb_train_psnr.update_batch()
                    rb_train_ssim.update_batch()
                    rb_train_niqe.update_batch()
                elif i_mode == "val":
                    rb_val_loss.update_batch()
                    rb_val_psnr.update_batch()
                    rb_val_ssim.update_batch()
                    rb_val_niqe.update_batch()
                elif i_mode == "test":
                    rb_test_loss.update_batch()
                    rb_test_psnr.update_batch()
                    rb_test_ssim.update_batch()
                    rb_test_niqe.update_batch()
                    
                #>>> new_record_system
                
                
                try:
                    del dl_ts_img_hr
                    del dl_ts_img_lr
                    del tensor_out_sr_set
                    del tensor_out_sr
                except:
                    pass
                
                i_batch += 1
                
            # End of "for path_x, path_y in dataloader_input:"
            # dataloader_input 종료됨
            
            
                
            
            #<<< new_record_system
            if i_mode == "train":
                str_result_epoch_loss = str(rb_train_loss.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_psnr = str(rb_train_psnr.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_ssim = str(rb_train_ssim.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_niqe = str(rb_train_niqe.update_epoch(is_return = True, is_print_sub = True))
                
            elif i_mode == "val":
                str_result_epoch_loss = str(rb_val_loss.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_psnr = str(rb_val_psnr.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_ssim = str(rb_val_ssim.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_niqe = str(rb_val_niqe.update_epoch(is_return = True, is_print_sub = True))
                
            elif i_mode == "test":
                str_result_epoch_loss = str(rb_test_loss.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_psnr = str(rb_test_psnr.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_ssim = str(rb_test_ssim.update_epoch(is_return = True, is_print_sub = True))
                str_result_epoch_niqe = str(rb_test_niqe.update_epoch(is_return = True, is_print_sub = True))
                
            # summary plt pil save
            if model_name == "MPRNet" or model_name == "ESRT":
                if not list_mp_buffer:
                    print("(caution) list_mp_buffer is emply...")
                else:
                    if i_mode == 'test':
                        if BUFFER_SIZE > 0:     # 버퍼 사용된 경우
                            tmp_is_best = rb_val_psnr.is_best_max
                        else:                   # 버퍼 사용 안하는 경우
                            tmp_is_best = rb_val_psnr.is_best_max or rb_test_psnr.is_best_max
                    else:
                        tmp_is_best = False
                    plts_saver(list_mp_buffer, is_best = tmp_is_best)
            
            # log total dict 업데이트
            tmp_str_contents = str_result_epoch_loss
            tmp_str_contents += "," + str_result_epoch_psnr + "," + str_result_epoch_ssim + "," + str_result_epoch_niqe
            #epoch 번호 - Loss PSRN SSIM NIQE mIoU IoUs
            update_dict_v2(str(i_epoch + 1), tmp_str_contents
                          ,in_dict_dict = d_d_log_total
                          ,in_dict_key = i_mode
                          ,in_print_head = "d_log_total_" + i_mode
                          )
            print("\n")
            
            # log 기록 업데이트 (epoch 단위)
            dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                         ,in_file_name = "new_log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                         ,in_dict_dict = d_d_log_epoch
                         ,in_dict_key = i_mode
                         )
            # log 기록 업데이트 (학습 전체 단위)
            dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                         ,in_file_name = "new_log_total_" + i_mode + ".csv"
                         ,in_dict_dict = d_d_log_total
                         ,in_dict_key = i_mode
                         )
            
            #>>> new_record_system
            
            
            
            
            # epoch 단위 scheduler 갱신 -> state_dict & check_point 저장 
            if i_mode == "train":
                if HP_SCHEDULER_UPDATE_INTERVAL == "epoch":
                    # 스케쥴러 갱신
                    scheduler.step()
                    print("scheduler.step()")
                
                # state_dict 저장경로
                tmp_path = PATH_OUT_MODEL + "state_dicts/"
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                torch.save(model.state_dict()
                          ,tmp_path + str(i_epoch + 1) +'_model_sr_state_dict.pt'
                          )
                
                if (i_epoch+1) % 10 == 0:
                    tmp_path = PATH_OUT_MODEL + "check_points/"
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    # 모델 체크포인트 저장
                    torch.save({'epoch': (i_epoch + 1)                           # (int) 중단 시점 epoch 값
                               ,'model_state_dict': model.state_dict()        # (state_dict) model.state_dict()
                               ,'optimizer_state_dict': optimizer.state_dict()   # (state_dict) optimizer.state_dict()
                               ,'scheduler_state_dict': scheduler.state_dict()   # (state_dict) scheduler.state_dict()
                               }
                              ,tmp_path + str(i_epoch + 1) +'_check_point.tar'
                              )
                
                
            # 에러 발생했던 경우, 로그 저장
            if flag_error != 0:
                dict_2_txt(in_file_path = PATH_OUT_LOG + "error/"
                          ,in_file_name = "log_error_" + i_mode + "_" + str(i_epoch) + ".csv"
                          ,in_dict = dict_log_error
                          )
            
            # [epoch 완료 -> 변수 초기화] ---
            
            print("epoch 완료")
            i_batch = 0
            
            

#=== End of trainer_sr


"""
print("End of trainer_sr.py")
