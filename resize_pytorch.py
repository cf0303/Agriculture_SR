import glob
import cv2
import os
import numpy as np
import random
import torch
SEED = 15

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      h,w,_= image.shape
      mean = 0
      var = 0
      sigma = random
      gauss = np.random.normal(mean,sigma,(h,w,_))
      gauss = gauss.reshape(h,w,_)
      noisy = image + gauss
      return noisy
   if noise_typ == "s&p":
      h,w,_ = image.shape
      s_vs_p = 0.5
      amount = 0.05
      out = np.copy(image)
      num_salt = np.ceil(amount * h*w*3 * s_vs_p)

      B_channel = out[:, :, 0]
      # B_channel = np.reshape(B_channel, [h*w,])
      G_channel = out[:, :, 1]
      # G_channel = np.reshape(G_channel, [h * w, ])
      R_channel = out[:, :, 2]
      # R_channel = np.reshape(R_channel, [h * w, ])

      salt_pixel_range_h = random.randint(0, h)
      salt_pixel_range_w = random.randint(0, w)
      salt_h_coords = [random.randint(0, i) for i in range(salt_pixel_range_h)]
      salt_w_coords = [random.randint(0, i) for i in range(salt_pixel_range_w)]
      print(int(num_salt))

      temp_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]]  # 255
      B_channel[temp_coords[0], temp_coords[1]] = 255
      temp_coords2 = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]] # 0
      B_channel[temp_coords2[0], temp_coords2[1]] = 0

      temp_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]]  # 255
      G_channel[temp_coords[0], temp_coords[1]] = 255
      temp_coords2 = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]] # 0
      G_channel[temp_coords2[0], temp_coords2[1]] = 0

      temp_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]]  # 255
      R_channel[temp_coords[0], temp_coords[1]] = 255
      temp_coords2 = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape[0:2]] # 0
      R_channel[temp_coords2[0], temp_coords2[1]] = 0

      B_channel = B_channel[:, :, np.newaxis]
      G_channel = G_channel[:, :, np.newaxis]
      R_channel = R_channel[:, :, np.newaxis]

      # a = lambda x,y,z: x[:, :, 0] == 255 and y[:, :, 1] == 0 and z[:, :, 2] == 0

      fun1 = lambda x: np.array(x[:, :, :]) == [255, 0, 0]
      fun2 = lambda x: np.array(x[:, :, :]) == [0, 255, 0]
      fun3 = lambda x: np.array(x[:, :, :]) == [0, 0, 255]

      fun4 = lambda x: np.array(x[:, :, :]) == [255, 255, 255]
      fun5 = lambda x: np.array(x[:, :, :]) == [0, 0, 0]

      out = np.concatenate([B_channel, G_channel, R_channel], -1)
      temp = np.ones_like(out, dtype=np.int32) * 255
      temp2 = np.zeros_like(out, dtype=np.int32)
      out = np.where(fun1(out), fun4(out), out) #B
      out = np.where(fun2(out), fun5(out), out) #G
      out = np.where(fun3(out), fun4(out), out) #R
      out = np.array(out, np.float32)
      print(out.shape)

      # # Salt mode
      # num_salt = np.ceil(amount * image.size * s_vs_p)
      # coords = [np.random.randint(0, i - 1, int(num_salt))
      #         for i in image.shape]
      # out[coords] = 255
      #
      # # Pepper mode
      # num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      # coords = [np.random.randint(0, i - 1, int(num_pepper))
      #         for i in image.shape]
      # out[coords] = 0
      return out
   if noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   if noise_typ =="speckle":
      h,w,_ = image.shape
      gauss = np.random.randn(h,w,_)
      gauss = gauss.reshape(h,w,_)
      noisy = image + image * gauss
      return noisy

def resize1(path_list, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        img = cv2.resize(img, dsize = (1280,960), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img,(3,3),0.1)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize = (w//8,h//8), interpolation=cv2.INTER_LINEAR)
        img = noisy('gauss',img)
        h, w, _ = img.shape

        img = cv2.imwrite(save_path + name + '.png', img)

def resize2(path_list, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        img = cv2.resize(img, dsize = (1280,960), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img,(3,3),0.1)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize = (w//16,h//16), interpolation=cv2.INTER_LINEAR)
        img = noisy('gauss',img)
        img = noisy('poisson',img)
        img = cv2.resize(img, dsize = (1280,960), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        h, w, _ = img.shape

        img = cv2.imwrite(save_path + name + '.png', img)

def resize3(path_list, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        img = cv2.resize(img, dsize = (1280,960), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img,(3,3),0.1)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize = (w//16,h//16), interpolation=cv2.INTER_LINEAR)
        img = noisy('gauss',img)
        img = noisy('s&p',img)
        img = cv2.resize(img, dsize = (1280,960), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = np.clip(img, 0, 255)
        h, w, _ = img.shape
        img = cv2.imwrite(save_path + name + '.png', img)

def resize(path_list, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        h, w, _ = img.shape
        img = cv2.resize(img, dsize = (w//2,h//2), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = cv2.imwrite(save_path + name +'.png', img)

def resize_4(path_list, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        h, w, _ = img.shape
        img = cv2.resize(img, dsize = (w//2,h//2), fx = 0, fy = 0, interpolation=cv2.INTER_LINEAR)
        img = cv2.imwrite(save_path + name +'.png', img)

# tr_low_resolution = glob.glob('C:/Users/sungje/Desktop/data/datasets_IJRR2017/train/image/*')
tr_low_resolution_IMG1 = glob.glob('E:/LAB/datasets/project_use/IJRR2017/A_set/train/images/*')
tr_low_resolution_IMG2 = glob.glob('E:/LAB/datasets/project_use/IJRR2017/A_set/val/images/*')
tr_low_resolution_IMG3 = glob.glob('E:/LAB/datasets/project_use/IJRR2017/A_set/test/images/*')
# tr_save_path = ('C:/Users/sungje/Desktop/data/datasets_IJRR2017/train/tr_low_resolution_LINEAR_64_1024/')
te_save_path_IMG1 = ('E:/LAB/datasets/project_use/IJRR2017_DLC/resize_all/images_resize_2/')
te_save_path_IMG2 = ('/media/sungjae/LINUX1/dataset/LAB/datasets/project_use/rice_s_n_w/resize_set/train/images/')
te_save_path_IMG3 = ('/media/sungjae/LINUX1/dataset/LAB/datasets/project_use/rice_s_n_w/resize_set/val/images/')


t_low_resolution_IMG1 = glob.glob('/LAB/datasets/project_use/IJRR2017_DLC/resize_all/images/*.png')
t_low_resolution_IMG2 = glob.glob('/media/sungjae/LINUX1/dataset/LAB/datasets/project_use/rice_s_n_w/resize_set/train/images/*.png')
t_low_resolution_IMG3 = glob.glob('/media/sungjae/LINUX1/dataset/LAB/datasets/project_use/rice_s_n_w/resize_set/val/images/*.png')
t_save_path_IMG1 = ('/LAB/datasets/project_use/IJRR2017_DLC/resize_8/images/')


# for i in range(len(te_low_resolution_IMG)):
#     name = te_low_resolution_IMG[i].split('\\')
#     name = name[0].split('/')[-1]
#     name = name.split('.')[0]
#     print(name)
# resize(tr_low_resolution, tr_save_path)
# resize(tr_low_resolution_IMG1, te_save_path_IMG1)
# resize(tr_low_resolution_IMG2, te_save_path_IMG2)
# resize(tr_low_resolution_IMG3, te_save_path_IMG3)

# resize_4(t_low_resolution_IMG1, t_save_path_IMG1)
resize(tr_low_resolution_IMG1, te_save_path_IMG1)
resize(tr_low_resolution_IMG2, te_save_path_IMG1)
resize(tr_low_resolution_IMG3, te_save_path_IMG1)

# resize1(tr_low_resolution_IMG_1, te_save_path_IMG_1)

# resize2(te_low_resolution_IMG, te_save_path_IMG_2)
# # resize3(te_low_resolution_IMG, te_save_path_IMG_3)
# resize(te_low_resolution_IMG, te_save_path_IMG)

###############target 안에 있는 이미지들 이름만 불러오기
# a = os.listdir("C:/Users/sungje/Desktop/data/IJRR2017/mpr/train/target/")
# w = open("C:/Users/sungje/Desktop/data/IJRR2017/mpr/train/train.txt", "w")
#
# for data in a:
#     w.write(data)
#     w.write("\n")
#     w.flush()

# f = open("train.txt", 'r')
# lines = f.readline()
# n=1
# for lpLine in f:
#     n= n+1
# f.close()



