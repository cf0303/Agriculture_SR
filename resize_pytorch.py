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

tr_low_resolution_IMG = glob.glob('')

te_save_path_IMG = ('')

t_low_resolution_IMG = glob.glob('')
t_save_path_IMG = ('')

resize(tr_low_resolution_IMG, te_save_path_IMG)



