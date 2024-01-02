
from torchvision.transforms import ToTensor
import cv2
import matplotlib.pyplot as plt
import glob
import cv2
import os
import numpy as np
import random
from skimage import io, color


def resize(path_list, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(path_list)):
        name = path_list[i].split('\\')
        name = name[0].split('/')[-1]
        name = name.split('.')[0]
        img = cv2.imread(path_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # io.imshow(img)
        # io.show()
        img = cv2.imwrite(save_path + name + '.png', img)

te_low_resolution_IMG = glob.glob('')
te_save_path_IMG = ('')

resize(te_low_resolution_IMG,te_save_path_IMG)
