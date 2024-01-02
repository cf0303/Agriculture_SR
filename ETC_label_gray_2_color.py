import os
from PIL import Image
from utils.data_tool import label_2_RGB, imshow_pil

path_in = ""
path_out = ""


#라벨 당 컬러 지정 (CamVid 12)
#label_color_map -> HP_COLOR_MAP
HP_COLOR_MAP = {0:  [128, 128, 128]  # 00 Sky
               ,1:  [128,   0,   0]  # 01 Building
               ,2:  [192, 192, 128]  # 02 Column_pole
               ,3:  [128,  64, 128]  # 03 Road
               ,4:  [  0,   0, 192]  # 04 Sidewalk
               ,5:  [128, 128,   0]  # 05 Tree
               ,6:  [192, 128, 128]  # 06 SignSymbol
               ,7:  [ 64,  64, 128]  # 07 Fence
               ,8:  [ 64,   0, 128]  # 08 Car
               ,9:  [ 64,  64,   0]  # 09 Pedestrian
               ,10: [  0, 128, 192]  # 10 Bicyclist
               ,11: [  0,   0,   0]  # 11 Void
               }

#----------------------------------------------------------------------------

if path_in[-1] == '/':
    path_in = path_in[:-1]

if path_out[-1] == '/':
    path_out = path_out[:-1]

if not os.path.exists(path_out):
    os.makedirs(path_out)

list_label_name = os.listdir(path_in)

#dict_pil_label = {}

for i_name in list_label_name:
    # i_name: 파일 이름
    print(i_name)
    pil_label_gray = Image.open(path_in + '/' + i_name)
    
    pil_label_color = label_2_RGB(pil_label_gray, HP_COLOR_MAP)
    
    #imshow_pil(pil_label_color)
    pil_label_color.save(path_out + '/' + i_name)
    
    


print("EoF: ETC_label_gray_2_color.py")