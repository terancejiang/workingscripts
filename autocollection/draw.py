import copy

import cv2
import os
import re
import pandas as pd
import numpy as np
from ast import literal_eval, parse
from tqdm import tqdm

def padding(xyxy, height, width, padding_size):
    if xyxy[0] - padding_size < 0:
        xyxy[0] = 0
    else:
        xyxy[0] = xyxy[0] - padding_size

    if xyxy[1] - padding_size < 0:
        xyxy[1] = 0
    else:
        xyxy[1] = xyxy[1] - padding_size

    if xyxy[2] + padding_size > width:
        xyxy[2] = width
    else:
        xyxy[2] = xyxy[2] + padding_size

    if xyxy[3] + padding_size > height:
        xyxy[3] = height
    else:
        xyxy[3] = xyxy[3] + padding_size

    return xyxy[0], xyxy[1], xyxy[2], xyxy[3]


def crop_image( obj, w, h, frame_with_color):
    obj_tl, obj_br = obj["coordinates"]

    if obj_tl[0] < 1:
        x1 = int(obj_tl[0] * w)
        y1 = int(obj_tl[1] * h)
        x2 = int(obj_br[0] * w)
        y2 = int(obj_br[1] * h)
    else:
        x1 = int(obj_tl[0] * w)
        y1 = int(obj_tl[1] * h)
        x2 = int(obj_br[0] * w)
        y2 = int(obj_br[1] * h)

    x1, y1, x2, y2 = padding([x1, y1, x2, y2], h, w, 10)

    obj_crop_image_color = frame_with_color[y1:y2, x1:x2]
    obj_crop_image_color_opencv = obj_crop_image_color

    return obj_crop_image_color_opencv


df = pd.read_csv('obj_04_13_exported.csv')
subdf = df[['location','image_url']]
for root, dirs, files in os.walk('ultra_obj_selected'):
    for file in tqdm(files):
        fname = file.split('.')[0]
        location = subdf[subdf['image_url'].str.contains(fname)]["location"]
        location = [tuple(x.split(',')) for x in re.findall("\((.*?)\)", str(location.values))]
        # parsed = parse(str(location.values))
        # location = [tuple(el.id for el in t.elts) for t in parsed.body[0].value.elts]
        # location = literal_eval(str(location.values))
        file = os.path.join(root, file)
        img = cv2.imread(file)
        h,w,c = img.shape
        x1 = int(float(location[0][0]) * w)
        y1 = int(float(location[0][1]) * h)
        x2 = int(float(location[1][0]) * w)
        y2 = int(float(location[1][1]) * h)
        img_draw = copy.deepcopy(img)
        cv2.rectangle(img_draw,(x1,y1),(x2,y2), (255,255,0),2)

        save_path = file.replace('ultra_obj_selected','ultra_obj_selected_draw')
        save_dir = "/".join(save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path,img_draw)

        save_path_crop = file.replace('ultra_obj_selected','ultra_obj_selected_crop')
        save_dir_crop = "/".join(save_path_crop.split('/')[:-1])
        if not os.path.exists(save_dir_crop):
            os.makedirs(save_dir_crop)
        x1, y1, x2, y2 = padding([x1, y1, x2, y2], h, w, 10)

        obj_crop_image_color = img[y1:y2, x1:x2]
        cv2.imwrite(save_path_crop,obj_crop_image_color)
