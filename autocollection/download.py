import os
import random

from shutil import copyfile

import pandas as pd
import ast
import requests
from tqdm import tqdm
from autocollection.aliyun_oss import oss_delete_file

def download_zxjg():
    df = pd.read_csv('执行结果1.txt')
    df = df.drop('id', axis=1)
    groups = df.groupby('type')

    for group_name,group in groups:

        if 'sink' in group_name:
            store_path_prefix = os.path.join('sink', group_name)
        else:
            store_path_prefix = os.path.join('ultrasonic', group_name)

        for index, row in tqdm(group.iterrows()):
            deviceid = row['device_id']
            store_path = os.path.join(store_path_prefix, deviceid)
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            url = row['image_url']
            filename = url.split('/')[-1]
            r = requests.get(url, allow_redirects=True)
            img_store_path = os.path.join(store_path, filename)
            label_store_path = os.path.join(store_path, filename[:-3]+'txt')
            open( img_store_path , 'wb').write(r.content)

            oss_delete_file(url)
            location = row['location']
            # with open(label_store_path,'w') as wtxt:

def select_from_autoc():
    for root, dirs, files in os.walk('ultra_nobj'):
        for d in dirs:
            for rootd,dirsd,filesd in os.walk(os.path.join(root,d)):
                for dd in dirsd:
                    imgs_num = os.listdir(os.path.join(root,d,dd))
                    if len(imgs_num) > 30:
                        imgs = random.sample(imgs_num,30)
                    else:
                        imgs = imgs_num
                    for img in imgs:
                        imgpath = os.path.join(os.getcwd(),'ultra_nobj_selected',d,img)
                        if not os.path.exists(os.path.join(os.getcwd(), 'ultra_nobj_selected',d)):
                            os.makedirs(os.path.join(os.getcwd(), 'ultra_nobj_selected',d))
                        ori_imgpath = os.path.join(root,d,dd,img)
                        copyfile(ori_imgpath, imgpath)
        break

def download_obj(file):
    df = pd.read_csv(file)
    for index, row in tqdm(df.iterrows()):
        try:
            url = row['image_url']
            ulra_type = row['ultrasonic_type']
            device_id = row['device_id']
            filename = url.split('/')[-1]

            if str(ulra_type)=='nan' :
                file_list = []
                for root, dirs, files in os.walk('/mnt/740535b3-57f4-42bf-9664-ce7b4ff7af2e/projects/workingscripts/autocollection/ultra_close'):
                    for d in dirs:
                        subdirs = os.listdir(os.path.join(root, d))
                        if device_id in subdirs:
                            ulra_type = d
                            break
                    break
            if device_id:
                store_location = os.path.join(os.getcwd(),'ultra_close', str(int(ulra_type)), str(device_id))
                if not os.path.exists(store_location):
                    os.makedirs(store_location)
                store_path = os.path.join(store_location, filename)
                if not os.path.exists(store_path):
                    r = requests.get(url, allow_redirects=True, timeout=2)
                    open( store_path , 'wb').write(r.content)
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':
    # select_from_autoc()
    # download_obj('noobj_04_13_exported.csv')
    download_obj('ultra_close_04-15.csv')

    # download_obj('obj_04_13_exported.csv')

            # location = ast.literal_eval(location)
            # print(row)
            # print(location[0])

# with open('2021-03-02-10-13-43_EXPORT_CSV_2176571_340_0.csv','r') as txt:
#     lines = txt.readlines()
#
# for li in lines:
#     li = li.replace('"','')
#     elements = li.split(',')
#     type = elements[1]
#     location = elements[2]
#     url = elements[3]
