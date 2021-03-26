import os
import pandas as pd
import ast
import requests
from tqdm import tqdm
import cv2
from autocollection.aliyun_oss import oss_delete_file
# df = pd.read_csv('ultraundetacted_03-22.txt')
#
#
# for index, row in tqdm(df.iterrows()):
#     did = row['device_id']
#     url = row["video_url"]
#     print(url)
#     vcap = cv2.VideoCapture(url)
#     success, frame = vcap.read()
#     cv2.imshow('img', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    # location = ast.literal_eval(location)
        # print(row)


# with open('2021-03-02-10-13-43_EXPORT_CSV_2176571_340_0.csv','r') as txt:
#     lines = txt.readlines()
#
# for li in lines:
#     li = li.replace('"','')
#     elements = li.split(',')
#     type = elements[1]
#     location = elements[2]
#     url = elements[3]
