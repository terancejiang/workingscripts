import os
import pandas as pd
import ast
import requests
from tqdm import tqdm
from autocollection.aliyun_oss import oss_delete_file
df = pd.read_csv('2021-03-02-10-13-43_EXPORT_CSV_2176571_340_0.csv')
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
        with open(label_store_path,'w') as wtxt:
            wtxt.write(location)
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
