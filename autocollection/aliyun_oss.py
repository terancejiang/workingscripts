import os
import oss2

from functools import lru_cache
from dynaconf import settings

local_endpoint = 'https://oss-cn-{}-internal.aliyuncs.com'  # 局域网配置

config = settings.get('ALI_OSS', {})
region = config.get('region', 'hangzhou')
local_endpoint = config.get('endpoint', local_endpoint.format(region))

@lru_cache()
def _get_bucket():
    bucket = oss2.Bucket(oss2.Auth(config['access_key_id'], config['access_key_secret']),
                         local_endpoint, config['bucket_name'])
    return bucket


def oss_upload_data(filename, data):
    """ 上传文件

    """
    # 用私有域名替换掉阿里的域名
    url_base = config.get('internal', 'https://{}.oss-cn-{}.aliyuncs.com/'.format(config['bucket_name'], region))

    bucket = _get_bucket()
    result = bucket.put_object(filename, data)
    if result.status == 200:
        origin_file_url = os.path.join(url_base, filename)
    else:
        origin_file_url = ''
    return origin_file_url


def oss_upload_file(origin_file, local_file):
    """ 上传文件

    """
    # 用私有域名替换掉阿里的域名
    url_base = config.get('internal', 'https://{}.oss-cn-{}.aliyuncs.com/'.format(config['bucket_name'], region))

    bucket = _get_bucket()
    result = bucket.put_object_from_file(origin_file, local_file)
    if result.status == 200:
        origin_file_url = os.path.join(url_base, origin_file)
    else:
        origin_file_url = ''
    return origin_file_url


def oss_delete_file(origin_file):
    """ 删除单个文件
    """
    res = oss_batch_delete_files([origin_file])
    return True if len(res) > 0 else False


def oss_batch_delete_files(files_list):
    """ 批量删除云端文件
    """
    bucket = _get_bucket()
    bucket_files = []
    for origin_file in files_list:
        if origin_file.startswith('http://') or origin_file.startswith('https://'):
            bucket_files.append(origin_file.split('/', 3)[-1])
        else:
            bucket_files.append(origin_file)

    result = bucket.batch_delete_objects(bucket_files)
    return result.deleted_keys


def oss_download_file(origin_file, local_file):
    """
    下载视频文件
    """
    bucket = _get_bucket()
    if origin_file.startswith('http://') or origin_file.startswith('https://'):
        origin_file = origin_file.split('/', 3)[-1]

    bucket.get_object_to_file(origin_file, local_file)
    return local_file


if __name__ == '__main__':

    pass