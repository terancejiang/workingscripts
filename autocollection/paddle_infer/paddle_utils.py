# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor
from dynaconf import settings


def parse_args(type=''):
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    dynaconf_config_paddle = settings.get('paddle', {})

    # general params
    parser = argparse.ArgumentParser()

    # if type == 'ultrasonic':
    # parser.add_argument("--top_k", type=int, default=3)
    ultra_models = [
        dynaconf_config_paddle.ULTRA_MODEL_FILE_0,
        dynaconf_config_paddle.ULTRA_MODEL_FILE_1,
        dynaconf_config_paddle.ULTRA_MODEL_FILE_2,
        dynaconf_config_paddle.ULTRA_MODEL_FILE_3]
    ultra_models_params = [
        (os.getcwd() + m + '.pdmodel', os.getcwd() + m + '.pdiparams') for m in ultra_models
    ]

    # model = dynaconf_config_paddle.ULTRA_MODEL
    # model_file = dynaconf_config_paddle.ULTRA_MODEL_FILE
    # model_file = os.getcwd() + model_file
    # params_file = dynaconf_config_paddle.ULTRA_PARAMS_FILE
    # params_file = os.getcwd() + params_file

    model_file_0, params_file_0 = ultra_models_params[0]
    model_file_1, params_file_1 = ultra_models_params[1]
    model_file_2, params_file_2 = ultra_models_params[2]
    model_file_3, params_file_3 = ultra_models_params[3]
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=False)

    # params for preprocess
    parser.add_argument("--resize_short", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # params for predict
    parser.add_argument("--model_file_0", type=str, default=model_file_0)
    parser.add_argument("--params_file_0", type=str, default=params_file_0)

    parser.add_argument("--model_file_1", type=str, default=model_file_1)
    parser.add_argument("--params_file_1", type=str, default=params_file_1)

    parser.add_argument("--model_file_2", type=str, default=model_file_2)
    parser.add_argument("--params_file_2", type=str, default=params_file_2)

    parser.add_argument("--model_file_3", type=str, default=model_file_3)
    parser.add_argument("--params_file_3", type=str, default=params_file_3)

    # else:
    parser.add_argument("--top_k", type=int, default=2)

    wash_model = dynaconf_config_paddle.WASH_MODEL
    wash_model_file = dynaconf_config_paddle.WASH_MODEL_FILE
    wash_model_file = os.getcwd() + wash_model_file
    wash_params_file = dynaconf_config_paddle.WASH_PARAMS_FILE
    wash_params_file = os.getcwd() + wash_params_file

    parser.add_argument("--wash_model_file", type=str, default=wash_model_file)
    parser.add_argument("--wash_params_file", type=str, default=wash_params_file)
    # parser.add_argument("--model", type=str, default=model)

    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_profile", type=str2bool, default=False)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_num_threads", type=int, default=10)
    parser.add_argument("--hubserving", type=str2bool, default=False)

    # params for infer
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--class_num", type=int, default=8)
    parser.add_argument(
        "--load_static_weights",
        type=str2bool,
        default=False,
        help='Whether to load the pretrained weights saved in static mode')

    # parameters for pre-label the images
    parser.add_argument(
        "--pre_label_image",
        type=str2bool,
        default=False,
        help="Whether to pre-label the images using the loaded weights")

    parser.add_argument("--pre_label_out_idr", type=str, default=None)

    parser.add_argument('-s', '--start-date', type=str, help='start date yyyy-mm-dd')
    parser.add_argument('-e', '--end-date', type=str, help='end date yyyy-mm-dd')
    parser.add_argument('-d', '--debug', required=False, help='Enable debug output',
                        dest='debug', action='store_true', default=False)
    parser.add_argument('-c', '--videoAnalysis', required=False, help='Consume video_analysis message from video queue',
                        action='store_true', default=False)

    # parser.add_argument('-p', '--audioJobProducer', required=False, help='produce message to audio queue',
    #                     action='store_true', default=False)
    parser.add_argument('-p', '--videoJobProducer', required=False, help='produce message to audio queue',
                        action='store_true', default=False)

    parser.add_argument('-a', '--audioAnalysis', required=False, help='Consume audio_analysis message from audio queue',
                        action='store_true', default=False)

    parser.add_argument('-t', '--tag-only', required=False,
                        help='only tag video, not calculate start/end time of event',
                        action='store_true', default=False)
    parser.add_argument('-l', '--log-path', required=False, help='log file path',
                        dest='log_path', default='/var/log/video_tagging/logs', type=str)
    parser.add_argument('-dev', '--device', required=False, help='device, cpu or cuda',
                        default='cpu', type=str)

    return parser.parse_args()

def create_predictors(args):
    config_ul_0 = Config(args.model_file_0, args.params_file_0)
    config_ul_1 = Config(args.model_file_1, args.params_file_1)
    config_ul_2 = Config(args.model_file_2, args.params_file_2)
    config_ul_3 = Config(args.model_file_3, args.params_file_3)
    config_wash = Config(args.wash_model_file, args.wash_params_file)
    ultra_model_0 = create_paddle_predictor(args,config_ul_0)
    ultra_model_1 = create_paddle_predictor(args,config_ul_1)
    ultra_model_2 = create_paddle_predictor(args,config_ul_2)
    ultra_model_3 = create_paddle_predictor(args,config_ul_3)
    wash_model = create_paddle_predictor(args,config_wash)
    return ultra_model_0,ultra_model_1,ultra_model_2,ultra_model_3,wash_model

def create_paddle_predictor(args,config):
    # config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
    config.set_cpu_math_library_num_threads(args.cpu_num_threads)

    if args.enable_profile:
        config.enable_profile()
    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=Config.Precision.Half
            if args.use_fp16 else Config.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


def preprocess(img, args):
    resize_op = ResizeImage(resize_short=args.resize_short)
    img = resize_op(img)
    crop_op = CropImage(size=(args.resize, args.resize))
    img = crop_op(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0
        normalize_op = NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        img = normalize_op(img)
    tensor_op = ToTensor()
    img = tensor_op(img)
    return img


def postprocess(output, topk):
    output = output.flatten()
    classes = np.argpartition(output, -topk)[-topk:]
    classes = classes[np.argsort(-output[classes])]
    scores = output[classes]
    return classes, scores


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp', 'PNG']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


class Base64ToCV2(object):
    def __init__(self):
        pass

    def __call__(self, b64str):
        import base64
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)[:, :, ::-1]
        return data
