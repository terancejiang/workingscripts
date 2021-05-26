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

import numpy as np
import cv2
import time

import sys

from dynaconf import settings

dynaconf_config_others = settings.get('others', {})
debug = dynaconf_config_others.DEBUG

sys.path.insert(0, ".")
import autocollection.paddle_infer.paddle_utils as utils


def predict(args, predictor,class_names, img='', topk=1):
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    classes_out = []
    scores_out = []

    open_cv_image = np.array(img)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()
    img = img[:, :, ::-1]

    inputs = utils.preprocess(img, args)
    inputs = np.expand_dims(
        inputs, axis=0).repeat(
        args.batch_size, axis=0).copy()
    input_tensor.copy_from_cpu(inputs)

    predictor.run()

    output = output_tensor.copy_to_cpu()
    classes, scores = utils.postprocess(output, topk)

    # dynaconf_config_paddle = settings.get('paddle', {})
    # if type == 'ultrasonic':
    #     classifiy_classes = dynaconf_config_paddle.PADDLE_CLASSES_ULTRASONIC
    # else:
    #     classifiy_classes = dynaconf_config_paddle.PADDLE_CLASSES_WASH
    classes = [class_names[cls] for cls in classes]

    # added
    if debug:
        # print("Current image file: {}".format(img_name))
        print("\ttop-1 class: {0}".format(classes[0]))
        print("\ttop-1 score: {0}".format(scores[0]))

    return classes, scores
    #
    # test_num = 500
    # test_time = 0.0
    # if not args.enable_benchmark:
    #     # for PaddleHubServing
    #     if args.hubserving:
    #         img_list = [args.image_file]
    #     # for predict only
    #     else:
    #         img_list = get_image_list(args.image_file)
    #
    #     for idx, img_name in enumerate(img_list):
    #         if not args.hubserving:
    #             # added
    #             if img:
    #                 img = img[:, :, ::-1]
    #             # end
    #
    #             else:
    #                 img = cv2.imread(img_name)[:, :, ::-1]
    #             assert img is not None, "Error in loading image: {}".format(
    #                 img_name)
    #         else:
    #             img = img_name
    #         inputs = utils.preprocess(img, args)
    #         inputs = np.expand_dims(
    #             inputs, axis=0).repeat(
    #             args.batch_size, axis=0).copy()
    #         input_tensor.copy_from_cpu(inputs)
    #
    #         predictor.run()
    #
    #         output = output_tensor.copy_to_cpu()
    #         classes, scores = utils.postprocess(output, args)
    #         if args.hubserving:
    #             return classes, scores
    #         # added
    #         if debug:
    #             print("Current image file: {}".format(img_name))
    #             print("\ttop-1 class: {0}".format(classes[0]))
    #             print("\ttop-1 score: {0}".format(scores[0]))
    #
    #         # added
    #         return classes, scores
    # else:
    #     for i in range(0, test_num + 10):
    #         inputs = np.random.rand(args.batch_size, 3, 224,
    #                                 224).astype(np.float32)
    #         start_time = time.time()
    #         input_tensor.copy_from_cpu(inputs)
    #
    #         predictor.run()
    #
    #         output = output_tensor.copy_to_cpu()
    #         output = output.flatten()
    #         if i >= 10:
    #             test_time += time.time() - start_time
    #         time.sleep(0.01)  # sleep for T4 GPU
    #
    #     fp_message = "FP16" if args.use_fp16 else "FP32"
    #     trt_msg = "using tensorrt" if args.use_tensorrt else "not using tensorrt"
    #     print("{0}\t{1}\t{2}\tbatch size: {3}\ttime(ms): {4}".format(
    #         args.model, trt_msg, fp_message, args.batch_size, 1000 * test_time
    #                                                           / test_num))


if __name__ == "__main__":
    args = utils.parse_args()
    from paddle.inference import Config
    from autocollection.paddle_infer.paddle_utils import create_paddle_predictor
    from shutil import copyfile
    from tqdm import tqdm
    config_ul_0 = Config(args.model_file_0, args.params_file_0)
    config_ul_1 = Config(args.model_file_1, args.params_file_1)
    config_ul_2 = Config(args.model_file_2, args.params_file_2)
    config_ul_3 = Config(args.model_file_3, args.params_file_3)
    config_wash = Config(args.wash_model_file, args.wash_params_file)
    ultra_model_0 = create_paddle_predictor(args, config_ul_0)
    ultra_model_1 = create_paddle_predictor(args, config_ul_1)
    ultra_model_2 = create_paddle_predictor(args, config_ul_2)
    ultra_model_3 = create_paddle_predictor(args, config_ul_3)
    wash_model = create_paddle_predictor(args, config_wash)
    predictor = utils.create_predictors(args)
    for root,dirs,files in os.walk('ultra_obj_selected_crop'):
        for d in dirs:
            filess = os.listdir(os.path.join(root,d))
            for file in tqdm(filess):
                fname = file
                file = os.path.join(root,d,file)
                img = cv2.imread(file)
                try:
                    classes_ultra_pred, scores_ultra_pred = predict(args, ultra_model_1,
                                                                    ['1_close', '1_open_obj', '1_open_nobj'], img, 1)
                except Exception as e:
                    print(e)
                # print(classes_ultra_pred)
                # print(scores_ultra_pred)
                dst = os.path.join('ultra_obj_selected_crop_0.7',d)
                fdst = os.path.join('ultra_obj_selected_crop_le0.7',d)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                if not os.path.exists(fdst):
                    os.makedirs(fdst)
                fname = '{}_{}_{}'.format(fname.split('-')[0],str(scores_ultra_pred)[1:5], fname.split('-')[-1])
                dst = os.path.join(dst,fname)
                fdst = os.path.join(fdst,fname)
                if scores_ultra_pred >= 0.7:
                    copyfile(file,dst)
                else:
                    copyfile(file, fdst)
        break
    img = cv2.imread('/hdd/projects/workingscripts/autocollection/ultra_close/0/C63088112/C63088112-1617395889591.jpg')

    # classes_ultra_pred, scores_ultra_pred = predict(args, ultra_model_1,
    #                                                 ['1_close','1_open_obj','1_open_nobj'], img, 1)

