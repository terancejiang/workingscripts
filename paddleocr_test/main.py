# from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# 模型路径下必须含有model和params文件
# ocr = PaddleOCR(det_model_dir='/home/jy/Projects/workingscripts/paddleocr_test/ocr_models/sast',
#                 rec_model_dir='//home/jy/Projects/workingscripts/paddleocr_test/ocr_models/ch_ppocr_server_v2.0_rec_infer',
#                 rec_char_dict_path='/home/jy/Projects/workingscripts/paddleocr_test/ocr_models/ppocr_keys_v1.txt',
#                 cls_model_dir='/home/jy/Projects/workingscripts/paddleocr_test/ocr_models/ch_ppocr_mobile_v2.0_cls_infer',
#                 use_angle_cls=True, use_gpu=False)
img_path = '/home/jy/Projects/workingscripts/paddleocr_test/2021-04-09_11-32.jpg'
# result = ocr.ocr(img_path, cls=True)
# for line in result:
#     print(line)
#
# # 显示结果
image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/jy/Projects/MeterProject/paddleocr/doc/fonts/simfang.ttf')

im_show = Image.fromarray(image)
im_show.save('result.jpg')



