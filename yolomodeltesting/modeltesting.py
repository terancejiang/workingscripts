import os
import cv2
import torch
import shutil
import time

# model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=False)
#
# model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='')
#
# state_dict = checkpoint['model'].float().state_dict()  # to FP32
# state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}
# model.load_state_dict(state_dict,strict=False)
#
# model = model.autoshape()
#
# # Unwrap the DistributedDataParallel module
# # module.layer -> layer
# # state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
#
# # Apply the state dict to the model

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='/media/jy/740535b3-57f4-42bf-9664-ce7b4ff7af2e/projects/workingscripts/yolomodeltesting/yolo5l_ultrasonic_detection_4classes.pt')  # custom model
# with open('/home/jy/projects/workingscripts/yolomodeltesting/执行结果1 (2).txt','r') as txt:
#     lines = txt.readlines()
# for f in lines:
#     f = f.replace('"','')
#     f = f.rstrip()
#     print(f)
#     cap = cv2.VideoCapture(f)
#     ret, frame = cap.read()
#     if frame is not None:
#
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         results = model(frame)
#         results.save()
#         shutil.copyfile('results/image0.jpg',
#                         'results/' + '%s-%s.jpg' % (
#                             str('old'), round(time.time() * 1000)))
inf_size = 16
files_t = []
for root ,dirs, files in os.walk('/media/jy/740535b3-57f4-42bf-9664-ce7b4ff7af2e/projects/workingscripts/autocollection/ultra_obj_selected'):
    files = [os.path.join(root,f) for f in files]
    files_t += files
total_num = len(files_t)
for i in range(0, total_num, inf_size ):
    im_size = min(total_num, i+inf_size)
    images = list()
    for j in range(i, im_size):
        images.append(files_t[j])
    imgs = list()
    for img in images:
        imgs.append(cv2.imread(img))
    results = model(imgs)
    print(results)


# model.conf = 0.35
# class_name = ['0','1','2','3']
#
#
# img = img[:, :, ::-1]
#
# results = self.model(img)