import os
import cv2
import torch
import shutil
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=False)
#
checkpoint = torch.hub.load_state_dict_from_url('/home/jy/projects/video_tagging/video_tagging/video_tagging/model_files/yolo5l_ultrasonic_detection_4classes.pt', map_location="cpu")

state_dict = checkpoint['model'].float().state_dict()  # to FP32
state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}
model.load_state_dict(state_dict,strict=False)

model = model.autoshape()
#
# # Unwrap the DistributedDataParallel module
# # module.layer -> layer
# # state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
#
# # Apply the state dict to the model

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='/home/jy/projects/video_tagging/video_tagging/video_tagging/model_files/yolo5l_ultrasonic_detection_4classes.pt')  # custom model
with open('/home/jy/projects/workingscripts/yolomodeltesting/执行结果1 (2).txt','r') as txt:
    lines = txt.readlines()
for f in lines:
    f = f.replace('"','')
    f = f.rstrip()
    print(f)
    cap = cv2.VideoCapture(f)
    ret, frame = cap.read()
    if frame is not None:

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = model(frame)
        results.save()
        shutil.copyfile('results/image0.jpg',
                        'results/' + '%s-%s.jpg' % (
                            str('old'), round(time.time() * 1000)))


