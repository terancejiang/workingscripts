from classification_conf_mat.pretty_print import plot_confusion_matrix_from_data
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def create_match():
    classes = ['0_close',  '0_open',  '1_close',  '1_open',  '2_close',  '2_open',  '3_close',  '3_open']
    flist=[]
    out = []
    for root_src,dir_src,files_src in os.walk('/home/jy/projects/imgs/dataprepare/b1-5_classify/batch5_ultrasonic_classification/b1_5_val'):
        for d in dir_src:
            files = os.listdir(os.path.join(root_src,d))
            for f in files:
                flist.append('{} {}'.format(f, classes.index(d)))

    for root_src,dir_src,files_src in os.walk('/home/jy/projects/imgs/dataprepare/b1-5_classify/batch5_ultrasonic_classification/inference'):
        for d in tqdm(dir_src):
            files = os.listdir(os.path.join(root_src,d))
            for f in files:
                for res in flist:
                    fname,y_lanbel = res.split(' ')
                    if f == fname:
                        out.append('{} {} {}'.format(f,y_lanbel,classes.index(d)))
                        with open('out.txt','a')as outtxt:
                            outtxt.writelines('{} {} {}'.format(f,y_lanbel,classes.index(d))+'\n')

                flist.append('{} {}'.format(f, classes.index(d)))
    print(out)

with open('out.txt','r') as txt:
    results = txt.readlines()
label = []
predict = []
for r in results:
    filename,l,p = r.split(' ')
    p = p[0]
    label.append(l)
    predict.append(p)

confm = confusion_matrix(label, predict)
print(confm)
columns = []
annot = True
cmap = 'Oranges'
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
#size::
fz = 12
figsize = [9,9]
if len(predict) > 10:
    fz=9
    figsize=[14,14]
plot_confusion_matrix_from_data(label, predict, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
# array = np.array( [[13,  0,  1,  0,  2,  0],
#                        [ 0, 50,  2,  0, 10,  0],
#                        [ 0, 13, 16,  0,  0,  3],
#                        [ 0,  0,  0, 13,  1,  0],
#                        [ 0, 40,  0,  1, 15,  0],
#                        [ 0,  0,  0,  0,  0, 20]])
#     #get pandas dataframe
#     df_cm = pd.DataFrame(array, index=range(1,7), columns=range(1,7))
#     #colormap: see this and choose your more dear
#     cmap = 'PuRd'
#     pretty_plot_confusion_matrix(df_cm, cmap=cmap)