import os
import random

path='D:/VOCtrainval_06-Nov-2007/yoloV3conf\keras-yolo3-master/VOCdevkit/VOC2007/'

trainval_percent = 0.66
train_percent = 0.5

xmlfilepath = path+'Annotations'
txtsavepath = path+'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
 
num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
 
ftrainval = open(path+'ImageSets/Main/trainval.txt', 'w')
ftest = open(path+'ImageSets/Main/test.txt', 'w')
ftrain = open(path+'ImageSets/Main/train.txt', 'w')
fval = open(path+'ImageSets/Main/val.txt', 'w')
 
for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
