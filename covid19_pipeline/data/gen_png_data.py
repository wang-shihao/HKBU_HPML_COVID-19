import json
import re
import os
import random

root_dir = '/home/datasets/MosMedData/COVID19_1110/pngs'
train = {}
test = {}

for cls in os.listdir(root_dir):
#for label in os.listdir(root_dir):
#    if label == 'CT-0':
#        cls = 'Normal'
#    else:
#        cls = 'NCP'
    if cls.startswith('.'):
    #if label.startswith('.'):
        continue
    #cls_dir = os.path.join(root_dir, label)
    cls_dir = os.path.join(root_dir, cls)
    train[cls] = {}
    test[cls] = {}
    all_imgs = set(os.listdir(cls_dir))
    size = len(all_imgs)
    train_set = set()
    test_set = set()
    while len(train_set) < size*0.7:
        train_set.add(random.choice(list(all_imgs)))

    test_set = all_imgs - train_set

    for name in train_set:
        pid = name
        train[cls][pid] = {}
        sid = pid
        train[cls][pid][sid] = sorted(os.listdir(os.path.join(cls_dir, pid, pid)))

    for name in test_set:
        pid = name
        test[cls][pid] = {}
        sid = pid
        test[cls][pid][sid] = sorted(os.listdir(os.path.join(cls_dir, pid, pid)))


train_file = open('nii_png_train.json', 'w')
test_file = open('nii_png_test.json', 'w')

json.dump(train, train_file, indent=4)
json.dump(test, test_file, indent=4)



