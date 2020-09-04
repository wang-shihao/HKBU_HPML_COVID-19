import json
import random


def balance(dict, min_value):
    keys = list(dict.keys())
    new_keys = set()
    while len(new_keys) < min_value:
        new_keys.add(random.choice(keys))

    for k in keys:
        if k not in new_keys:
            dict.pop(k)


    return dict



filename = './nii_png_train.json'
train_jfile = open(filename)
train = json.load(train_jfile)

cls_to_label = {
    #png
    'CP': 0, 'NCP': 1, 'Normal': 2,
    #nii
    'CT-0': 2, 'CT-1': 1, 'CT-2': 1, 'CT-3': 1, 'CT-4': 1
}

cls_dict = {0:{}, 1:{}, 2:{}}

for cls in train.keys():
    for patient,v in train[cls].items():
        cls_dict[cls_to_label[cls]][patient] = v
    
cls_counts = [len(cls_dict[0].keys()), len(cls_dict[1].keys()), len(cls_dict[2].keys())]
print(cls_counts)
min_value = min(i for i in cls_counts if i > 0)
print(min_value)

for cls in cls_dict.keys():
    if len(cls_dict[cls].keys()) > min_value:
        cls_dict[cls] = balance(cls_dict[cls], min_value)


cls_counts = [len(cls_dict[0].keys()), len(cls_dict[1].keys()), len(cls_dict[2].keys())]
print(cls_counts)
min_value = min(i for i in cls_counts if i > 0)
print(min_value)

cls_dict['Normal'] = cls_dict.pop(2)
cls_dict['NCP'] = cls_dict.pop(1)
cls_dict['CP'] = cls_dict.pop(0)

outfile = open(filename.replace('.json', '_balanced.json'), 'w')

#json.dump(cls_dict, outfile, indent=4)

