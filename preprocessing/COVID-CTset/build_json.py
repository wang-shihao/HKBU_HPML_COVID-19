import pandas as pd
import json
import numpy as np
from collections import OrderedDict


def parse_csv(csv_file):
    data = pd.read_csv(csv_file)
    info = OrderedDict()
    for (filename,c) in data.values:
        if c not in info:
            info[c] = OrderedDict() # { 'covid':{} }

        _, pid, sr, sr_id, img_name = filename.split('_')
        if pid not in info[c]:
            info[c][pid] = OrderedDict() # { 'covid':{ 'patient1':{} } }

        sr = sr + "_" + sr_id
        if sr not in info[c][pid]:
            info[c][pid][sr] = [img_name]
        else:
            info[c][pid][sr].append(img_name)
    
    with open(csv_file.replace('csv','json'), 'w') as f:
        json.dump(info, f, indent=4)
    print(f"{csv_file} done.")
    return info

for i in range(1, 6):
    folder = i
    train = f'./CSV/train{folder}.csv'
    test = f'./CSV/test{folder}.csv'
    val = f'./CSV/validation{folder}.csv'
    for file in [train, test, val]:
        parse_csv(file)