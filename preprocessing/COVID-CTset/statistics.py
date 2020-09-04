from matplotlib import pyplot as plt
import os
import json


info = {'min':9999, 'max':0, 'min_dir':'', 'max_dir':'','mean':0,'slice_num_list':[]}
for root, child, files in os.walk('./COVID-CTset/'):
    if files:
        num = len(files)
        if num < info['min']:
            min_ = num
            info['min']=min_
            info['min_dir']=root
        if num > info['max']:
            max_ = num
            info['max']=max_
            info['max_dir']=root
        info['slice_num_list'].append(num)

slice_num_list = info['slice_num_list']
counts = {}
for k in slice_num_list:
    if k not in counts:
        counts[k] = 1
    else:
        counts[k] += 1
x = sorted(counts)
y = [counts[i] for i in x]
plt.plot(x, y)
plt.pause(-1)

info = {'min': 8,
 'max': 406,
 'min_dir': './COVID-CTset/normal4\\patient222\\SR_3',
 'max_dir': './COVID-CTset/covid1\\patient78\\SR_4',
 'mean': 47,
 'slice_num_list': []
}