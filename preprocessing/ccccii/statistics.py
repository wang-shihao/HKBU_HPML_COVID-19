import json

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def patient_num(data):
    tmp = {}
    for c in data:
        tmp[c] = len(data[c])
    return tmp

def scan_num(data):
    tmp = {}
    for c in data:
        tmp[c] = 0
        for pid in data[c]:
            tmp[c] += len(data[c][pid])
    return tmp

def slice_num(data):
    tmp = {}
    for c in data:
        tmp[c] = 0
        for pid in data[c]:
            for scandid in data[c][pid]:
                tmp[c] += len(data[c][pid][scandid])
    return tmp

def slice_info(data):
    smin = 99999
    smax = 0
    smean = 0
    sall = 0
    count = 0
    for c in data:
        for pid in data[c]:
            for scandid in data[c][pid]:
                slice_num =  len(data[c][pid][scandid])
                if slice_num > smax: smax = slice_num
                if slice_num < smin: smin = slice_num
                sall += slice_num
                count += 1
    smean = sall / count
    result = {'min':smin, 'max':smax, 'mean':smean, 'all':sall}
    print(result,count)
    return result


ctr = load_json('ct_train.json')
ctv = load_json('ct_val.json')
cte = load_json('ct_test.json')
num_info = {}
################################################################
#                patient information                          #
################################################################
tr = patient_num(ctr)
tv = patient_num(ctv)
te = patient_num(cte)
train = {}
total = {}
for key in tr:
    train[key] = tr[key] + tv[key]
    total[key] = train[key] + te[key]
num_info['patient'] ={'train': train, 'test': te, 'total': total}

################################################################
#                scan information                              #
################################################################

tr = scan_num(ctr)
tv = scan_num(ctv)
te = scan_num(cte)
train = {}
total = {}
for key in tr:
    train[key] = tr[key] + tv[key]
    total[key] = train[key] + te[key]

num_info['scan'] ={'train': train, 'test': te, 'total': total}


################################################################
#                slce information                              #
################################################################
tr = slice_num(ctr)
tv = slice_num(ctv)
te = slice_num(cte)
train = {}
total = {}
for key in tr:
    train[key] = tr[key] + tv[key]
    total[key] = train[key] + te[key]

num_info['slice'] ={'train': train, 'test': te, 'total': total}


ct_all = {}
for c in ctr:
    ct_all[c] = {}
    ct_all[c].update(ctr[c])
    ct_all[c].update(ctv[c])
    ct_all[c].update(cte[c])
num_info['slice_info'] = slice_info(ct_all)
print(num_info)

with open('ct_statistics.json', 'w') as f:
    json.dump(num_info, f, indent=4)