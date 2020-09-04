import os
import json
import random

def split_dataset(CT_file, suffix=''):
    with open(CT_file, 'r') as f:
        CT = json.load(f)
    train_ratio = 0.6
    val_ratio = 0.2
    ct_tr = {'CP': {}, 'NCP':{}, 'Normal':{}}
    ct_va = {'CP': {}, 'NCP':{}, 'Normal':{}}
    ct_te = {'CP': {}, 'NCP':{}, 'Normal':{}}
    num_info = {
        'All': {'NCP':0, 'CP':0, 'Normal':0},
        'Train': {'NCP':0, 'CP':0, 'Normal':0},
        'Val': {'NCP':0, 'CP':0, 'Normal':0},
        'Test': {'NCP':0, 'CP':0, 'Normal':0},
    }
    for cls_name in CT:
        scan_num = 0
        for pid in CT[cls_name]:
            num_info['All'][cls_name] += len(CT[cls_name][pid])
            scan_num += len(CT[cls_name][pid])
        print(f"{cls_name} has {scan_num} scans.")
        tr_num = 0
        va_num = 0
        te_num = 0
        thelist = CT[cls_name]
        for pid in range(len(CT[cls_name])):
            theone, _ = random.choice(list(thelist.items()))
            #print(theone)
            #print(len(CT[cls_name][theone]))
            #print(theone)
            if tr_num < int(num_info['All'][cls_name]*train_ratio):
                #print(cls_name)
                #print(theone)
                tr_num += len(CT[cls_name][theone])
                ct_tr[cls_name][theone] = CT[cls_name][theone]
                thelist.pop(theone)
            elif va_num < int(num_info['All'][cls_name]*val_ratio):
                va_num += len(CT[cls_name][theone])
                ct_va[cls_name][theone] = CT[cls_name][theone]
                thelist.pop(theone)
            else:
                te_num += len(CT[cls_name][theone])
                ct_te[cls_name][theone] = CT[cls_name][theone]
                thelist.pop(theone)
        num_info['Train'][cls_name] = tr_num
        num_info['Val'][cls_name] = va_num
        num_info['Test'][cls_name] = te_num
    print(num_info)

    #for cls_name in CT:
    #    tr_num = 0
    #    va_num = 0
    #    te_num = 0
    #    for pid in CT[cls_name]:
    #        if tr_num < int(num_info['All'][cls_name]*train_ratio):
    #            tr_num += len(CT[cls_name][pid])
    #            ct_tr[cls_name][pid] = CT[cls_name][pid]
    #        elif tr_num < int(num_info['All'][cls_name]*val_ratio):
    #            tr_num += len(CT[cls_name][pid])
    #            ct_tr[cls_name][pid] = CT[cls_name][pid]
    #        else:
    #            te_num += len(CT[cls_name][pid])
    #            ct_te[cls_name][pid] = CT[cls_name][pid]
    #    num_info['Train'][cls_name] = tr_num
    #    num_info['Val'][cls_name] = va_num
    #    num_info['Test'][cls_name] = te_num
    #print(num_info)
    for cls_name in CT:
        print(f"{cls_name} All={len(CT[cls_name])} Train={len(ct_tr[cls_name])} Test={len(ct_te[cls_name])}")
    with open(f'ct{suffix}_train.json', 'w') as f:
        json.dump(ct_tr, f, indent=4)
    with open(f'ct{suffix}_val.json', 'w') as f:
        json.dump(ct_va, f, indent=4)
    with open(f'ct{suffix}_test.json', 'w') as f:
        json.dump(ct_te, f, indent=4)
    with open(f'ct{suffix}_num_info.json', 'w') as f:
        json.dump(num_info, f, indent=4)

if __name__ == '__main__':
    split_dataset('CT_data.json', '')
    #split_dataset('CT_cleaned_data.json', '_cleaned')
