import json
import os

nii_png_test = open('./nii_png_test.json')
nii_png_train = open('./nii_png_train.json')

combined_train = open('./ct_train.json')
combined_test = open('./ct_test.json')

j_nii_train = json.load(nii_png_train)
j_nii_test = json.load(nii_png_test)
j_mix_train = json.load(combined_train)
j_mix_test = json.load(combined_test)

for cls, val in j_nii_test.items():
    if cls == 'CT-0':
        for pid, pval in val.items():
            j_mix_test['Normal'][str(int(pid)*10000)] = pval
    else:
        for pid, pval in val.items():
            j_mix_test['NCP'][str(int(pid)*10000)] = pval

for cls, val in j_nii_train.items():
    if cls == 'CT-0':
        for pid, pval in val.items():
            j_mix_train['Normal'][str(int(pid)*10000)] = pval
    else:
        for pid, pval in val.items():
            j_mix_train['NCP'][str(int(pid)*10000)] = pval

final_train = open('./ct_mixed_train.json', 'w')
train_str = json.dump(j_mix_train, final_train, indent=4)
final_test = open('./ct_mixed_test.json', 'w')
test_str = json.dump(j_mix_test, final_test, indent=4)
