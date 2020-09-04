import json

trainin = json.load(open('covid_ctset/train1.json'))
testin = json.load(open('covid_ctset/test1.json'))

niiin = json.load(open('mosmed/nii_train.json'))

for c,v in trainin.items():
    if c == 'normal':
        if c not in niiin.keys():
            niiin[c] = {}

        for c1,v1 in v.items():
            niiin[c][c1] = v1

for c,v in testin.items():
    if c == 'normal':
        if c not in niiin.keys():
            niiin[c] = {}

        for c1,v1 in v.items():
            niiin[c][c1] = v1

outfile = open('mosmed/mixed_nii_train.json', 'w')
json.dump(niiin, outfile, indent=4)

