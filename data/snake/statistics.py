from collections import defaultdict
from tqdm import tqdm
import csv
import numpy as np
import os
import torch


def load_csv_annotations(fp, row_length=7):
    lines = []
    with open(fp, 'r') as rf:
        cr = csv.reader(rf)
        for row in cr:
            if len(row) < row_length:
                continue
            if row[0] == 'observation_id':
                names = row
                continue
            lines.append({name : row[idx] for idx, name in enumerate(names)})
    return lines


def get_class_and_idx_mapper():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items = load_csv_annotations(fp=os.path.join(root, 'train_split.csv'))
    classes = set()
    for it in items:
        classes.add(it['class_id'])
    classes = sorted(list(classes))
    class_to_idx = {name:ind for ind,name in enumerate(classes)}
    return classes, class_to_idx


def get_items_all():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items_all = []
    items_1 = load_csv_annotations(fp=os.path.join(root,'valid_split.csv'))
    items_all.extend(items_1)
    items_2 = load_csv_annotations(fp=os.path.join(root,'train_split.csv'))
    items_all.extend(items_2)
    return items_all


def get_percent_index(sorted_hist, percent):
    num_total = sum(sorted_hist)
    num_percent = int(num_total * percent)
    count = 0
    for idx, num in enumerate(sorted_hist):
        count += num
        if count >= num_percent:
            return idx


item_list = get_items_all()
hist = defaultdict(int)

for it in tqdm(item_list):
    hist[it['class_id']] += 1

hist = sorted(hist.values())
print(len(hist), sum(hist))
print(hist[0:10])
print(hist[-10:])

for p in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]:
    idx = get_percent_index(hist, p)
    num = hist[idx]
    print(p, idx, num)

# hist = sorted(hist, reverse=True)
# print(hist[0:10])
# print(hist[-10:])
# for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     idx = get_percent_index(hist, p)
#     num = hist[idx]
#     print(p, idx, num)


idx_ref = get_percent_index(hist, percent=0.1)
num_ref = hist[idx_ref]
print(idx_ref, num_ref)
hist_new = []
for t in hist:
    if t < num_ref:
        t_coef = num_ref // t + 1
        hist_new.append(t_coef * t)
    else:
        hist_new.append(t)


# for idx, t in enumerate(hist):
#     if t < num_ref:
#         print(t, hist_new[idx])
#     else:
#         break


print(sum(hist))
print(sum(hist_new))
