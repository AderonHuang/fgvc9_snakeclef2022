from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
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


def get_items():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items = load_csv_annotations(fp=os.path.join(root,'valid_split.csv'))
    return items


def get_location_class_mapper1():
    file_path = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/SnakeCLEF2022-ISOxSpeciesMapping.csv'
    lines = []
    with open(file_path, 'r') as rf:
        cr = csv.reader(rf)
        for row in cr:
            if len(row) < 219:
                continue
            if row[0] == 'binomial':
                names = row
                continue
            lines.append({name : row[idx] for idx, name in enumerate(names)})
    loc2names = defaultdict(list)
    for it in lines:
        for k, v in it.items():
            if k != 'binomial' and v == '1':
                loc2names[k].append(it['binomial'])
    print(len(loc2names))


def get_location_class_mapper2():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items_all = []
    items_1 = load_csv_annotations(fp=os.path.join(root,'valid_split.csv'))
    items_all.extend(items_1)
    items_2 = load_csv_annotations(fp=os.path.join(root,'train_split.csv'))
    items_all.extend(items_2)
    loc2classes = defaultdict(set)
    for it in items_all:
        loc2classes[it['code']].add(it['class_id'])
    return loc2classes

def load_statistics_from_file(root):
    import json
    file_path = os.path.join(root, 'statistics_on_train_test.json')
    ret = json.loads(open(file_path, 'r').readline().strip())
    classes = ret['classes']
    endemics = ret['endemics']
    location_codes = ret['location_codes']
    class_to_idx = ret['class_to_idx']
    cls_hist = ret['cls_hist']
    countries = ret['countries']
    classes_dist = ret['classes_dist']
    return classes, endemics, location_codes, class_to_idx, cls_hist, countries, classes_dist


def evaluate(model_result_file, use_location_filter=True, tao=0):
    print(model_result_file)
    snake_test_result = torch.load(model_result_file)
    item_list = get_items()
    classes, class_to_idx = get_class_and_idx_mapper()
    loc2classes = get_location_class_mapper2()
    # print(len(classes), len(class_to_idx), len(loc2classes), len(snake_test_result))

    num_correct = 0
    y_true_list = []
    y_pred_list = []
    root = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/datasets/snakeclef2022'
    classes, endemics, location_codes, class_to_idx, cls_hist, countries, classes_dist = load_statistics_from_file(root)
    sum_dist = sum(classes_dist)
    cls_dist_ts = torch.Tensor([i/sum_dist for i in classes_dist])
    cls_dist_ts_log = tao * torch.log(cls_dist_ts)

    for idx_st, st in enumerate(snake_test_result):
        v1, v2 = st
        data_item = item_list[int(v1.detach().cpu().numpy())]
        sm_scores = torch.nn.Softmax(dim=0)(v2-cls_dist_ts_log).detach().cpu().numpy()
        if use_location_filter:
            idx_sort = np.argsort(sm_scores)[::-1]
            for idx_max in idx_sort:
                cls_max = classes[idx_max]
                if cls_max in loc2classes[data_item['code']]:
                    break
        else:
            idx_max = np.argmax(sm_scores)
            cls_max = classes[idx_max]
        if cls_max == data_item['class_id']:
            num_correct += 1
        y_true_list.append(class_to_idx[data_item['class_id']])
        y_pred_list.append(class_to_idx[cls_max])

    if use_location_filter:
        print('loc_filter: 1')
    else:
        print('loc_filter: 0')

    # print('acc: ', num_correct / len(snake_test_result))
    print('f1_score: %f, accuracy_score: %f\n' 
            % (f1_score(y_true_list, y_pred_list, average='macro'), accuracy_score(y_true_list, y_pred_list)))


for file_path in [
    # 'output/MetaFG_0/snake_v0/result_snakeclef2022valid.tc',
    # 'output/MetaFG_meta_0/snake_v0_meta/result_snakeclef2022valid_e300.tc',
    # 'output/MetaFG_meta_0/snake_v0_meta_ovs/result_snakeclef2022valid_e300.tc',
    # 'output/MetaFG_2/snake_v2/result_snakeclef2022valid_inat21.tc',
    # 'output/MetaFG_2/snake_v2/result_snakeclef2022valid_img21k.tc',
    # 'output/MetaFG_meta_2/snake_v2_meta/result_snakeclef2022valid_e300.tc',
    # 'output/MetaFG_meta_2/snake_v2_meta_e114/result_snakeclef2022valid.tc',
    # 'output/MetaFG_meta_2/snake_v2_meta_ovs/result_snakeclef2022valid_e82.tc',
    # 'output/MetaFG_meta_2/snake_v2_meta/result_snakeclef2022valid_e300.tc',
    # 'output/MetaFG_meta_0/snake_v0_meta_ovs_e120_softmax/result_snakeclef2022valid.tc',
    # 'output/MetaFG_meta_0/snake_v0_meta_ovs_e120_seesaw/result_snakeclef2022valid.tc',
    'output/MetaFG_meta_2/snake_v2_meta2438_e124/result_snakeclef2022valid.tc',
]:
    output_root_dir = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne'
    result_file = os.path.join(output_root_dir, file_path)
    # evaluate(model_result_file=result_file, use_location_filter=False)
    for tao in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    # for tao in [0.3, 0.4, 0.5]:
        print(tao)
        evaluate(model_result_file=result_file, use_location_filter=True, tao=tao)
