from collections import defaultdict
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


def get_location_class_mapper():
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


def get_country_class_mapper():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items_all = []
    items_1 = load_csv_annotations(fp=os.path.join(root,'valid_split.csv'))
    items_all.extend(items_1)
    items_2 = load_csv_annotations(fp=os.path.join(root,'train_split.csv'))
    items_all.extend(items_2)
    cty2classes = defaultdict(set)
    for it in items_all:
        cty2classes[it['country']].add(it['class_id'])
    return cty2classes


def get_items():
    root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'
    items = load_csv_annotations(fp=os.path.join(root, 'SnakeCLEF2022-TestMetadata.csv'), row_length=5)
    return items


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


# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_2/snake_v2/result_snakeclef2022test_v3.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_2/snake_v2_meta_ovs/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_2/snake_v2_meta_ovs_trainval/result_snakeclef2022test_e222.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_2/snake_v2_meta/result_snakeclef2022test_e300.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_0/snake_v0_meta_ovs_e120_softmax/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_0/snake_v0_meta_ovs_e120_seesaw/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_meta2438_e124/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_meta2438_e104/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_meta2438_e164/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_meta2438_e144/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_sslv1_meta2438_ovs_tv_lr0p1_e104/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_sslv1_meta2438_ovs_tv_e104/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_inat21_meta2438_ovs_tv_e104/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_ssl_meta2438_ovs_tv_e49/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_inat21_meta2438_ovs_tv_e104/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_sslv1_meta2438_ovs_tv_lr0p1_e164/result_snakeclef2022test.tc'
# model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_sslv1_meta2438_ovs_tv_e144/result_snakeclef2022test.tc'
model_result_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_sslv1_meta2438_ovs_tv_e84/result_snakeclef2022test.tc'
snake_test_result = torch.load(model_result_file)
item_list = get_items()
classes, class_to_idx = get_class_and_idx_mapper()
loc2classes = get_location_class_mapper()
cty2classes = get_country_class_mapper()
print(len(classes), len(class_to_idx), len(loc2classes))


root = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/datasets/snakeclef2022'
classes, endemics, location_codes, class_to_idx, cls_hist, countries, classes_dist = load_statistics_from_file(root)
sum_dist = sum(classes_dist)
cls_dist_ts = torch.Tensor([i/sum_dist for i in classes_dist])
tao = 0.7
cls_dist_ts_log = tao * torch.log(cls_dist_ts)


observation_to_classes = defaultdict(list)
print(len(snake_test_result))
for idx_st, st in enumerate(snake_test_result):
    v1, v2 = st
    data_item = item_list[int(v1.detach().cpu().numpy())]
    sm_scores = torch.nn.Softmax(dim=0)(v2-cls_dist_ts_log).detach().cpu().numpy()
    # idx_max = np.argmax(sm_scores)
    # cls_max = classes[idx_max]
    # score_max = sm_scores[idx_max]
    idx_sort = np.argsort(sm_scores)[::-1]
    for idx_max in idx_sort:
        cls_max = classes[idx_max]
        if cls_max in loc2classes[data_item['code']]:
            score_max = sm_scores[idx_max]
            break
    observation_to_classes[data_item['observation_id']].append((cls_max, score_max))
print(len(observation_to_classes))


result_list = []
for obv_id, class_id_scores in observation_to_classes.items():
    sorted_scores = sorted(class_id_scores, key=lambda x: x[1], reverse=True)
    select_cls_id_according_to_max_score = sorted_scores[0][0]
    # Get most.
    dist = defaultdict(int)
    for p in class_id_scores:
        dist[p[0]] += 1
    dist_list = []
    for k, v in dist.items():
        dist_list.append((k, v))
    new_dist_list = sorted(dist_list, key=lambda x: x[1], reverse=True)
    select_cls_id_according_to_the_most = new_dist_list[0][0]
    # Deal with conflict.
    if len(new_dist_list) > 1:
        if new_dist_list[0][1] == new_dist_list[1][1]:
            selected_cls_id_final = select_cls_id_according_to_max_score
        else:
            # selected_cls_id_final = select_cls_id_according_to_the_most
            selected_cls_id_final = select_cls_id_according_to_max_score
    else:
        assert select_cls_id_according_to_max_score == select_cls_id_according_to_the_most
        selected_cls_id_final = select_cls_id_according_to_max_score
    # print(selected_cls_id_final, class_id_scores)
    result_list.append((obv_id, selected_cls_id_final))
print(len(result_list))


output_file = model_result_file + '.result.csv'
with open(output_file, 'w') as wf:
    wf.write('ObservationId,class_id\n')
    for it in result_list:
        wf.write('%s,%s\n' % (it[0], it[1]))
