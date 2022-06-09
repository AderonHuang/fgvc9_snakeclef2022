import csv
import os


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


root = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/datasets/snakeclef2022'
items_valid = load_csv_annotations(fp=os.path.join(root, 'train_split.csv'))
items_test = load_csv_annotations(fp=os.path.join(root, 'SnakeCLEF2022-TestMetadata.csv'), row_length=5)
print(len(items_test))


file_path = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake_ne/output/MetaFG_meta_2/snake_v2_meta2438_e144/v2_e144_0p7.csv'
lines_pred = [l.strip().split(',') for l in open(file_path, 'r').readlines()][1:]
items_pred = [dict(observation_id=line[0], class_id=line[1]) for line in lines_pred]
obsv_to_cls = {line[0]: line[1] for line in lines_pred}
cls_to_name = {it['class_id']: it['binomial_name'] for it in items_valid}
print(len(items_pred), len(obsv_to_cls), len(cls_to_name))


for it in items_test:
    obsv_id = it['observation_id']
    cls_id = obsv_to_cls[obsv_id]
    it['class_id'] = cls_id
    it['binomial_name'] = cls_to_name[cls_id]


test_file = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/test_split_pseudo.csv'

with open(test_file, 'w') as wf:
    wf.write('observation_id,endemic,binomial_name,country,code,class_id,file_path\n')
    for it in items_test:
        wf.write('%s,%s,"%s","%s","%s",%s,%s\n' % (it['observation_id'], it['endemic'], it['binomial_name'], it['country'], it['code'], it['class_id'], it['file_path']))
