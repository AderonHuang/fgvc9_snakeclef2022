from collections import defaultdict
from tqdm import tqdm
import csv
import os
import random


def load_csv_annotations(fp):
    lines = []
    with open(fp, 'r') as rf:
        cr = csv.reader(rf)
        for row in tqdm(cr):
            if len(row) < 7:
                continue
            if row[0] == 'observation_id':
                names = row
                continue
            lines.append({name : row[idx] for idx, name in enumerate(names)})
        print('load %d annotations from %s' % (len(lines), fp))
    return lines


file_path = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/SnakeCLEF2022-TrainMetadata.csv'
item_list = load_csv_annotations(fp=file_path)


bio_name_hist = defaultdict(int)
class_id_hist = defaultdict(int)
country_hist = defaultdict(int)
code_hist = defaultdict(int)
observation_hist = defaultdict(int)
bio_name_to_img_list = defaultdict(list)
for item in item_list:
    bio_name_hist[item['binomial_name']] += 1
    class_id_hist[item['class_id']] += 1
    country_hist[item['country']] += 1
    code_hist[item['code']] += 1
    observation_hist[item['observation_id']] += 1
    bio_name_to_img_list[item['binomial_name']].append(item['file_path'])


print('item_list', len(item_list))
print('bio_name_hist', len(bio_name_hist))
print('class_id_hist', len(class_id_hist))
print('country_hist', len(country_hist))
print('code_hist', len(code_hist))
print('observation_hist', len(observation_hist))


root_dir = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/SnakeCLEF2022-small_size/SnakeCLEF2022-small_size'
for idx_bio_name, bio_name in enumerate(tqdm(sorted(bio_name_to_img_list.keys()))):
    img_list = sorted(bio_name_to_img_list[bio_name])
    print(bio_name, len(img_list))
    # for img_path in img_list[0:3]:
    #     cur_path = os.path.join(root_dir, img_path)
    #     tmp = img_path.split('/')
    #     assert len(tmp) == 3, img_path
    #     new_img_name = '_'.join([tmp[1], tmp[0], tmp[2]])
    #     cmd = 'cp %s result/%s' % (cur_path, new_img_name)
    #     os.system(cmd)
    if idx_bio_name > 10:
        break


num_train = int(len(item_list) * 0.8)
random.seed(1)
random.shuffle(item_list)
train_list = item_list[:num_train]
valid_list = item_list[num_train:]
train_file = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/train_split.csv'
valid_file = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022/valid_split.csv'


with open(train_file, 'w') as wf:
    wf.write('observation_id,endemic,binomial_name,country,code,class_id,file_path\n')
    for it in train_list:
        wf.write('%s,%s,"%s","%s","%s",%s,%s\n' % (it['observation_id'], it['endemic'], it['binomial_name'], it['country'], it['code'], it['class_id'], it['file_path']))

with open(valid_file, 'w') as wf:
    wf.write('observation_id,endemic,binomial_name,country,code,class_id,file_path\n')
    for it in valid_list:
        wf.write('%s,%s,"%s","%s","%s",%s,%s\n' % (it['observation_id'], it['endemic'], it['binomial_name'], it['country'], it['code'], it['class_id'], it['file_path']))
