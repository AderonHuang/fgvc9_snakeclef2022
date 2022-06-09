import csv
import os


def load_csv_annotations(fp):
    lines = []
    with open(fp, 'r') as rf:
        cr = csv.reader(rf)
        for row in cr:
            if len(row) < 7:
                continue
            if row[0] == 'observation_id':
                names = row
                continue
            lines.append({name : row[idx] for idx, name in enumerate(names)})
    return lines


istrain = True
root = '/gruntdata/DL_dataset/wuyou.zc/wuyou.zc/data/fine_grained/snakeclef2022'

imgs_root = os.path.join(root, 'SnakeCLEF2022-small_size', 'SnakeCLEF2022-small_size')
if istrain:
    data_file = os.path.join(root,'train_split.csv')
else:
    data_file = os.path.join(root,'valid_split.csv')

items = load_csv_annotations(fp=os.path.join(root,'SnakeCLEF2022-TrainMetadata.csv'))
classes = set()
location_codes, endemics = set(), set()
for it in items:
    classes.add(it['class_id'])
    endemics.add(it['endemic'])
    location_codes.add(it['code'])
classes = sorted(list(classes))
endemics = sorted(list(endemics))
location_codes = sorted(list(location_codes))
class_to_idx = {name:ind for ind,name in enumerate(classes)}

print(len(classes), list(classes)[0:5])
print(len(endemics), list(endemics)[0:5])
print(len(location_codes), list(location_codes)[0:5])
print(location_codes.index('AF'))

aux_info = True
items = load_csv_annotations(fp=data_file)
images_and_targets = []
images_info = []
for it in items:
    file_path = os.path.join(imgs_root, it['file_path'])
    class_name = it['class_id']
    target = class_to_idx[class_name]
    if aux_info:
        meta_data = [0] * len(location_codes)
        idx_code = location_codes.index(it['code'])
        meta_data[idx_code] = 1
        meta_data = meta_data + [endemics.index(it['endemic'])]
        images_and_targets.append([file_path, target, meta_data])
    else:
        images_and_targets.append([file_path, target])

print(len(images_and_targets))
