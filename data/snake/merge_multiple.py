from collections import defaultdict
import json


file_path_list = [
    '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_2/snake_v2_meta_ovs_trainval/result_snakeclef2022test_e222.tc.result.csv',  # 0.75288
    '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_2/snake_v2_meta/result_snakeclef2022test_e300.tc.result.csv',  # 0.72997
    '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/MetaFG_meta_0/snake_v0_meta/result_snakeclef2022test_e300.tc.result.csv',  # 0.72190
]

obs2list = defaultdict(list)


for file_path in file_path_list:
    lines = [l.strip() for l in open(file_path, 'r').readlines()]
    lines = lines[1:]
    for line in lines:
        obs, class_id = line.split(',')
        obs2list[obs].append(class_id)
    print(len(obs2list), len(lines))


result_list = []
for obs, class_list in obs2list.items():
    # Get the most.
    dist = defaultdict(int)
    for p in class_list:
        dist[p] += 1
    dist_list = []
    for k, v in dist.items():
        dist_list.append((k, v))
    new_dist_list = sorted(dist_list, key=lambda x: x[1], reverse=True)
    result_list.append((obs, new_dist_list[0][0]))


output_file = '/mnt/workspace/wuyou.zc/sources/FineGrained/MetaFormer_snake/output/result_snakeclef2022test_merge.csv'
# with open(output_file, 'w') as wf:
#     wf.write('ObservationId,class_id\n')
#     for it in result_list:
#         wf.write('%s,%s\n' % (it[0], it[1]))


def get_items(file_path):
    lines = [l.strip().split(',') for l in open(file_path, 'r').readlines()][1:]
    items = {line[0]: line[1] for line in lines}
    return items


for file_path in file_path_list:
    it1 = get_items(file_path=file_path)
    it2 = get_items(file_path=output_file)
    cnt = 0
    for k, v in it1.items():
        if v!= it2[k]:
            cnt += 1
    print(cnt)
