import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco



random.seed(0)
test_subset = 0.50
num_images = 5000


# 1. Validate Split

def get_labels(dataset, test_set):
    labels_all, labels_test, labels_val = np.zeros(0), np.zeros(0), np.zeros(0)

    for img_idx in range(num_images):
        labels_all = np.append(labels_all, dataset.get_ann_info(img_idx)['labels'])
        if img_idx in test_set:
            labels_test = np.append(labels_test, dataset.get_ann_info(img_idx)['labels'])
        else:
            labels_val = np.append(labels_test, dataset.get_ann_info(img_idx)['labels'])

    return labels_all, labels_test, labels_val

# Opening JSON file

filepath = '/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_val2017.json'
dataset = coco.CocoDataset(filepath, [], test_mode=True)

# returns JSON object as
# a dictionary
test_set = set(random.sample(range(num_images), int(num_images * test_subset)))

all_labels, test_labels, val_labels = get_labels(dataset, test_set)
n, bins, patches = plt.hist([all_labels, test_labels, val_labels], 80, density = True)
plt.show()

# 2. Generate Split

f = open(filepath)
data = json.load(f)

test_data = {'info': data['info'], 'licenses': data['licenses'], 'images': list(), 'annotations': list(), 'categories': data['categories']}
val_data = {'info': data['info'], 'licenses': data['licenses'], 'images': list(), 'annotations': list(), 'categories': data['categories']}

nogt = 0
for i in range(len(data['images'])):
    img_id =  data['images'][i]['id']
    ann_ids = dataset.coco.get_ann_ids(img_ids=[img_id])
    anns = dataset.coco.load_anns(ann_ids)

    ''' #Comment out for splitting for gt no gt
    #if i in test_set:
        # We do not want images in test set without any ground truth box
        # Check number of ground truths
    num_gt = len(anns)
    if num_gt == 0:
        nogt += 1
        val_data['images'].append(data['images'][i])
        for ann in anns:
            val_data['annotations'].append(ann)
    else:
        test_data['images'].append(data['images'][i])
        for ann in anns:
            test_data['annotations'].append(ann)
   # else:
   #     val_data['images'].append(data['images'][i])
   #     for ann in anns:
   #         val_data['annotations'].append(ann)
   '''

    if i in test_set:
        # We do not want images in test set without any ground truth box
        # Check number of ground truths
        num_gt = len(anns)
        if num_gt == 0:
            nogt += 1
            val_data['images'].append(data['images'][i])
            for ann in anns:
                val_data['annotations'].append(ann)
        else:
            test_data['images'].append(data['images'][i])
            for ann in anns:
                test_data['annotations'].append(ann)
    else:
        val_data['images'].append(data['images'][i])
        for ann in anns:
            val_data['annotations'].append(ann)

# with open('/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_withgt_val2017.json', 'w') as outfile:
#    json.dump(test_data, outfile)

# with open('/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_withnogt_val2017.json', 'w') as outfile:
#    json.dump(val_data, outfile)

with open('/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_robustness_val2017.json', 'w') as outfile:
    json.dump(val_data, outfile)

with open('/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_robustness_test2017.json', 'w') as outfile:
    json.dump(test_data, outfile)

# 3. Datasets Statistics
print(nogt)
print('----------------Validation Set------------------')
print('Number of Images=', len(val_data['images']))
print('Number of Annotations=', len(val_data['annotations']))
print('Minimum Number of Examples for a Class=', min(n[2]))


print('----------------Test Set------------------')
print('Number of Images=', len(test_data['images']))
print('Number of Annotations=', len(test_data['annotations']))
print('Minimum Number of Examples for a Class=', min(n[1]))

# Closing file
f.close()
