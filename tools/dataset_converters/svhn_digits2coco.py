import os

import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms

from scipy.io import loadmat
from PIL import Image

import numpy as np
from pycocotools.coco import COCO
import json

class Custom_SVHN(datasets.SVHN):
    def __getitem__(self, index: int):

        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        return img, target

test_dataset = Custom_SVHN("./data/svhn/train")
# test_dataset = Custom_SVHN("./data/svhn/test")

SVHN_CAT = list()
for i in range(10):
    cat = {'id': i, 'name': i}
    SVHN_CAT.append(cat)

## COCO is also required to match classes
coco_path = '/home/kemal/Repositories/mmdetection_robustness/data/coco/annotations/instances_val2017.json'
coco_val = COCO(coco_path)

# 2. Initialize data structures
svhn_data = {'info': 'Natural OOD-SVHN subset', 'licenses': coco_val.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories': coco_val.dataset['categories'], 'orig_categories': SVHN_CAT}

# 2. Initialize data structures
svhn_data = {'info': 'Natural OOD-SVHN subset', 'licenses': coco_val.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories': coco_val.dataset['categories'], 'orig_categories': SVHN_CAT}

img_id_ctr = 0
# data_path = 'svhn_digits_test/'
data_path = "./data/svhn/svhn_digits_train/"
for img, target in test_dataset:
    # Save image
    img_name = str(img_id_ctr) + ".jpg"
    img.save(data_path + img_name)
    

    # Create annotation
    w, h = 32, 32
    img_info = {'license': 0, 'file_name': data_path + img_name, 'height': h, 'width': w, 'id': img_id_ctr}
    img_id_ctr += 1
    svhn_data['images'].append(img_info)

# out_file_path = data_path + 'svhn_digits_test_annotations.json'
out_file_path = data_path + 'svhn_digits_train_annotations.json'

with open(out_file_path, 'w') as outfile:
    json.dump(svhn_data, outfile)
