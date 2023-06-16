import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import sys

OOD_CLASSES  =  {'Actinopterygii', 'Amphibia', 'Animalia',
               'Arachnida', 'Insecta', 'Mollusca', 'Reptilia'}

def coco2robustod(data_path):
    inat = coco.CocoDataset(data_path, [], test_mode=True)

    coco_path = 'data/coco/annotations/instances_val2017.json'
    coco_val = COCO(coco_path)

    # 2. Initialize data structures
    out_data = {'info': 'OOD-inat Split', 'licenses': inat.coco.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories' : coco_val.dataset['categories'], 'orig_categories': inat.coco.dataset['categories']}
    # 3. Create Natural Covariate Shift and OOD splits
    # Natural Covariate Shift: Include an image only if it has the same classes with COCO
    # OOD Split: Include an image if it has no classes with COCO
    INAT_CAT_TO_SUPERCAT = {}
    for cat in inat.coco.dataset['categories']:
        INAT_CAT_TO_SUPERCAT[cat['id']]= cat['supercategory']

    ood_gen_counter = 0
    for i in range(len(inat.coco.dataset['images'])):
        # [{'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': '', 'filename': 'images/v1/patch8/objects365_v1_00420917.jpg'}]
        img_id =  inat.coco.dataset['images'][i]['id']
        ann_ids = inat.coco.get_ann_ids(img_ids=[img_id])
        anns = inat.coco.load_anns(ann_ids)
        # Ex. ann: {'area': 25740.0, 'iscrowd': 0, 'image_id': 0, 'bbox': [365, 229, 330, 156], 'category_id': 0, 'id': 0}
        
        num_ood = 0
        for ann in anns:
            ann_cat_name = INAT_CAT_TO_SUPERCAT[ann['category_id']]
            if ann_cat_name in OOD_CLASSES:
                num_ood += 1

        if num_ood > 0 and num_ood == len(anns):
            # Add to OOD-general
            out_data['images'].append(inat.coco.dataset['images'][i])
            for ann in anns:
                out_data['annotations'].append(ann)
            ood_gen_counter += 1

    out_file_path = data_path[:-5] + '_robust_od.json'
    with open(out_file_path, 'w') as outfile:
        json.dump(out_data, outfile)
    return out_file_path

def convert():
    inat_path = 'data/inaturalist/annotations/val_2017_bboxes.json'
    return coco2robustod(inat_path)

# 3. Datasets Statistics
