import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco
import mmdet.datasets.objects365 as obj
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import svhnl


def coco2robustod(data_path):
  SVHN_CAT = list()
  for i in range(10):
    cat = {'id': i, 'name': i}
    SVHN_CAT.append(cat)

  # 1. Read Data
  f = open(data_path)
  svhn_raw = json.load(f)
  print('Number of images:', len(svhn_raw))

  ## COCO is also required to match classes
  coco_path = 'data/coco/annotations/instances_val2017.json'
  coco_val = COCO(coco_path)

  # 2. Initialize data structures
  svhn_data = {'info': 'Natural OOD-SVHN subset', 'licenses': coco_val.dataset['licenses'], 'images': list(), 'annotations': list(), 'categories': coco_val.dataset['categories'], 'orig_categories': SVHN_CAT}

  img_id_ctr = 0
  ann_id_ctr = 0
  offset = 100000
  for img in svhn_raw:
    w, h = Image.open(data_path[:-13] + img['name']).size
    image_id = offset + img_id_ctr
    img_info = {'license': 0, 'file_name': data_path[10:-13] + img['name'], 'height': h, 'width': w, 'id': image_id}
    img_id_ctr += 1
    svhn_data['images'].append(img_info)
    
    for box in img['boxes']:
      ann = {'id': ann_id_ctr, 'iscrowd': 0, 'area': w*h, 'bbox': [box['left'], box['top'], box['width'], box['height']], 'image_id': image_id, 'category_id': int(box['label'])}
      svhn_data['annotations'].append(ann)
      ann_id_ctr += 1

  out_file_path = data_path[:-5] + '_robust_od.json'
  with open(out_file_path, 'w') as outfile:
    json.dump(svhn_data, outfile)

  return out_file_path

def convert():
  mat_file_paths = ['data/svhn/train/digitStruct.mat', 'data/svhn/test/digitStruct.mat']
  file_paths = ['data/svhn/train/svhn_ann.json', 'data/svhn/test/svhn_ann.json']

  out_file_paths = []

  #for mat_path, path in zip(mat_file_paths, file_paths):
  #  svhnl.ann_to_json(file_path=mat_path, save_path=path, bbox_type='normalize')

  for path in file_paths:
    out_file_paths.append(coco2robustod(path))

  return out_file_paths
