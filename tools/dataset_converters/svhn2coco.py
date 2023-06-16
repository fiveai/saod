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

SVHN_CAT = list()
for i in range(10):
  cat = {'id': i, 'name': i}
  SVHN_CAT.append(cat)

plot_img = 0
write_json = 1

# 1. Read Data
subset = 'test'
svhn_path = 'data/svhn/'+subset+'/svhn_ann.json'
f = open(svhn_path)
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
  w, h = Image.open('data/svhn/'+ subset + '/' + img['name']).size
  if plot_img:
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(Image.open('data/svhn/'+subset + '/' + img['name']))
  image_id = offset + img_id_ctr
  img_info = {'license': 0, 'file_name': subset + '/' + img['name'], 'height': h, 'width': w, 'id': image_id}
  img_id_ctr += 1
  svhn_data['images'].append(img_info)
  
  for box in img['boxes']:
    ann = {'id': ann_id_ctr, 'iscrowd': 0, 'area': w*h, 'bbox': [box['left'], box['top'], box['width'], box['height']], 'image_id': image_id, 'category_id': int(box['label'])}
    svhn_data['annotations'].append(ann)
    ann_id_ctr += 1

    if plot_img:
      # Create a Rectangle patch
      box = patches.Rectangle((ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3], linewidth=2, edgecolor='r', facecolor='none')

      # Add the patch to the Axes
      ax.add_patch(box)
  if img_id_ctr == 10:
    break
  
  if plot_img:
    plt.show()  

if write_json:
    with open('data/svhn/'+subset+'/generalod_ood_svhn_test_toy.json', 'w') as outfile:
        json.dump(svhn_data, outfile)

# 3. Datasets Statistics
