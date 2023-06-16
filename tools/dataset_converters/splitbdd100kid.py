import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco
import mmdet.datasets.cityscapes as city
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import sys
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
import torch
from scipy.optimize import linear_sum_assignment

NUIMAGES_TO_BDD = {'pedestrian':['pedestrian', "other person"],
                   'vehicle':['car', 'truck', 'bus', 'motorcycle', 'train', "trailer", "other vehicle"],
                   'bicycle': 'bicycle'}

FIX_CLASSES = {'rider':['bicycle', 'motorcycle'] # Confusion with vehicle 
}

def reverse_dict(NUIMAGES_TO_BDD):
    BDD_TO_NUIMAGES = {}
    for k, v in NUIMAGES_TO_BDD.items():
        if type(v) is list:
            for v_ in v:
                BDD_TO_NUIMAGES[v_] = k
        else:
            BDD_TO_NUIMAGES[v] = k
    return BDD_TO_NUIMAGES

BDD100K_TO_NUIMAGES = reverse_dict(NUIMAGES_TO_BDD)

def show_ann(dataset, img_id, subset, anns=None):
    #Get image details
    im = dataset.coco.loadImgs(img_id)[0]

    # Open image
    img = Image.open('data/bdd100k/images/'+subset+'/'+im['filename'])

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    if anns is not None:
        if isinstance(anns, dict):
          ann = anns
          # Create a Rectangle patch
          box = patches.Rectangle((ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3], linewidth=2, edgecolor='r', facecolor='none')

          # Add the patch to the Axes
          ax.add_patch(box)
        else:
          for ann in anns:
            # Create a Rectangle patch
            box = patches.Rectangle((ann['bbox'][0], ann['bbox'][1]), ann['bbox'][2], ann['bbox'][3], linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(box)


    plt.show()



write_json = 0
plot_ann  = 0

subset = 'val'
image_num_limit = 20000

# 1. Read Data
bdd100k_path = 'data/bdd100k/annotations/det_'+subset+'_coco.json'
bdd100k = city.CocoDataset(bdd100k_path, [], test_mode=True)

## COCO is also required to match classes
nuimages_path = 'data/nuimages/annotations/nuimages_v1.0-val.json'
nuimages = COCO(nuimages_path)

# 2. Initialize data structures
cov_shift_data = {'info': 'Natural Covariate Shift Split', 'licenses': ['please check out bdd100k dataset license'], 'images': list(), 'annotations': list(), 'categories': nuimages.dataset['categories']}

# 3. Create Natural Covariate Shift

BDD100K_ID_TO_NAME = {}
for cat in bdd100k.coco.dataset['categories']:
    BDD100K_ID_TO_NAME[cat['id']]= cat['name']

NUIMAGES_NAME_TO_ID = {}
for cat in nuimages.dataset['categories']:
    NUIMAGES_NAME_TO_ID[cat['name']]= cat['id']

nat_cov_sh_counter = 0
ood_gen_counter = 0
amb_counter = 0
no_gt_counter = 0
min_iou_ctr = 0

def get_bboxes(anns, classes):
    boxes = list()
    for ann in anns:
      ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
      if ann_cat_name in classes:
        box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
        boxes.append(box)

    return torch.from_numpy(np.array(boxes))


for i in range(len(bdd100k.coco.dataset['images'])):
    # [{'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': '', 'filename': 'images/v1/patch8/objects365_v1_00420917.jpg'}]
    img_id =  bdd100k.coco.dataset['images'][i]['id']
    ann_ids = bdd100k.coco.get_ann_ids(img_ids=[img_id])
    anns = bdd100k.coco.load_anns(ann_ids)
    # Ex. ann: {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
    
    if subset == 'train' and bdd100k.coco.dataset['images'][i]['weather'] == 'clear':
      continue

    num_id, num_amb, num_fix = 0, 0, 0
    fix_ann =[]
    for ann in anns:
        ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
        if ann_cat_name in BDD100K_TO_NUIMAGES.keys():
            num_id += 1
        elif ann_cat_name in FIX_CLASSES.keys():
            num_fix += 1
            fix_ann.append(ann)

    if num_fix > 0:
      rider_boxes = get_bboxes(fix_ann, classes = [*FIX_CLASSES])
      riding_boxes = get_bboxes(anns, classes = FIX_CLASSES['rider'])

      # There is a rider but there is not what he/she rides
      if riding_boxes.size(0) == 0:
        continue
      
      # Assign riders and riding
      ious = bbox_overlaps(rider_boxes, riding_boxes).numpy()
      riders, riding = linear_sum_assignment(-ious)
      RIDERS = {}
      for i in range(len(riders)):
        RIDERS[riding[i]] = riders[i]

      # Ignore if the iou between a rider and riding is below some threhold.
      # This is because there can be wrong assignment, different 
      # riding items such as prams
      min_iou = 1.0
      for k,v in RIDERS.items():
        if ious[v,k] < min_iou:
          min_iou = ious[v,k]

      if min_iou < 0.10:
        min_iou_ctr += 1
        continue

      # Combine rider and riding and assign to riding
      riding_ctr = 0
      for ann in anns:
        ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
        if ann_cat_name in FIX_CLASSES['rider']:
          if riding_ctr in riding:
            tl_x = min(rider_boxes[RIDERS[riding_ctr]][0], riding_boxes[riding_ctr][0])
            tl_y = min(rider_boxes[RIDERS[riding_ctr]][1], riding_boxes[riding_ctr][1])
            br_x = max(rider_boxes[RIDERS[riding_ctr]][2], riding_boxes[riding_ctr][2])
            br_y = max(rider_boxes[RIDERS[riding_ctr]][3], riding_boxes[riding_ctr][3])
            ann['bbox'] = [tl_x.item(), tl_y.item(), br_x.item() - tl_x.item(), br_y.item() - tl_y.item()]
          riding_ctr += 1

    # Now generate dataset with corrected boxes
    if num_id > 0:
        # Add to Natural Covariate Shift
        nat_cov_sh_counter += 1
        cov_shift_data['images'].append(bdd100k.coco.dataset['images'][i])
        for ann in anns:
            ann_cat_name = BDD100K_ID_TO_NAME[ann['category_id']]
            if ann_cat_name in BDD100K_TO_NUIMAGES.keys():
                coco_ann_cat_name = BDD100K_TO_NUIMAGES[ann_cat_name]
                ann['category_id'] = NUIMAGES_NAME_TO_ID[coco_ann_cat_name]
                cov_shift_data['annotations'].append(ann)
        if subset == 'train' and nat_cov_sh_counter == image_num_limit:
          break
    else:
      no_gt_counter += 1
        


print(len(bdd100k.coco.dataset['images']), nat_cov_sh_counter, no_gt_counter, min_iou_ctr)
if write_json:
    with open('data/bdd100k/annotations/av_natural_covariance_shift_bdd100k_'+subset+'.json', 'w') as outfile:
        json.dump(cov_shift_data, outfile)

# 3. Datasets Statistics
