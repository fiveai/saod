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

NUIMAGES_TO_CITYSCAPES = {'pedestrian':'person',
                        'vehicle':['car', 'truck', 'bus', 'motorcycle', 'train'],
                        'bicycle': 'bicycle'}

AMBIGUOUS_CLASSES = { # Confusion with vehicle 
                    }

FIX_CLASSES = {'rider':['bicycle', 'motorcycle'] # Confusion with vehicle 
                    }

def reverse_dict(NUIMAGES_TO_CITYSCAPES):
    CITYSCAPES_TO_NUIMAGES = {}
    for k, v in NUIMAGES_TO_CITYSCAPES.items():
        if type(v) is list:
            for v_ in v:
                CITYSCAPES_TO_NUIMAGES[v_] = k
        else:
            CITYSCAPES_TO_NUIMAGES[v] = k
    return CITYSCAPES_TO_NUIMAGES

CITYSCAPES_TO_NUIMAGES = reverse_dict(NUIMAGES_TO_CITYSCAPES)

def show_ann(dataset, img_id, anns=None):
    #Get image details
    im = dataset.coco.loadImgs(img_id)[0]

    # Open image
    img = Image.open('data/cityscapes/leftImg8bit/train/'+im['filename'])

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



plot_dist = 0
write_json =0

plot_ann  = 0
plot_img = 0

# 1. Read Data
# cs_path = 'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'
cs_path = 'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
# cs_path = 'data/cityscapes/annotations/av_natural_covariance_shift_cityscapes_train.json'

cityscapes = city.CityscapesDataset(cs_path, [], test_mode=True)

## COCO is also required to match classes
nuimages_path = 'data/nuimages/annotations/nuimages_v1.0-mini.json'
if plot_dist:
    nuimages = coco.CocoDataset(nuimages_path, [], test_mode=True)

    labels_obj = np.zeros(0)
    for img_idx in range(len(cityscapes.coco.dataset['images'])):
        labels_obj = np.append(labels_obj, cityscapes.get_ann_info(img_idx)['labels'])

    labels_coco = np.zeros(0)
    for img_idx in range(len(nuimages.coco.dataset['images'])):
        labels_coco = np.append(labels_coco, nuimages.get_ann_info(img_idx)['labels'])

    n, bins, patches = plt.hist([labels_coco, labels_obj], bins=np.arange(0, 3 + 1, 1)-0.5)
    plt.xticks(range(len([*NUIMAGES_TO_CITYSCAPES])), [*NUIMAGES_TO_CITYSCAPES], size='small', rotation = 90)
    plt.yscale("log")
    plt.show()
    print("Coco Number of Examples:", n[0])
    print("Objects365 Number of Examples:", n[1])
    sys.exit()

nuimages = COCO(nuimages_path)

# 2. Initialize data structures
cov_shift_data = {'info': 'Natural Covariate Shift Split', 'licenses': ['please check out cityscapes dataset license'], 'images': list(), 'annotations': list(), 'categories': nuimages.dataset['categories']}
ood_data = {'info': 'OOD-general Split', 'licenses': ['please check out cityscapes dataset license'], 'images': list(), 'annotations': list(), 'categories': nuimages.dataset['categories'], 'orig_categories': cityscapes.coco.dataset['categories']}
# 3. Create Natural Covariate Shift and OOD splits
# Natural Covariate Shift: Include an image only if it has the same classes with COCO
# OOD Split: Include an image if it has no classes with COCO
CITYSCAPES_ID_TO_NAME = {}
for cat in cityscapes.coco.dataset['categories']:
    CITYSCAPES_ID_TO_NAME[cat['id']]= cat['name']

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

      ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
      if ann_cat_name in classes:
        box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
        boxes.append(box)

    return torch.from_numpy(np.array(boxes))


for i in range(len(cityscapes.coco.dataset['images'])):
    # [{'height': 512, 'id': 420917, 'license': 5, 'width': 769, 'file_name': 'images/v1/patch8/objects365_v1_00420917.jpg', 'url': '', 'filename': 'images/v1/patch8/objects365_v1_00420917.jpg'}]
    img_id =  cityscapes.coco.dataset['images'][i]['id']
    ann_ids = cityscapes.coco.get_ann_ids(img_ids=[img_id])
    anns = cityscapes.coco.load_anns(ann_ids)
    # Ex. ann: {'id': 51, 'iscrowd': 0, 'isfake': 0, 'area': 1584.6324365356109, 'isreflected': 0, 'bbox': [491.3955078011, 88.1856689664, 35.65588379410002, 44.442382796800004], 'image_id': 420917, 'category_id': 84}
    
    num_id, num_amb, num_fix = 0, 0, 0
    fix_ann =[]
    for ann in anns:
        ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
        if ann_cat_name in CITYSCAPES_TO_NUIMAGES.keys():
            num_id += 1
        elif ann_cat_name in AMBIGUOUS_CLASSES:
            num_amb += 1
        elif ann_cat_name in FIX_CLASSES.keys():
            num_fix += 1
            fix_ann.append(ann)

    if num_amb > 0:
        show_ann(cityscapes, img_id, anns)
        amb_counter += 1
        continue

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
        ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
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
        cov_shift_data['images'].append(cityscapes.coco.dataset['images'][i])
        for ann in anns:
            ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
            if ann_cat_name in CITYSCAPES_TO_NUIMAGES.keys():
                coco_ann_cat_name = CITYSCAPES_TO_NUIMAGES[ann_cat_name]
                ann['category_id'] = NUIMAGES_NAME_TO_ID[coco_ann_cat_name]
                cov_shift_data['annotations'].append(ann)
                if plot_ann:
                    print('Class is:', ann_cat_name)
                    show_ann(cityscapes, img_id, ann)

    else:
      no_gt_counter += 1
        


print(len(cityscapes.coco.dataset['images']), nat_cov_sh_counter, no_gt_counter, min_iou_ctr)
if write_json:
    with open('data/cityscapes/annotations/av_natural_covariance_shift_cityscapes_val.json', 'w') as outfile:
        json.dump(cov_shift_data, outfile)

# 3. Datasets Statistics
