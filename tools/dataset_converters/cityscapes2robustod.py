# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp
from os.path import exists

import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

import os
import json
from tqdm import tqdm
import mmdet.datasets.cityscapes as city
from pycocotools.coco import COCO
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps

NUIMAGES_TO_CITYSCAPES = {'pedestrian':'person',
                        'vehicle':['car', 'truck', 'bus', 'motorcycle', 'train'],
                        'bicycle': 'bicycle'}

FIX_CLASSES = {'rider':['bicycle', 'motorcycle']}

def get_bboxes(anns, class_dict, classes):
  boxes = list()
  for ann in anns:
    ann_cat_name = class_dict[ann['category_id']]
    if ann_cat_name in classes:
      box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
      boxes.append(box)

  return torch.from_numpy(np.array(boxes))

def reverse_dict(NUIMAGES_TO_CITYSCAPES):
    CITYSCAPES_TO_NUIMAGES = {}
    for k, v in NUIMAGES_TO_CITYSCAPES.items():
        if type(v) is list:
            for v_ in v:
                CITYSCAPES_TO_NUIMAGES[v_] = k
        else:
            CITYSCAPES_TO_NUIMAGES[v] = k
    return CITYSCAPES_TO_NUIMAGES

def collect_files(img_dir, gt_dir):
    suffix = 'leftImg8bit.png'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '**/*.png')):
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_instanceIds.png'
        # Note that labelIds are not converted to trainId for seg map
        segm_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        label = CSLabels.id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        category_id = label.id
        iscrowd = int(inst_id < 1000)
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name, osp.basename(segm_file)))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in CSLabels.labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json

def coco2robustod(data_path, limit=-1):
  # 1. Read Data
  ds = city.CocoDataset(data_path, [], test_mode=True)
  nuimages_path = 'data/nuimages/annotations/nuimages_v1.0-val.json'
  nuimages = COCO(nuimages_path)

  # 2. Initialize out data
  out_data = {'info': 'Natural Covariate Shift Split', 'licenses': ['please check out cityscapes dataset license'], 'images': list(), 'annotations': list(), 'categories': nuimages.dataset['categories']}

  CITYSCAPES_TO_NUIMAGES = reverse_dict(NUIMAGES_TO_CITYSCAPES)
  CITYSCAPES_ID_TO_NAME = {}
  for cat in ds.coco.dataset['categories']:
        CITYSCAPES_ID_TO_NAME[cat['id']]= cat['name']

  NUIMAGES_NAME_TO_ID = {}
  for cat in nuimages.dataset['categories']:
        NUIMAGES_NAME_TO_ID[cat['name']]= cat['id']


  counter = 0
  no_gt_counter = 0
  min_iou_ctr = 0

  for i in range(len(ds.coco.dataset['images'])):
    img_id =  ds.coco.dataset['images'][i]['id']
    ann_ids = ds.coco.get_ann_ids(img_ids=[img_id])
    anns = ds.coco.load_anns(ann_ids)

    num_id, num_fix = 0, 0
    fix_ann =[]
    for ann in anns:
        ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
        if ann_cat_name in CITYSCAPES_TO_NUIMAGES.keys():
            num_id += 1
        elif ann_cat_name in FIX_CLASSES.keys():
            num_fix += 1
            fix_ann.append(ann)

    if num_fix > 0:
      rider_boxes = get_bboxes(fix_ann, CITYSCAPES_ID_TO_NAME, classes = [*FIX_CLASSES])
      riding_boxes = get_bboxes(anns, CITYSCAPES_ID_TO_NAME, classes = FIX_CLASSES['rider'])

      # There is a rider but there is not what he/she rides
      if riding_boxes.size(0) == 0:
        continue

      # Assign riders and riding

      ious = bbox_overlaps(rider_boxes, riding_boxes).numpy()
      riders, riding = linear_sum_assignment(-ious)
      RIDERS = {}
      for j in range(len(riders)):
        RIDERS[riding[j]] = riders[j]

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
        counter += 1
        out_data['images'].append(ds.coco.dataset['images'][i])
        for ann in anns:
            ann_cat_name = CITYSCAPES_ID_TO_NAME[ann['category_id']]
            if ann_cat_name in CITYSCAPES_TO_NUIMAGES.keys():
                coco_ann_cat_name = CITYSCAPES_TO_NUIMAGES[ann_cat_name]
                ann['category_id'] = NUIMAGES_NAME_TO_ID[coco_ann_cat_name]
                out_data['annotations'].append(ann)
    else:
      no_gt_counter += 1

  out_file_path = data_path[:-5] + '_robust_od.json'
  with open(out_file_path, 'w') as outfile:
    json.dump(out_data, outfile)

  return out_file_path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('--cityscapes_path', default='data/cityscapes', help='cityscapes data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('--out-dir', default='annotations', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    parser.add_argument(
        '--subset', default=1, type=str, help='dummy')
    args = parser.parse_args()
    return args


def convert():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = osp.join(cityscapes_path, args.out_dir)
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    set_name = dict(train='instancesonly_filtered_gtFine_train.json',
                    val='instancesonly_filtered_gtFine_val.json')

    out_file_paths = []

    for split, json_name in set_name.items():
      out_file_path = osp.join(out_dir, json_name)
      if not exists(out_file_path):
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It took {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, out_file_path)

      print('Mapping CityScapes classes to ID classes for training set. See:' + out_file_path[:-5] + '_robust_od.json' )
      out_file_paths.append(coco2robustod(out_file_path))

    return out_file_paths
