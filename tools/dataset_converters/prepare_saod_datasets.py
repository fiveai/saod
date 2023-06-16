import argparse
import pdb
import bdd100k2robustod
import cityscapes2robustod
import inaturalist2robustod
import nuimages2robustod
import objects3652robustod
import svhn2robustod
import test2corr
import mmdet.datasets.coco as coco
import mmdet.datasets.robust_od_av as av

import mmcv
import shutil
import json
import os
from os.path import exists

# path = 'data/robustod/annotations/general_od_test.json'
# output_path = 'data/robustod/annotations/general_od_test_corr.json'

path = 'data/robustod/annotations/av_od_test.json'
output_path = 'data/robustod/annotations/av_od_test_corr.json'


IMAGE_PREFIX = {'general_od_id': ['objects365/val/'],
                'av_od_id': ['bdd100k/images/train/', 'bdd100k/images/val/'],
                'ood': ['inaturalist/images/', 'objects365/train/', 'objects365/val/', 'svhn/', 'svhn/']}

OUTPUT_PATHS = {'general_od_id': {'train': 'data/saod/annotations/saod_gen_train.json',
                                  'val': 'data/saod/annotations/saod_gen_val.json',
                                  'test': 'data/saod/annotations/obj45k.json',
                                  'corr': 'data/saod/annotations/obj45k_corr.json'},
                'av_od_id': {'train': 'data/saod/annotations/saod_av_train.json',
                             'val': 'data/saod/annotations/saod_av_val.json',
                             'test': 'data/saod/annotations/bdd45k.json',
                             'corr': 'data/saod/annotations/bdd45k_corr.json'},
                'ood': 'data/saod/annotations/sinobj110kood.json'}

if not exists('data/saod'):
    os.makedirs('data/saod')

if not exists('data/saod/annotations'):
    os.makedirs('data/saod/annotations')


def gettrainvalfiles(trainval_paths, subset):
    mmcv.mkdir_or_exist('data/saod/annotations/')
    if 'all' in subset:
        shutil.copyfile(trainval_paths['general_od_id']['train'], OUTPUT_PATHS['general_od_id']['train'])
        shutil.copyfile(trainval_paths['general_od_id']['val'], OUTPUT_PATHS['general_od_id']['val'])
        shutil.copyfile(trainval_paths['av_od_id']['train'], OUTPUT_PATHS['av_od_id']['train'])
        shutil.copyfile(trainval_paths['av_od_id']['val'], OUTPUT_PATHS['av_od_id']['val'])
    else:
        if 'saod_gen' in subset:
            shutil.copyfile(trainval_paths['general_od_id']['train'], OUTPUT_PATHS['general_od_id']['train'])
            shutil.copyfile(trainval_paths['general_od_id']['val'], OUTPUT_PATHS['general_od_id']['val'])
        if 'saod_av' in subset:
            shutil.copyfile(trainval_paths['av_od_id']['train'], OUTPUT_PATHS['av_od_id']['train'])
            shutil.copyfile(trainval_paths['av_od_id']['val'], OUTPUT_PATHS['av_od_id']['val'])


def combinetestfiles(all_paths):
    for subset, paths in all_paths.items():
        paths.sort()
        if len(paths) == 0:
            continue

        if subset == 'av_od_id':
            datasets = [av.RobustODAV(path, [], test_mode=True) for path in paths]
        else:
            datasets = [coco.CocoDataset(path, [], test_mode=True) for path in paths]

        out_data = {}
        out_data['info'] = 'SAOD Dataset'
        out_data['licenses'] = datasets[0].coco.dataset['licenses']
        out_data['images'] = []
        out_data['annotations'] = []

        img_id_ctr = 0
        ann_id_ctr = 0

        image_prefix = IMAGE_PREFIX[subset]

        for i, ds in enumerate(datasets):
            for j in range(len(ds.coco.dataset['images'])):
                img_id = ds.coco.dataset['images'][j]['id']
                ann_ids = ds.coco.get_ann_ids(img_ids=[img_id])
                anns = ds.coco.load_anns(ann_ids)

                img_info = ds.coco.dataset['images'][j]
                img_info['id'] = img_id_ctr
                img_info['file_name'] = image_prefix[i] + img_info['file_name']
                out_data['images'].append(img_info)

                for ann in anns:
                    ann_info = ann
                    ann_info['id'] = ann_id_ctr
                    ann_info['image_id'] = img_id_ctr
                    # If it is OOD then set category id to 0
                    if subset == 'ood':
                        ann_info['category_id'] = 0

                    out_data['annotations'].append(ann_info)
                    ann_id_ctr += 1
                img_id_ctr += 1

        out_data['categories'] = datasets[0].coco.dataset['categories']

        if subset == 'ood':
            out_path = OUTPUT_PATHS[subset]
        else:
            out_path = OUTPUT_PATHS[subset]['test']

        with open(out_path, 'w') as outfile:
            json.dump(out_data, outfile)


def main():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--subset', help='Subsets to generate', default=None)
    args = parser.parse_args()

    if args.subset is not None:
        general_od_id_datasets = {'coco', 'objects365'}
        av_od_id_datasets = {'bdd100k', 'nuimages'}
        ood_datasets = {'inaturalist', 'objects365', 'svhn'}
        datasets_to_convert = set()
        if 'all' in args.subset:
            datasets_to_convert = datasets_to_convert.union(general_od_id_datasets)
            datasets_to_convert = datasets_to_convert.union(av_od_id_datasets)
            datasets_to_convert = datasets_to_convert.union(ood_datasets)
        else:
            if 'saod_gen' in args.subset:
                datasets_to_convert = datasets_to_convert.union(general_od_id_datasets)
            if 'saod_av' in args.subset:
                datasets_to_convert = datasets_to_convert.union(av_od_id_datasets)
            if 'ood' in args.subset:
                datasets_to_convert = datasets_to_convert.union(ood_datasets)

        converted_file_paths = {'general_od_id': list(), 'av_od_id': list(), 'ood': list()}
        trainval_paths = {'general_od_id': dict(), 'av_od_id': dict()}
        
        for dataset in datasets_to_convert:
            if dataset == 'bdd100k':
                print('converting BDD100K...')
                paths = bdd100k2robustod.convert()
                converted_file_paths['av_od_id'].extend(paths)

            if dataset == 'coco':
                print('Skipping COCO as it is ID...')
                trainval_paths['general_od_id']['train'] = 'data/coco/annotations/instances_train2017.json'
                trainval_paths['general_od_id']['val'] = 'data/coco/annotations/instances_val2017.json'

            if dataset == 'inaturalist':
                print('converting inaturalist...')
                path = inaturalist2robustod.convert()
                converted_file_paths['ood'].append(path)

            if dataset == 'nuimages':
                print('converting nuimages...')
                if exists('data/nuimages/annotations/nuimages_v1.0-train.json') and exists(
                        'data/nuimages/annotations/nuimages_v1.0-val.json'):
                    trainval_paths['av_od_id']['train'] = 'data/nuimages/annotations/nuimages_v1.0-train.json'
                    trainval_paths['av_od_id']['val'] = 'data/nuimages/annotations/nuimages_v1.0-val.json'
                else:
                    paths = nuimages2robustod.convert()
                    trainval_paths['av_od_id']['train'] = paths[0]
                    trainval_paths['av_od_id']['val'] = paths[1]

            if dataset == 'objects365':
                print('converting objects365...')
                paths = objects3652robustod.convert()
                if 'all' in args.subset or 'ood' in args.subset:
                    converted_file_paths['ood'].append(paths[0])
                    converted_file_paths['ood'].append(paths[2])

                if 'all' in args.subset or 'saod_gen' in args.subset:
                    converted_file_paths['general_od_id'].append(paths[1])

            if dataset == 'svhn':
                print('converting svhn...')
                paths = svhn2robustod.convert()
                converted_file_paths['ood'].extend(paths)


        gettrainvalfiles(trainval_paths, args.subset)
        combinetestfiles(converted_file_paths)
        
        if 'saod_gen' in args.subset:
            test2corr.convert(OUTPUT_PATHS['general_od_id']['test'], OUTPUT_PATHS['general_od_id']['corr'])
        elif 'saod_av' in args.subset:
            test2corr.convert(OUTPUT_PATHS['av_od_id']['test'], OUTPUT_PATHS['av_od_id']['corr'])
        print('Dataset is created.')

if __name__ == '__main__':
    main()
