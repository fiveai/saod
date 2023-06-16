import copy
import json
import mmdet.datasets.coco as coco

def convert(path, output_path, duplication=3):
    ds = coco.CocoDataset(path, [], test_mode=True)

    out_data = {}
    out_data['info'] = 'OD-Robust ID dataset Corruption'
    out_data['licenses'] = ds.coco.dataset['licenses']
    out_data['images'] = []
    out_data['annotations'] = []

    ann_id_ctr = 0
    offset = 1e10

    for j in range(len(ds.coco.dataset['images'])):
        img_id = ds.coco.dataset['images'][j]['id']
        ann_ids = ds.coco.get_ann_ids(img_ids=[img_id])
        anns = ds.coco.load_anns(ann_ids)

        for i in range(duplication):
            img_info = copy.deepcopy(ds.coco.dataset['images'][j])
            img_info['id'] = int(img_id + (offset * (i + 1)))
            out_data['images'].append(img_info)

            for ann in anns:
                ann_info = copy.deepcopy(ann)
                ann_info['id'] = ann_id_ctr
                ann_info['image_id'] = img_info['id']
                out_data['annotations'].append(ann_info)
                ann_id_ctr += 1

    out_data['categories'] = ds.coco.dataset['categories']

    with open(output_path, 'w') as outfile:
        json.dump(out_data, outfile)

    out_dataset = coco.CocoDataset(output_path, [], test_mode=True)
