import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random
import mmdet.datasets.coco as coco
import mmdet.datasets.robust_od_av as av

# Specify Data
av_od = 1
gen_od = 0
ood = 0
if av_od:
    print('Comparing nuimages val and RobustOD AV-OD test')
    paths = ['data/nuimages/annotations/nuimages_v1.0-val.json',
             'data/robustod/annotations/av_od_test.json']
    legend_labels = ['nuImages val', 'BDD45K']
    K = 3
    out_file = "figures/avoddatacomparison.pdf"
    datasets = [av.RobustODAV(path, [], test_mode=True) for path in paths]
elif gen_od:
    print('Comparing coco val and RobustOD Gen-OD test')
    paths = ['data/coco/annotations/instances_val2017.json',
             'data/robustod/annotations/general_od_test.json']
    legend_labels = ['COCO val', 'Obj45K']
    K = 80
    out_file = "figures/genoddatacomparison.pdf"
    datasets = [coco.CocoDataset(path, [], test_mode=True) for path in paths]
elif ood:
    print('Collecting OOD')
    paths = ['data/objects365/annotations/zhiyuan_objv2_val_robust_od_ood.json',
             'data/objects365/annotations/zhiyuan_objv2_train_robust_od_ood.json',
             'data/inaturalist/annotations/val_2017_bboxes_robust_od_ood.json',
             'data/svhn/train/svhn_ann_robust_od.json',
             'data/svhn/test/svhn_ann_robust_od.json']
    legend_labels = ['objects365', 'inaturalist', 'svhn']
    K = 1
    out_file = 'ood.pdf'
    datasets = [coco.CocoDataset(path, [], test_mode=True) for path in paths]

# Collect Labels
all_labels = []
for i, dataset in enumerate(datasets):
    labels = np.zeros(0)
    for img_idx in range(len(dataset.coco.dataset['images'])):
        if ood:
            label = 0
        else:
            label = dataset.get_ann_info(img_idx)['labels']
        if i == 1:
            breakpoint()
        labels = np.append(labels, label)
    all_labels.append(labels)


if ood:
    # Concat OOD datasets
    all_labels[0] = np.concatenate([all_labels[0], all_labels[1]])
    all_labels[3] = np.concatenate([all_labels[3], all_labels[4]])
    all_labels.pop(1)
    all_labels.pop(-1)

print('Number of total annotations in the datasets:', *[len(labels) for labels in all_labels])
print('Number of total images in the datasets:', *[len(dataset.coco.dataset['images']) for dataset in datasets])
fig, ax = plt.subplots()

plt.rcParams['font.size'] = '25'
font_size = 25

n, bins, patches = plt.hist(all_labels, bins=np.arange(0, K + 1, 1) - 0.5)

categories = [cat['name'] for cat in datasets[0].coco.dataset['categories']]
if av_od:
    x_tick = categories
    x_tick_loc = range(len(categories))
    fig.set_size_inches(12, 10, forward=True)

elif gen_od:
    x_tick_loc = range(0, len(categories))
    x_tick = categories
    plt.xticks(rotation=90)
    fig.set_size_inches(23, 10, forward=True)

elif ood:
    x_tick = [0]
    x_tick_loc = [0]

plt.xticks(x_tick_loc, x_tick, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.yscale("log")

plt.legend(legend_labels)
fig.tight_layout()

#plt.show()

plt.savefig(out_file, format='pdf')
