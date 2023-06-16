## Preparing the Datasets for Self-aware Object Detection Task

We use 6 datasets to create the following 3 splits of datasets:

- General Object Detection (SAOD-Gen)
- Autonomous Vehicle Object Detection (SAOD-AV)
- OOD

Depending on the data splits you need, please download the datasets as described below. We provide the necessary links, descriptions and the resulting directory structure in detail in the following. Note that some of the datasets require registration of the users. So, please make sure that you subscribe first for such datasets in order to access to the datasets below.

####The links and details of the used datasets:

- [BDD100k](https://bdd-data.berkeley.edu/portal.html#download) (used in SAOD-Gen): Please download ```100K Images``` for the images and ```Detection 2020 Labels``` for the annotations.
- [COCO](https://cocodataset.org/#download) (used in SAOD-Gen): Follow the standard directory structure suggested by mmdetection for coco. So, please download ```2017 Train images```, ```2017 Val images``` and ```2017 Train/Val Annotations```.
- [iNaturalist](https://github.com/visipedia/inat_comp/blob/master/2017/README.md) (used in OOD subset): Please download training and validation images as well as validation bounding box annotations.
- [nuImages](https://www.nuscenes.org/nuimages#download) (used in AV-OD): Please download ```Metadata``` and ```Samples```.
- [Objects365](https://open.baai.ac.cn/data-set-detail/MTI2NDc=/MTA=/true) (used in Gen-OD and OOD): Please download all patches and annotations for training and validation subsets.
- [SVHN](http://ufldl.stanford.edu/housenumbers/) (used in OOD): Please download ```train.tar.gz``` and ```test.tar.gz```. Note that we use entire images here, not cropped digits.

####The directory structure for SAOD

Please ensure that the downloaded files are organized as follows:

```text
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── bddk100k 
│   │   ├── annotations
│   │   │   ├── det_train.json
│   │   │   ├── det_val.json
│   │   ├── images
│   │   │   ├── train
│   │   │   |   ├── 70K images
│   │   │   ├── val
│   │   │   |   ├── 20K images
│   ├── coco
│   │   ├── annotations
│   │   │   ├── 6 json files
│   │   ├── train2017
│   │   │   ├── 118287 images
│   │   ├── val2017
│   │   │   ├── 5000 images
│   ├── inaturalist
│   │   ├── annotations
│   │   │   ├── val_2017_bboxes.json
│   │   ├── images
│   │   │   ├── train_val_images
|   │   │   │   ├── Actinopterygii
|   │   │   │   |   ├── 53 directories, each of which containing images
|   │   │   │   ├── ... 
|   │   │   │   ├── Reptilia 
|   │   │   │   |   ├── 289 directories, each of which containing images
│   ├── nuimages
│   │   ├── samples
│   │   │   ├── CAM_BACK
|   │   │   │   ├── 17105 images
│   │   │   ├── ...
│   │   │   ├── CAM_FRONT_RIGHT
|   │   │   │   ├── 14613 images
│   │   ├── v1.0-train
│   │   │   ├── 10 json files
│   │   ├── v1.0-val
│   │   │   ├── 10 json files
│   ├── objects365
│   │   ├── annotations
│   │   │   ├── zhiyuan_objv2_train.json
│   │   │   ├── zhiyuan_objv2_val.json
│   │   ├── train
│   │   │   ├── images
|   │   │   │   ├── v1
|   |   │   │   │   ├── patch0
|   |   |   │   │   │   ├── 34797 images
|   |   │   │   │   ├── ...
|   │   |   │   │   ├── patch15
|   |   |   │   │   │   ├── 31477 images
|   │   │   │   ├── v2
|   |   │   │   │   ├── patch16
|   |   |   │   │   │   ├── 34341 images
|   |   │   │   │   ├── ...
|   │   |   │   │   ├── patch50
|   |   |   │   │   │   ├── 34357 images
│   │   ├── val
│   │   │   ├── images
|   │   │   │   ├── v1
|   |   │   │   │   ├── patch0
|   |   |   │   │   │   ├── 1311 images
|   |   │   │   │   ├── ...
|   │   |   │   │   ├── patch15
|   |   |   │   │   │   ├── 1038 images
|   │   │   │   ├── v2
|   |   │   │   │   ├── patch16
|   |   |   │   │   │   ├── 1789 images
|   |   │   │   │   ├── ...
|   │   |   │   │   ├── patch43
|   |   |   │   │   │   ├── 1293 images
│   ├── svhn
│   │   ├── train
│   │   │   ├── 33404 items
│   │   ├── test
│   │   │   ├── 13070 items
```

####Generating the annotation files
Depending on which subset to generate, please run the following script:

```shell
python tools/dataset_converters/prepare_saod_datasets.py  [--subset ${SUBSET}]
```

such that subset can be either `all`, `saod_gen`, `saod_av` or `ood`. saod_gen and saod_av options will only generate in-distribution test splits, and ood option produces the ood annotations. As an example, if you want to benchmark self-aware object detector only on our SAOD-Gen setting, then you should call the command twice with `saod_gen` and `ood` arguments. 

The process may take some time, finally you should find following files (or a subset depending on the used argument for subset) under `data/saod/annotations/`:

- saod_gen_train.json
- saod_gen_val.json
- obj45k.json
- obj45k_corr.json
- saod_av_train.json
- saod_av_val.json
- bdd45k.json
- bdd45k_corr.json
- sinobj110kood.json
