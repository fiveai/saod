from saod_evaluation import *
import argparse
import io
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='SAOD evaluation')

    ## DATASET PARAMETERS
    parser.add_argument('config', help='The model to be evaluated (should match the directory name in configs/saod/test. E.g., faster_rcnn_r50_fpn_straug_3x_coco)')
    
    # Whether or not using corrupted images for evaluation. Should be True for a proper benchmarking
    parser.add_argument('--benchmark_with_corr', help='Whether or not to benchmark with corrupted images',
                        default=True)


    ## THE PURPOSE of USING THIS TOOLKIT: There are 4 possible use-cases
    ## (i) Benchmarking SAODets
    ## (ii) only OOD evaluation
    ## (iii) only calibration and accuracy evaluation
    ## (iv) standard COCO Evaluation with top-100 detections

    # Benchmark using DAQ. Should be False if the toolkit is used only for calibration or OOD evaluation
    parser.add_argument('--benchmark', help='Benchmark the model', default=False)

    # Options: isonotic_regression, linear_regression, histogram_binning, identity
    parser.add_argument('--calibrate', help='Whether or not to calibrate', default=False)

    # TO DO: Try to remove or properly incorporate these arguments
    parser.add_argument('--ood_evaluate', help='whether or not to evaluate OOD performance', default=False)

    # We also provide the option to use standard COCO evaluation
    parser.add_argument('--evaluate_top_100', help='Using top-100 detections with the standard COCO style evaluation',
                        default=False)

    ## IMAGE-LEVEL UNCERTAINTY and DETECTION-LEVEL CONFIDENCE THRESHOLDING METHODS
    # We use 0.95 in the ablation and -1 for using Our Pseudo ID/OOD set approach
    parser.add_argument('--image_level_threshold', help='TPR threshold in (0,1) or -1 for ours', default=-1, type=float)

    # We use 0.50 in the ablation or -1 for using LRP-Optimal thresholds
    parser.add_argument('--detection_level_threshold', help='Detection threshold in (0,1) or -1 for ours', default=-1,
                        type=float)

    ## DETECTION-LEVEL UNCERTAINTY ESTIMATION PARAMETERS
    # For softmax-based (Faster R-CNN, Prob-Faster R-CNN variants), 0: '1-score', 1: 'entropy', 2:'ds'
    # For sigmoid-based (all others), 0: '1-score', 1: 'entropy', 2:'ds', 3:'avg_entropy', 4:'max_class_entropy',
    parser.add_argument('--cls_unc_type', help='Classification uncertainty type', default=0, type=int)

    # For Probabilistic detectors 3:'determinant', 4:'gaussian_entropy', 5:'trace'
    parser.add_argument('--loc_unc_type', help='Localization uncertainty type', default=-1, type=int)

    ## AGGREGATION PARAMETERS TO OBTAIN IMAGE-LEVEL UNCERTAINTY
    # Options: mean or sum
    parser.add_argument('--aggregation', help='Reduction type for image level', default='mean')
    parser.add_argument('--max_det_num', help='Number of the most confident detections in an image', default=3, type=int)

    ## EVALUATION PARAMETERS    
    # Number of calibration bins to approximate the calibration error
    parser.add_argument('--num_calibration_bin', help='Number of bins to compute calibration error', default=25,
                        type=int)
    # TP validation threshold
    parser.add_argument('--tau', help='TP validation threshold', default=0.10, type=float)

    # DAQ PARAMETERS: alpha and beta refer to the weights of the components while computing harmonic mean in IDQ and DAQ.
    # In our paper, we weight them equally, corresponding to their default settings here.
    parser.add_argument('--alpha', help='The weights of OOD, ID and Domain Shifted splits to compute DAQ', default=[1, 1, 1])
    parser.add_argument('--beta', help='The weights of LaECE and LRP to compute IDQ', default=0.5, type=float)

    ## SUPPORTED PLOTS
    # Plot reliability diagrams
    parser.add_argument('--plot_reliability_diagram', help='Whether or not to save reliability diagram', default=False)

    args = parser.parse_args()

    return args

def get_paths(config):
        
    prefix = 'detections/' + config

    if 'coco' in config:
        # Annotation files
        val_annFile = 'data/coco/annotations/instances_val2017.json'
        test_annFile = 'data/saod/annotations/obj45k.json'
        corr_annFile = 'data/saod/annotations/obj45k_corr.json'

        det_files = {'val': [prefix + '_severity_0.pkl.bbox.json',
                             prefix + '_pseudoid_severity_0.pkl.bbox.json',
                             prefix + '_pseudoood_severity_0.pkl.bbox.json'],
                     'test': prefix + '_obj45k_severity_0.pkl.bbox.json',
                     'corr': [prefix + '_obj45k_severity_1.pkl.bbox.json',
                              prefix + '_obj45k_severity_3.pkl.bbox.json',
                              prefix + '_obj45k_severity_5.pkl.bbox.json'],
                     'ood': prefix + '_sinobj110kood_severity_0.pkl.bbox.json'}

        # They match with detection files above
        dataset_size = {'val': [5000, 4952, 5000],
                        'test': 45391,
                        'corr': 45391 * 3,
                        'ood': 110428}

    else:
        # Annotation files
        val_annFile = 'data/saod/annotations/saod_av_val.json'
        test_annFile = 'data/saod/annotations/bdd45k.json'
        corr_annFile = 'data/saod/annotations/bdd45k_corr.json'

        det_files = {'val': [prefix + '_severity_0.pkl.bbox.json',
                             prefix + '_pseudoid_severity_0.pkl.bbox.json',
                             prefix + '_pseudoood_severity_0.pkl.bbox.json'],
                     'test': prefix + '_bdd45k_severity_0.pkl.bbox.json',
                     'corr': [prefix + '_bdd45k_severity_1.pkl.bbox.json',
                              prefix + '_bdd45k_severity_3.pkl.bbox.json',
                              prefix + '_bdd45k_severity_5.pkl.bbox.json'],
                     'ood': prefix + '_sinobj110kood_severity_0.pkl.bbox.json'}

        # They match with detection files above
        dataset_size = {'val': [16445, 14345, 16445],
                        'test': 44831,
                        'corr': 44831 * 3,
                        'ood': 110428}

    return val_annFile, test_annFile, corr_annFile, det_files, dataset_size


def calibrate_and_evaluate(calibration_error, gt, id_data, calibration_method, calibration_model,
                           num_calibration_bin, tau, save_plot, detection_level_threshold,
                           dataset_classes):
    detections = load_detections_from_file(id_data)
    detections = threshold_detections(detections, detection_level_threshold, dataset_classes)

    print('------ACCURACY------')
    # Calibrate val data detections
    detections = calibration_error.calibrate(detections, calibration_method, calibration_model, dataset_classes)

    # Compute calibration error and save reliability diagram
    calibration_error = Calibration(num_calibration_bin, tau, gt, detections, 'bbox')
    calibration_error.evaluate()
    calibration_error.prepare_input()
    calibration_error.compute_single_errors()
    calibration_error.compute_LRP()

    print(f'LRP = {np.nanmean(calibration_error.lrps["lrp"]) * 100:.1f}')
    print(f'LRP Loc = {np.nanmean(calibration_error.lrps["lrp_loc"]) * 100:.1f}')
    print(f'LRP FP = {np.nanmean(calibration_error.lrps["lrp_fp"]) * 100:.1f}')
    print(f'LRP FN = {np.nanmean(calibration_error.lrps["lrp_fn"]) * 100:.1f}')

    print('------CALIBRATION------')
    LaECE = calibration_error.accumulate_errors()
    print(f'LaECE= {LaECE * 100:.1f}')

    if save_plot:
        calibration_error.plot_reliability_diagram(LaECE)


def get_corrupted_image_ids(sev_idx, image_list):
    offset = 1e10
    all_valid_img = []
    for img in image_list:
        all_valid_img.append(int(img + (offset * (sev_idx + 1))))
    return all_valid_img


def threshold_detections(detections, detection_level_threshold, dataset_classes):
    del_items = []
    for idx, detection in enumerate(detections):
        if detection['score'] < detection_level_threshold[dataset_classes.index(detection['category_id'])]:
            del_items.append(idx)
    for idx in sorted(del_items, reverse=True):
        del detections[idx]
    return detections


def COCO_evaluation(annFile, detections, valid_img=None, remove_img=None, tau=None):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(detections)
    id_evaluator = COCOeval(cocoGt, cocoDt, 'bbox')
    if tau:
        id_evaluator.params.iouThrs = np.array([tau])
    if remove_img is not None:
        id_evaluator.params.imgIds = list(set(id_evaluator.params.imgIds).difference(remove_img))
    elif valid_img is not None:
        id_evaluator.params.imgIds = list(valid_img)
    id_evaluator.evaluate()
    id_evaluator.accumulate()
    id_evaluator.summarize()
    return id_evaluator


def get_detection_thresholds(annFile, detections, thr, tau=0.10):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(detections)
    id_evaluator = COCOeval(cocoGt, cocoDt, 'bbox')
    id_evaluator.params.areaRng = [id_evaluator.params.areaRng[0]]
    id_evaluator.params.areaRngLbl = ['all']
    id_evaluator.params.iouThrs = np.array([tau])
    id_evaluator.params.maxDets = [100]

    id_evaluator.evaluate()
    id_evaluator.accumulate()

    # LRP-Optimal Thresholds
    if thr == -1:
        print('Obtaining detection-level threshold using LRP-optimal thresholds...')
        return id_evaluator.eval['lrp_opt_thr'].squeeze()
    else:
        print('Obtaining detection-level threshold using a fixed confidence score...')
        return np.ones(len(id_evaluator.eval['lrp_opt_thr'])) * thr

def load_detections_from_file(path):
    f = open(path)
    detections = json.load(f)
    f.close()
    return detections
