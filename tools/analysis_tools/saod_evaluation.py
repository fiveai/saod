import torch

from saod_utils import *

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as sk
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

class Calibration(COCOeval):
    def __init__(self, bin_count=25, tau=0.10, *args):
        gt = COCO(args[0])
        super(Calibration, self).__init__(gt, gt.loadRes(args[1]), args[2])

        self.params.areaRng = [self.params.areaRng[0]]
        self.params.areaRngLbl = ['all']
        self.params.iouThrs = np.array([tau])
        self.params.maxDets = [100]
        self.tau = tau

        self.bin_count = bin_count
        self.bins = np.linspace(0.0, 1.0, self.bin_count + 1, endpoint=True)
        self.calibration_info = dict()

        self.errors = np.zeros([len(self.params.catIds), self.bin_count])
        self.weights_per_bin = np.zeros([len(self.params.catIds), self.bin_count])
        self.prec_iou = np.zeros([len(self.params.catIds), self.bin_count])

        self.lrps = {'lrp': np.zeros(len(self.params.catIds)) - 1, 'lrp_loc': np.zeros(len(self.params.catIds)) - 1,
                     'lrp_fp': np.zeros(len(self.params.catIds)) - 1, 'lrp_fn': np.zeros(len(self.params.catIds)) - 1}

    def prepare_input(self, p=None):
        '''
        Accumulate per image evaluation results and
        store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating detections for calibration error...')
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            self.calibration_info[k] = dict()
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly
                    # different results.
                    # mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                           inds]
                    dtIoU = np.concatenate(
                        [e['dtIoUs'][:, 0:maxDet] for e in E], axis=1)[:, inds]

                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    self.calibration_info[k]['scores'] = dtScoresSorted
                    self.calibration_info[k]['tps'] = np.logical_and(dtm, np.logical_not(dtIg))[0]
                    self.calibration_info[k]['fps'] = np.logical_and(np.logical_not(dtm),
                                                                     np.logical_not(dtIg))[0]
                    self.calibration_info[k]['iou'] = np.multiply(dtIoU, self.calibration_info[k]['tps'])[0]
                    self.calibration_info[k]['npig'] = npig

    def compute_single_errors(self):
        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                continue

            # Find total number of valid detections for this class
            total_det = cl_input['tps'].sum() + cl_input['fps'].sum()

            # If no detection, then ignore
            if total_det == 0:
                continue

            for i in range(self.bin_count):
                # Find detections in this bin
                bin_all_det = np.logical_and(self.bins[i] <= cl_input['scores'],
                                             cl_input['scores'] < self.bins[i + 1])
                bin_tps = np.logical_and(cl_input['tps'], bin_all_det)
                bin_fps = np.logical_and(cl_input['fps'], bin_all_det)
                bin_det = np.logical_or(bin_tps, bin_fps)
                bin_scores = cl_input['scores'][bin_det]
                bin_ious = cl_input['iou'][bin_tps]

                # Count number of tps in this bin
                num_tp = bin_tps.sum()

                # Count number of fps in this bin
                num_fp = bin_fps.sum()

                # Count number of detections in this bin
                num_det = num_tp + num_fp

                if num_det == 0:
                    self.errors[cl, i] = np.nan
                    self.weights_per_bin[cl, i] = 0
                    self.prec_iou[cl, i] = np.nan
                    continue

                # Find error
                if len(bin_ious) > 0:
                    norm_iou = bin_ious
                    norm_total_iou = norm_iou.sum()
                else:
                    norm_total_iou = 0

                self.prec_iou[cl, i] = norm_total_iou / num_det

                # Average of Scores in this bin
                mean_score = bin_scores.mean()
                self.errors[cl, i] = np.abs(self.prec_iou[cl, i] - mean_score)

                # Weight of the bin
                self.weights_per_bin[cl, i] = num_det / total_det

    def histogram_binning(self):
        out_bins = dict()
        out_scores = dict()
        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                out_bins[cl] = np.zeros(0)
                out_scores[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                out_bins[cl] = np.zeros(0)
                out_scores[cl] = np.zeros(0)
                continue

            # If there are very few detections from a class,
            # then decrease the number of bins.
            if num_valid_dets < self.bin_count*2:
                bin_count = np.int32(num_valid_dets / 2)
            else:
                bin_count = self.bin_count

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets]
            # Note that scores are already sorted
            valid_labels = cl_input['iou'][valid_dets]

            # initialization
            bin_det_count = (num_valid_dets + 1) / bin_count

            A = np.arange(0, num_valid_dets, bin_det_count)
            A = np.int32(np.ceil(np.append(A, num_valid_dets)))
            out_scores[cl] = np.zeros(0)
            out_bins[cl] = np.zeros(0)

            for i in range(bin_count):
                low = A[i]
                high = A[i + 1]
                out_scores[cl] = np.append(out_scores[cl], valid_labels[low:high].mean())
                out_bins[cl] = np.append(out_bins[cl], valid_scores[low])
            out_bins[cl] = np.append(out_bins[cl], valid_scores[-1])
            out_bins[cl][0] = 1
            out_bins[cl][-1] = 0

        return tuple([out_scores, out_bins])

    def linear_regression(self):
        linear_regression = dict()
        tps = np.zeros(len(self.calibration_info.items()))
        all_dets = np.zeros(len(self.calibration_info.items()))
        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                linear_regression[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            tps[cl] = cl_input['tps'].sum()
            num_valid_dets = valid_dets.sum()
            all_dets[cl] = num_valid_dets

            # If no detection, then ignore
            if num_valid_dets == 0:
                linear_regression[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            # Note that scores are already sorted
            valid_labels = cl_input['iou'][valid_dets]

            linear_regression[cl] = LinearRegression().fit(valid_scores, valid_labels)

        return linear_regression


    def isotonic_regression(self):
        isotonic_regression = dict()
        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                isotonic_regression[cl] = np.zeros(0)
                continue

            # Find total number of valid detections for this class
            valid_dets = np.logical_or(cl_input['tps'], cl_input['fps'])
            num_valid_dets = valid_dets.sum()

            # If no detection, then ignore
            if num_valid_dets == 0:
                isotonic_regression[cl] = np.zeros(0)
                continue

            # Note that scores and ious are sorted wrt scores
            valid_scores = cl_input['scores'][valid_dets].reshape((-1, 1))
            # Note that scores are already sorted
            valid_labels = cl_input['iou'][valid_dets]

            isotonic_regression[cl] = IsotonicRegression(y_min=0., y_max=1.).fit(valid_scores, valid_labels)
        return isotonic_regression

    def identity(self):
        return None

    def calibrate(self, detections, calibration_type, calibration_model, dataset_classes):
        # Identity
        if calibration_model is None:
            return detections
        # Calibrators
        for detection in detections:
            cl = dataset_classes.index(detection['category_id'])
            if calibration_type == 'histogram_binning':
                if len(calibration_model[0][cl]) > 0:
                    for j, bin_high_score in enumerate(calibration_model[1][cl]):
                        if detection['score'] > bin_high_score:
                            this_bin = j - 1
                            break
                    detection['score'] = calibration_model[0][cl][this_bin]
            elif calibration_type == 'linear_regression' or calibration_type == 'isotonic_regression':
                if type(calibration_model[cl]) is not np.ndarray:
                    detection['score'] = \
                        np.clip(calibration_model[cl].predict(np.array(detection['score']).reshape(-1, 1)), 0, 1)[0]

        return detections

    def accumulate_errors(self):
        # print('Class-wise LaECEs:', repr(np.nansum(self.weights_per_bin * self.errors, axis=1)))
        LaECE = np.nanmean(np.nansum(self.weights_per_bin * self.errors, axis=1))
        return LaECE

    def plot_reliability_diagram(self, LaECE, cl=-1, fontsize=16):
        delta = 1.0 / self.bin_count
        x = np.arange(0, 1, delta)

        if cl == -1:
            bin_acc = np.nanmean(self.prec_iou, axis=0)
            bin_weights = np.nanmean(self.weights_per_bin, axis=0)
        else:
            bin_acc = self.prec_iou[cl]
            bin_weights = self.weights_per_bin[cl]
        nan_idx = (bin_weights == 0)
        bin_acc[nan_idx] = 0

        # size and axis limits
        plt.figure(figsize=(5, 5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)
        # plot bars and identity line
        plt.bar(x, bin_acc, color='b', width=delta, align='edge', edgecolor='k',
                label=r'Precision $ \times \mathrm{\bar{IoU}}$',
                zorder=5)
        plt.bar(x, bin_weights, color='mistyrose', alpha=0.5, width=delta, align='edge',
                edgecolor='r', hatch='/', label='% of Samples', zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
        # labels and legend
        plt.xlabel('Confidence', fontsize=fontsize)
        plt.legend(loc='upper left', framealpha=1.0, fontsize=fontsize)
        plt.text(0.05, 0.70, 'LaECE= %.1f%%' % (LaECE * 100), fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        plt.show()

    def compute_LRP(self):
        for cl, cl_input in self.calibration_info.items():
            # For corrupted images, all images with this class is already rejected
            if 'tps' not in cl_input.keys():
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = np.nan
                self.lrps['lrp'][cl] = np.nan
                continue

            # Find total number of valid detections for this class
            tp_num = cl_input['tps'].sum()
            fp_num = cl_input['fps'].sum()
            fn_num = cl_input['npig'] - tp_num

            # If there is detection
            if tp_num + fp_num > 0:
                # There is some TPs
                if tp_num > 0:
                    total_loc = tp_num - cl_input['iou'].sum()
                    self.lrps['lrp'][cl] = (total_loc / (1 - self.tau) + fp_num +
                                            fn_num) / (tp_num + fp_num + fn_num)
                    self.lrps['lrp_loc'][cl] = total_loc / tp_num
                    self.lrps['lrp_fp'][cl] = fp_num / (tp_num + fp_num)
                    self.lrps['lrp_fn'][cl] = fn_num / cl_input['npig']
                else:
                    self.lrps['lrp_loc'][cl] = np.nan
                    self.lrps['lrp_fp'][cl] = np.nan
                    self.lrps['lrp_fn'][cl] = 1.
                    self.lrps['lrp'][cl] = 1.
            else:
                self.lrps['lrp_loc'][cl] = np.nan
                self.lrps['lrp_fp'][cl] = np.nan
                self.lrps['lrp_fn'][cl] = 1.
                self.lrps['lrp'][cl] = 1.


class UncertaintyProcessor:
    def __init__(self, aggregation, max_det_num, cls_unc_type, loc_unc_type,
                 detfile, dataset_size, image_level_threshold):
        self.aggregation = aggregation
        self.max_det_num = max_det_num
        self.cls_unc_type = cls_unc_type
        self.loc_unc_type = loc_unc_type
        self.detfile = detfile
        self.dataset_size = dataset_size
        self.image_level_threshold = image_level_threshold

        self.MAX_UNC = 1e12

    def get_uncertainty(self, detections):
        # Return a dict with
        # image_id: [list of uncertainties]
        uncertainty = dict()
        for detection in detections:
            if detection['image_id'] in uncertainty.keys():
                uncertainty[detection['image_id']].append(
                    [1 - detection['score'], *detection['uncertainty'], detection['category_id']])
            else:
                uncertainty[detection['image_id']] = list(
                    [[1 - detection['score'], *detection['uncertainty'], detection['category_id']]])
        return uncertainty

    def boxes_to_image_unc(self, detections, bounds=None):
        # TO DO: Implement image_ids and uncertainty as a dictionary
        image_ids = []
        uncertainty = []

        for img_id, val in detections.items():
            values = np.array(val)
            # Use score or computed uncertainty
            if self.cls_unc_type != -1:
                cls_unc = values[:, self.cls_unc_type]
                if bounds is not None:
                    cls_unc = (cls_unc - bounds[0]) / (bounds[1] - bounds[0])
            if self.loc_unc_type != -1:
                loc_unc = values[:, self.loc_unc_type]
                if bounds is not None:
                    loc_unc = (loc_unc - bounds[2]) / (bounds[3] - bounds[2])

            if self.cls_unc_type == -1:
                unc = loc_unc
            elif self.loc_unc_type == -1:
                unc = cls_unc

            # Use all detections
            if self.max_det_num == -1:
                uncertainty.append(np.sum(unc)) if self.aggregation == 'sum' else uncertainty.append(np.mean(unc))
                image_ids.append(img_id)
            # Use top-k detections
            elif self.max_det_num > 0:
                # Get ids of top-k
                if len(unc) > self.max_det_num:
                    idx = np.argpartition(unc, self.max_det_num)[:self.max_det_num]
                    unc = unc[idx]
                uncertainty.append(np.sum(unc)) if self.aggregation == 'sum' else uncertainty.append(np.mean(unc))
                image_ids.append(img_id)

        return uncertainty, image_ids

    def get_id_ood_labels(self, id_unc, ood_unc, id_size, ood_size):
        # If there is no detection in, then label as OOD
        add_id = [self.MAX_UNC] * (id_size - len(id_unc))
        id_unc.extend(add_id)

        add_ood = [self.MAX_UNC] * (ood_size - len(ood_unc))
        ood_unc.extend(add_ood)

        pos = -np.array(id_unc).reshape((-1, 1))
        neg = -np.array(ood_unc).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(id_unc)] += 1
        return labels, examples

    def get_ood_threshold(self, labels, examples):
        if self.image_level_threshold == -1:
            print('Obtaining Image-level threshold using pseudo-OOD approach...')
            # Get OOD threshold
            detections_withgt = load_detections_from_file(self.detfile['val'][1])
            uncertainty_withgt = self.get_uncertainty(detections_withgt)
            # Get image-level uncertainties from detection-level uncertainties
            unc_withgt, _ = self.boxes_to_image_unc(uncertainty_withgt)

            # Get OOD threshold
            detections_withnogt = load_detections_from_file(self.detfile['val'][2])
            uncertainty_withnogt = self.get_uncertainty(detections_withnogt)
            # Get image-level uncertainties from detection-level uncertainties
            unc_withnogt, _ = self.boxes_to_image_unc(uncertainty_withnogt)

            labels, examples = self.get_id_ood_labels(unc_withgt, unc_withnogt, self.dataset_size['val'][1],
                                                      self.dataset_size['val'][2])

            fpr, tpr, thr = sk.roc_curve(labels, examples)
            spec = 1 - fpr
            f_score = stats.hmean([tpr, spec], axis=0)
            idx = np.argmax(f_score)
            thr = -thr[idx]
        else:
            print('Obtaining Image-level threshold using a fixed TPR threshold...')
            idx = int(self.image_level_threshold * len(labels))
            thr = np.sort(-examples)[idx]

        # print(f'Score threshold = {(1 - thr) * 100 :.1f}')
        return thr


class BaselineSAOD:
    def train(self, detfile, annFile, dataset_size, processunc, dataset_classes, num_calibration_bin, tau,
              calibrate, thr=-1):
        print('----------MAKING OBJECT DETECTOR SELF-AWARE-------------')
        # Get OOD threshold
        detections_val = load_detections_from_file(detfile[1])
        uncertainty_val = processunc.get_uncertainty(detections_val)

        # Get image-level uncertainties from detection-level uncertainties
        unc_validation, _ = processunc.boxes_to_image_unc(uncertainty_val)

        labels, examples = processunc.get_id_ood_labels(unc_validation, [], dataset_size[1], 0)

        # Compute f-score and TP@FP
        print('----------1. OBTAINING IMAGE-LEVEL REJECTION THRESHOLD-------------')
        thr_image = processunc.get_ood_threshold(labels, examples)

        # Get LRP-Optimal Thresholds on all images
        print('----------2. OBTAINING DETECTION-LEVEL CONFIDENCE SCORE THRESHOLD-------------')
        thr_detections = get_detection_thresholds(annFile, detections_val, thr)
        
        print('------3. TRAINING CALIBRATOR------')
        # remove detections
        detections = threshold_detections(detections_val, thr_detections, dataset_classes)

        # Get calibrator
        calibration_error = Calibration(num_calibration_bin, tau, annFile, detections, 'bbox')

        calibration_error.evaluate()
        calibration_error.prepare_input()

        calibrator = getattr(calibration_error, calibrate)
        print('----------OBJECT DETECTOR IS MADE SELF-AWARE!-------------')

        # Get calibration model
        calibration_model = calibrator()

        return thr_image, thr_detections, calibration_model

    def inference(self, detfile, thr_image, thr_detections, calibration_model, calibration_type, process_unc,
                  dataset_classes, ood=False):
        # Load detections
        top_100_detections = load_detections_from_file(detfile)

        detection_level_uncertainty = process_unc.get_uncertainty(top_100_detections)

        # Get image-level uncertainties from detection-level uncertainties
        image_level_uncertainty, image_ids = process_unc.boxes_to_image_unc(detection_level_uncertainty)

        # Threshold images from image uncertainty threshold
        valid_imgs = image_level_uncertainty < thr_image
        valid_image_ids = set(np.asarray(image_ids)[valid_imgs])

        # No need to modify detections if OOD
        if ood:
            return image_level_uncertainty, top_100_detections, valid_image_ids
        else:
            # Only keep detections from surviving images with thresholds larger than the specified threshold
            valid_detections = []
            for idx, detection in enumerate(top_100_detections):
                cl = dataset_classes.index(detection['category_id'])
                if detection['image_id'] in valid_image_ids and detection['score'] >= thr_detections[cl]:
                    if calibration_type == 'histogram_binning':
                        if len(calibration_model[0][cl]) > 0:
                            for j, bin_high_score in enumerate(calibration_model[1][cl]):
                                if detection['score'] > bin_high_score:
                                    this_bin = j - 1
                                    break
                            detection['score'] = calibration_model[0][cl][this_bin]
                    elif calibration_type == 'linear_regression' or calibration_type == 'isotonic_regression':
                        if type(calibration_model[cl]) is not np.ndarray:
                            detection['score'] = \
                                np.clip(calibration_model[cl].predict(np.array(detection['score']).reshape(-1, 1)), 0,
                                        1)[0]
                    valid_detections.append(detection)
            return image_level_uncertainty, valid_detections, valid_image_ids

class Benchmark:
    def __init__(self, alpha, beta, thr_image, with_corr=True):
        # This class populates main performance measures
        self.alpha = alpha
        self.beta = beta
        self.performance = {}
        performance_keys = ['DAQ', 'BA', 'TPR', 'TNR', 'AUROC', 'IDQ', 'LaECE', 'LRP', 'LRP_Loc', 'LRP_FP', 'LRP_FN',
                            'IDQ_C', 'LaECE_C', 'LRP_C', 'LRP_Loc_C', 'LRP_FP_C', 'LRP_FN_C', 'ACCEPT_RATE_C']
        self.reset_performance(performance_keys)
        self.thr_image = thr_image
        self.with_corr = with_corr
        if not self.with_corr:
            # If not using corrupted images just make the weight of corrupted images 0
            self.alpha[2] = 0

    def summarize(self):
        print('----------OVERALL PERFORMANCE-------------')
        print(f'DAQ = {self.performance["DAQ"] * 100:.1f}')
        print('------------OOD PERFORMANCE---------------')
        print(f'BA = {self.performance["BA"] * 100:.1f}')
        print(f'TPR = {self.performance["TPR"] * 100:.1f}')
        print(f'TNR = {self.performance["TNR"] * 100:.1f}')
        print(f'AUROC = {self.performance["AUROC"] * 100:.1f}')
        print('------------ID PERFORMANCE---------------')
        print(f'IDQ = {self.performance["IDQ"] * 100:.1f}')
        print(f'LaECE = {self.performance["LaECE"] * 100:.1f}')
        print(f'LRP = {self.performance["LRP"] * 100:.1f}')
        print(f'LRP Loc = {self.performance["LRP_Loc"] * 100:.1f}')
        print(f'LRP FP = {self.performance["LRP_FP"] * 100:.1f}')
        print(f'LRP FN = {self.performance["LRP_FN"] * 100:.1f}')
        if self.with_corr:
            print('------------CORRUPTION PERFORMANCE---------------')
            print(f'IDQ = {self.performance["IDQ_C"] * 100:.1f}')
            print(f'LaECE = {self.performance["LaECE_C"] * 100:.1f}')
            print(f'LRP = {self.performance["LRP_C"] * 100:.1f}')
            print(f'LRP Loc = {self.performance["LRP_Loc_C"] * 100:.1f}')
            print(f'LRP FP = {self.performance["LRP_FP_C"] * 100:.1f}')
            print(f'LRP FN = {self.performance["LRP_FN_C"] * 100:.1f}')
            print(f'ACCEPT \% = {self.performance["ACCEPT_RATE_C"] * 100:.1f}')

    def reset_performance(self, performance_keys):
        for key in performance_keys:
            self.performance[key] = -1

    def getLRP(self, calibration_error, valid_img=None, corr=False):
        if valid_img is not None:
            calibration_error.params.imgIds = list(valid_img)

        calibration_error.evaluate()
        calibration_error.prepare_input()
        calibration_error.compute_single_errors()
        calibration_error.compute_LRP()
        if corr:
            self.performance["LRP_C"] = np.nanmean(calibration_error.lrps["lrp"])
            self.performance["LRP_Loc_C"] = np.nanmean(calibration_error.lrps["lrp_loc"])
            self.performance["LRP_FP_C"] = np.nanmean(calibration_error.lrps["lrp_fp"])
            self.performance["LRP_FN_C"] = np.nanmean(calibration_error.lrps["lrp_fn"])
        else:
            self.performance["LRP"] = np.nanmean(calibration_error.lrps["lrp"])
            self.performance["LRP_Loc"] = np.nanmean(calibration_error.lrps["lrp_loc"])
            self.performance["LRP_FP"] = np.nanmean(calibration_error.lrps["lrp_fp"])
            self.performance["LRP_FN"] = np.nanmean(calibration_error.lrps["lrp_fn"])

    def getLaECE(self, calibration_error, corr=False):
        if corr:
            self.performance["LaECE_C"] = calibration_error.accumulate_errors()
        else:
            self.performance["LaECE"] = calibration_error.accumulate_errors()

    def ood_benchmark(self, labels, examples):
        fpr, tpr, thr = sk.roc_curve(labels, examples)
        spec = 1 - fpr
        ba = stats.hmean([tpr, spec], axis=0)
        idx = np.argmax(-thr > self.thr_image)
        self.performance["BA"] = ba[idx]
        self.performance["TPR"] = tpr[idx]
        self.performance["TNR"] = spec[idx]
        self.performance["AUROC"] = sk.roc_auc_score(labels, examples)

    def id_benchmark(self, detections, annFile, num_calibration_bin, tau):
        calibration_error = Calibration(num_calibration_bin, tau, annFile,
                                        detections, 'bbox')
        self.getLRP(calibration_error)
        self.getLaECE(calibration_error)

    def unify_detections(self, detection_list, valid_img_list):
        # Correct image ids in detection list and valid img list
        offset = 1e10
        all_dets = []
        for i, detections in enumerate(detection_list):
            for detection in detections:
                detection['image_id'] = int(detection['image_id'] + (offset * (i + 1)))
            all_dets.extend(detections)

        all_valid_img = []
        for i, valid_img in enumerate(valid_img_list):
            for img in valid_img:
                all_valid_img.append(int(img + (offset * (i + 1))))
        return all_dets, all_valid_img

    def corr_benchmark(self, detection_list, annfile, num_calibration_bin, tau,
                       valid_img_list, dataset_size):
        # We add 1e10 to image ids for severity 1, 2e10 to image ids for severity 2 and
        # 3e8 to image ids for severity 5
        detections, valid_img = self.unify_detections(detection_list, valid_img_list)
        self.performance["ACCEPT_RATE_C"] = (len(valid_img) - (dataset_size / 3) * 2) / (dataset_size / 3)

        calibration_error = Calibration(num_calibration_bin, tau, annfile,
                                        detections, 'bbox')
        self.getLRP(calibration_error, valid_img, corr=True)
        self.getLaECE(calibration_error, corr=True)

    def compute_IDQ(self, corr=False):
        if corr:
            self.performance["IDQ_C"] = 1 / (self.beta / (1 - self.performance["LRP_C"]) + (1 - self.beta) /
                                             (1 - self.performance["LaECE_C"]))
        else:
            self.performance["IDQ"] = 1 / (self.beta / (1 - self.performance["LRP"]) + (1 - self.beta) /
                                           (1 - self.performance["LaECE"]))

    def compute_DAQ(self):
        self.compute_IDQ()
        if self.with_corr:
            self.compute_IDQ(corr=True)
        self.performance["DAQ"] = (self.alpha[0] + self.alpha[1] + self.alpha[2]) / \
                                  (self.alpha[0] / self.performance["BA"] + self.alpha[1] / self.performance["IDQ"] +
                                   self.alpha[2] / self.performance["IDQ_C"])


def main():
    args = parse_args()

    # Get detection and annotation paths for the datasets
    val_annFile, test_annFile, corr_annFile, det_files, dataset_size = get_paths(args.config)
    dataset_classes = list(COCO(val_annFile).cats.keys())

    ## OPTION 1: Obtain a Self-aware Object Detector and Evaluate it
    if args.benchmark:
        # Initialize Uncertainty Processor
        process_unc = UncertaintyProcessor(args.aggregation, args.max_det_num,
                                           args.cls_unc_type, args.loc_unc_type,
                                           det_files, dataset_size,
                                           args.image_level_threshold)
        # Initialize baseline self-aware object detector
        baseline = BaselineSAOD()

        # Train refers to: Get image-level threshold, detection-level threshold and calibrators based on Alg. 1
        thr_image, thr_detections, calibrator = baseline.train(det_files['val'], val_annFile,
                                                               dataset_size['val'], process_unc,
                                                               dataset_classes, args.num_calibration_bin,
                                                               args.tau,
                                                               args.calibrate,
                                                               args.detection_level_threshold)

        # Run inference on ID split
        print('----------INFERENCE OF SELF-AWARE OBJECT DETECTOR-------------')
        print('----------1. INFERENCE ON IN-DISTRIBUTION IMAGES-------------')
        image_level_unc_test, detections_test, _ = baseline.inference(det_files['test'],
                                                                      thr_image,
                                                                      thr_detections, calibrator,
                                                                      args.calibrate,
                                                                      process_unc,
                                                                      dataset_classes)

        # Run inference on OOD split
        print('----------2. INFERENCE ON OUT-OF-DISTRIBUTION IMAGES-------------')
        image_level_unc_ood, detections_ood, _ = baseline.inference(
            det_files['ood'],
            thr_image,
            thr_detections, calibrator,
            args.calibrate, process_unc,
            dataset_classes, ood=True)

        labels, examples = process_unc.get_id_ood_labels(image_level_unc_test, image_level_unc_ood,
                                                         dataset_size['test'], dataset_size['ood'])

        evaluator = Benchmark(args.alpha, args.beta, thr_image, with_corr=args.benchmark_with_corr)

        evaluator.ood_benchmark(labels, examples)
        evaluator.id_benchmark(detections_test, test_annFile, args.num_calibration_bin, args.tau)

        if args.benchmark_with_corr:
            detections_corrupted = []
            valid_img_corrupted = []
            print('----------3. INFERENCE ON CORRUPTED IMAGES-------------')
            for sev_idx, corr_det_file in enumerate(det_files['corr']):
                _, detections, valid_image_ids = baseline.inference(corr_det_file, thr_image,
                                                                    thr_detections, calibrator,
                                                                    args.calibrate, process_unc,
                                                                    dataset_classes)
                detections_corrupted.append(detections)

                # If sev_idx is 0 or 1; then we consider them as In-distribution
                # For sev_idx = 2, the detector has the flexibility to reject
                if sev_idx == 0 or sev_idx == 1: 
                    valid_img_corrupted.append([img['id'] for img in COCO(test_annFile).dataset['images']])
                else:
                    valid_img_corrupted.append(valid_image_ids)

            evaluator.corr_benchmark(detections_corrupted, corr_annFile, args.num_calibration_bin,
                                     args.tau, valid_img_corrupted,
                                     dataset_size['corr'])

        print('----------EVALUATING THE SELF-AWARE OBJECT DETECTOR-------------')
        evaluator.compute_DAQ()
        evaluator.summarize()

    ## OPTION 2: Only Evaluate the Quality of the Image-level Uncertainty
    elif args.ood_evaluate:

        assert args.cls_unc_type > -1 or args.loc_unc_type > -1,\
            "Uncertainty type is not specified for OOD evaluation."

        # Initialize Uncertainty Processor
        process_unc = UncertaintyProcessor(args.aggregation, args.max_det_num,
                                           args.cls_unc_type, args.loc_unc_type,
                                           det_files, dataset_size,
                                           args.image_level_threshold)

        # Get uncertainties from detection files in json format
        detections_id = load_detections_from_file(det_files['test'])
        uncertainty_id = process_unc.get_uncertainty(detections_id)

        detections_ood = load_detections_from_file(det_files['ood'])
        uncertainty_ood = process_unc.get_uncertainty(detections_ood)

        # Get image-level uncertainties from detection-level uncertainties
        unc_id_vals, _ = process_unc.boxes_to_image_unc(uncertainty_id)
        unc_ood_vals, _ = process_unc.boxes_to_image_unc(uncertainty_ood)

        labels, examples = process_unc.get_id_ood_labels(unc_id_vals, unc_ood_vals, dataset_size['test'],
                                                         dataset_size['ood'])

        print('----------OOD Detection Performance-------------')
        auroc = sk.roc_auc_score(labels, examples)
        print(f'AUROC = {auroc * 100:.1f}')
    
    ## OPTION 3: Evaluate the Calibration Performance and Accuracy
    elif args.calibrate:

        assert args.calibrate in ['isotonic_regression', 'linear_regression','histogram_binning', 'identity'],\
            "Provided calibrator is not implemented"

        ## TRAIN CALIBRATOR ON VAL SET
        # Read detections
        detections = load_detections_from_file(det_files['val'][0])

        # get thresholds for classes
        args.detection_level_threshold = get_detection_thresholds(val_annFile, detections, args.detection_level_threshold)

        # modify detections
        detections = threshold_detections(detections, args.detection_level_threshold, dataset_classes)

        print('------EVALUATING LRP ERROR AND LaECE------')
        calibration_error = Calibration(args.num_calibration_bin, args.tau,
                                            val_annFile, detections, 'bbox')
        calibration_error.evaluate()
        calibration_error.prepare_input()
        calibration_method = getattr(calibration_error, args.calibrate)

        # Get calibration model
        calibration_model = calibration_method()

        ## EVALUATE CALIBRATION PERFORMANCE ON TEST SET
        # Load test ground truth and detections
        calibrate_and_evaluate(calibration_error, test_annFile, det_files['test'], args.calibrate, calibration_model,
                               args.num_calibration_bin, args.tau, args.plot_reliability_diagram,
                               args.detection_level_threshold, dataset_classes)

    ## OPTION 4: Standard COCO style evaluation for clean test file
    elif args.evaluate_top_100:
        # Here, we follow standard COCO Evaluation by using tau=[0.50:0.95,0.05] for AP and 0.50 for LRP
        COCO_evaluation(test_annFile, det_files['test'])
        
        return

if __name__ == '__main__':
    main()
