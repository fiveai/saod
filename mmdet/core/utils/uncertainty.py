# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

class Uncertainty:
    def entropy(self, logits):

        # Compute entropy given logits

        # Logits to classification scores
        scores = F.softmax(logits, dim=-1)

        entropy = -torch.sum(scores * torch.log(scores), dim = 1)

        return entropy

    def max_class_entropy(self, logits):

        # Compute entropy of the maximum scoring class for sigmoid-based detectors

        # Logits to classification scores
        scores, _ = logits.sigmoid().max(dim=-1)

        entropy = -(scores * torch.log(scores) + (1 - scores) * torch.log(1 - scores))

        return entropy

    def avg_entropy(self, logits):

        # Compute average entropy of scores for sigmoid-based detectors

        # Logits to classification scores
        scores = logits.sigmoid()

        entropy = -(scores * torch.log(scores) + (1 - scores) * torch.log(1 - scores))

        return entropy.mean(dim=-1)

    def ds(self, logits):

        # Compute dempster-shafer

        # Logits to classification scores
        K = logits.shape[1]

        ds = K / (K + torch.sum(torch.exp(logits), dim = 1))

        return ds

    def gaussian_entropy(self, covariance, D=4, pi = 3.14159265359):

        # Compute entropy of predicted gaussian bounding boxes

        # Assume diagonal prediction
        determinant = torch.prod(covariance, -1)

        sp_entropy = (D / 2) * (1 + np.log(2 * pi)) + (1 / 2) * torch.log(determinant)

        return sp_entropy

    def determinant(self, covariance):

        # Compute determinant of predicted gaussian bounding boxes

        # Assume diagonal prediction
        determinant = torch.prod(covariance, -1)

        return determinant

    def trace(self, covariance):

        # Compute trace of predicted gaussian bounding boxes

        # Assume diagonal prediction
        trace = torch.sum(covariance, -1)

        return trace

    def get_uncertainties(self, uncertainties, logits=None, covariance=None):
        result = {}
        if "cls_type" in uncertainties and logits is not None:
            result['cls'] = {}
            for unc in uncertainties['cls_type']:
                unc_func = getattr(self, unc)
                result['cls'][unc] = unc_func(self, logits)

        if "loc_type" in uncertainties and covariance is not None:
            result['loc'] = {}
            for unc in uncertainties['loc_type']:
                unc_func = getattr(self, unc)
                result['loc'][unc] = unc_func(self, covariance)

        return result

