import json
import pdb
import mmdet.datasets.coco as coco


paths = [	'detections/faster_rcnn_r50_fpn_1x_all_id_cov_shift_robustness_severity_0.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_id_cov_shift_robustness_severity_1.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_id_cov_shift_robustness_severity_2.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_id_cov_shift_robustness_severity_3.pkl.bbox.json']


paths = [	'detections/faster_rcnn_r50_fpn_1x_all_ood_mixup_robustness_severity_0.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_ood_mixup_robustness_severity_1.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_ood_mixup_robustness_severity_2.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_ood_mixup_robustness_severity_3.pkl.bbox.json',
			'detections/faster_rcnn_r50_fpn_1x_all_ood_mixup_robustness_severity_4.pkl.bbox.json']			

output_path = 'detections/general_od_ood.pkl.bbox.json'

OFFSET = 1000000
all_detections = []
for i, path in enumerate(paths):
	f = open(path)
	detections = json.load(f)
	for detection in detections:
		detection['image_id'] += int(i*OFFSET)
	all_detections.extend(detections)
	f.close()

with open(output_path, 'w') as outfile:
  json.dump(all_detections, outfile)
