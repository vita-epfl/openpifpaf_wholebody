import logging
import json
import zipfile
import copy

import numpy as np
from .evaluation_wb import test_body, test_face, test_foot, test_lefthand, test_righthand, test_wholebody

from openpifpaf.metric.base import Base

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    COCOeval = None

LOG = logging.getLogger(__name__)

class WholeBodyMetric_wb_protocoll(Base):
    text_labels_keypoints = ['AP', 'AP0.5', 'AP0.75', 'APM', 'APL',
                             'AR', 'AR0.5', 'AR0.75', 'ARM', 'ARL']
    text_labels_bbox = ['AP', 'AP0.5', 'AP0.75', 'APS', 'APM', 'APL',
                        'ART1', 'ART10', 'AR', 'ARS', 'ARM', 'ARL']

    def __init__(self, coco, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0,
                 keypoint_oks_sigmas=None):
        super().__init__()

        if category_ids is None:
            category_ids = [1]
        
        self.max_per_image = max_per_image
        self.category_ids = category_ids
        self.iou_type = iou_type
        self.small_threshold = small_threshold
        self.keypoint_oks_sigmas = keypoint_oks_sigmas
        
        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.coco = pycocotools.coco.COCO(coco)
        

        if self.iou_type == 'keypoints':# or self.iou_type == 'body_and_foot_keypoints' or self.iou_type == 'only_body_keypoints':
            self.text_labels = self.text_labels_keypoints
        elif self.iou_type == 'bbox':
            self.text_labels = self.text_labels_bbox
        else:
            LOG.warning('Unknown iou type "%s". Specify text_labels yourself.', self.iou_type)

        LOG.debug('max = %d, category ids = %s, iou_type = %s',
                  self.max_per_image, self.category_ids, self.iou_type)

    def _stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        coco_eval = self.coco.loadRes(predictions)
                
        for count in coco_eval.anns:
            ann_orig = copy.deepcopy(coco_eval.anns[count])
            coco_eval.anns[count]["lefthand_kpts"] = ann_orig["keypoints"][91*3:112*3]
            coco_eval.anns[count]["righthand_kpts"] = ann_orig["keypoints"][112*3:133*3]
            coco_eval.anns[count]["face_kpts"] = ann_orig["keypoints"][23*3:91*3]
            coco_eval.anns[count]["foot_kpts"] = ann_orig["keypoints"][17*3:23*3]
            coco_eval.anns[count]["keypoints"] = ann_orig["keypoints"][0:17*3] 
        
        test_body(self.coco, coco_eval)
        test_foot(self.coco, coco_eval)
        test_face(self.coco, coco_eval)
        test_lefthand(self.coco, coco_eval)
        test_righthand(self.coco, coco_eval)
        stats = test_wholebody(self.coco, coco_eval)
        
        return stats

    # pylint: disable=unused-argument
    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        image_id = int(image_meta['image_id'])
        self.image_ids.append(image_id)

        if self.small_threshold:
            predictions = [pred for pred in predictions
                           if pred.scale(v_th=0.01) >= self.small_threshold]
        if len(predictions) > self.max_per_image:
            predictions = predictions[:self.max_per_image]

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            pred_data['image_id'] = image_id
            pred_data = {
                k: v for k, v in pred_data.items()
                if k in ('category_id', 'score', 'keypoints', 'bbox', 'image_id')
            }
            image_annotations.append(pred_data)

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            n_keypoints = 133
            image_annotations.append({
                'image_id': image_id,
                'category_id': 1,
                'keypoints': np.zeros((n_keypoints * 3,)).tolist(),
                'bbox': [0, 0, 1, 1],
                'score': 0.001,
            })

        if LOG.getEffectiveLevel() == logging.DEBUG:
            self._stats(image_annotations, [image_id])
            LOG.debug(image_meta)

        self.predictions += image_annotations

    def write_predictions(self, filename, *, additional_data=None):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        predictions_wb = copy.deepcopy(predictions)
        for ann in predictions_wb:
            ann_orig = copy.deepcopy(ann)
            ann["lefthand_kpts"] = ann_orig["keypoints"][91*3:112*3]
            ann["righthand_kpts"] = ann_orig["keypoints"][112*3:133*3]
            ann["face_kpts"] = ann_orig["keypoints"][23*3:91*3]
            ann["foot_kpts"] = ann_orig["keypoints"][17*3:23*3]
            ann["keypoints"] = ann_orig["keypoints"][0:17*3]
        with open(filename + '.pred_wb.json', 'w') as f:
            json.dump(predictions_wb, f)
        LOG.info('wrote %s.pred_wb.json', filename)
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)

    def stats(self):
        data = {
            'stats': self._stats().tolist(),
            'text_labels': self.text_labels,
        }

        return data
