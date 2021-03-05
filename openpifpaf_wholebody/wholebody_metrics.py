import logging
import json
import zipfile

import numpy as np

from openpifpaf.metric.base import Base

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    COCOeval = None

LOG = logging.getLogger(__name__)

class WholeBodyMetric(Base):
    text_labels_keypoints = ['AP', 'AP0.5', 'AP0.75', 'APM', 'APL',
                             'AR', 'AR0.5', 'AR0.75', 'ARM', 'ARL']
    text_labels_bbox = ['AP', 'AP0.5', 'AP0.75', 'APS', 'APM', 'APL',
                        'ART1', 'ART10', 'AR', 'ARS', 'ARM', 'ARL']

    def __init__(self, coco, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0,
                 keypoint_oks_sigmas=None,
                 kps_select = "all"):
        super().__init__()

        if category_ids is None:
            category_ids = [1]
        
        self.max_per_image = max_per_image
        self.category_ids = category_ids
        self.iou_type = iou_type
        self.small_threshold = small_threshold
        self.keypoint_oks_sigmas = keypoint_oks_sigmas
        self.kps_select = kps_select

        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.coco = pycocotools.coco.COCO(coco)
# =============================================================================
#         if self.kps_select=='only_body':
#             # assert len(self.coco.anns[self.coco.anns.keys()[0]]['keypoints'])==17, "When evaluating only body, please provide an annotation val file with only the 17 body kps"
#             for count in range(len(self.coco.anns)):
#                 self.coco.anns[count+1]['keypoints']=self.coco.anns[count+1]['keypoints'][0:51]
# =============================================================================

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
        
        self.eval = COCOeval(self.coco, coco_eval, iouType=self.iou_type)
        LOG.info('cat_ids: %s', self.category_ids)
        if self.category_ids:
            self.eval.params.catIds = self.category_ids
        if self.keypoint_oks_sigmas is not None:
            self.eval.params.kpt_oks_sigmas = np.asarray(self.keypoint_oks_sigmas)

        if image_ids is not None:
            print('image ids', image_ids)
            self.eval.params.imgIds = image_ids
        if self.kps_select=='only_body':
            text = "Results for considering ONLY body keypoints"
            # assert len(self.coco.anns[self.coco.anns.keys()[0]]['keypoints'])==17, "When evaluating only body, please provide an annotation val file with only the 17 body kps"
            for ind in self.eval.cocoGt.anns:
                self.eval.cocoGt.anns[ind]["keypoints"]=self.eval.cocoGt.anns[ind]["keypoints"][0:17*3]
            for ind in self.eval.cocoDt.anns:
                self.eval.cocoDt.anns[ind]["keypoints"]=self.eval.cocoDt.anns[ind]["keypoints"][0:17*3]
        elif self.kps_select=='only_feet':
            text = "Results for considering ONLY feet keypoints"
            # assert len(self.coco.anns[self.coco.anns.keys()[0]]['keypoints'])==17, "When evaluating only body, please provide an annotation val file with only the 17 body kps"
            for ind in self.eval.cocoGt.anns:
                self.eval.cocoGt.anns[ind]["keypoints"]=self.eval.cocoGt.anns[ind]["keypoints"][17*3:23*3]
            for ind in self.eval.cocoDt.anns:
                self.eval.cocoDt.anns[ind]["keypoints"]=self.eval.cocoDt.anns[ind]["keypoints"][17*3:23*3]
        elif self.kps_select=='only_face':
            text = "Results for considering ONLY face keypoints"
            # assert len(self.coco.anns[self.coco.anns.keys()[0]]['keypoints'])==17, "When evaluating only body, please provide an annotation val file with only the 17 body kps"
            for ind in self.eval.cocoGt.anns:
                self.eval.cocoGt.anns[ind]["keypoints"]=self.eval.cocoGt.anns[ind]["keypoints"][23*3:91*3]
            for ind in self.eval.cocoDt.anns:
                self.eval.cocoDt.anns[ind]["keypoints"]=self.eval.cocoDt.anns[ind]["keypoints"][23*3:91*3]
        elif self.kps_select=='only_hands':
            text = "Results for considering ONLY feet keypoints"
            # assert len(self.coco.anns[self.coco.anns.keys()[0]]['keypoints'])==17, "When evaluating only body, please provide an annotation val file with only the 17 body kps"
            for ind in self.eval.cocoGt.anns:
                self.eval.cocoGt.anns[ind]["keypoints"]=self.eval.cocoGt.anns[ind]["keypoints"][91*3:133*3]
            for ind in self.eval.cocoDt.anns:
                self.eval.cocoDt.anns[ind]["keypoints"]=self.eval.cocoDt.anns[ind]["keypoints"][91*3:133*3]
        elif self.kps_select=='all':
            text = "Results for considering all 133 keypoints"
        else:
            raise Exception('Keypoints for evaluation need to be specified: only_body or all')
        self.eval.evaluate()
        LOG.info("\n\n#########   "+text+"   ######### \n")
        self.eval.accumulate()
        self.eval.summarize()
        return self.eval.stats

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
