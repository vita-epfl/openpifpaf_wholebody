# Note: This code is from the authors of COCO wholebody

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from pycocotools.coco import COCO

from .myeval_body import MYeval_body
from .myeval_foot import MYeval_foot
from .myeval_face import MYeval_face
from .myeval_lefthand import MYeval_lefthand
from .myeval_righthand import MYeval_righthand
from .myeval_wholebody import MYeval_wholebody

def parse_args():
    parser = argparse.ArgumentParser(description='COCO-WholeBody mAP Evaluation')
    parser.add_argument('--res_file',
                        help='tha path to result file',
                        required=True,
                        type=str)
    parser.add_argument('--gt_file',
                        help='tha path to gt file',
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def test_body(coco,coco_dt):
    print('\n body mAP ----------------------------------')
    coco_eval = MYeval_body(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_foot(coco,coco_dt):
    print('\n foot mAP ----------------------------------')
    coco_eval = MYeval_foot(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_face(coco,coco_dt):
    print('\n face mAP ----------------------------------')
    coco_eval = MYeval_face(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_lefthand(coco,coco_dt):
    print('\n lefthand mAP ----------------------------------')
    coco_eval = MYeval_lefthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_righthand(coco,coco_dt):
    print('\n righthand mAP ----------------------------------')
    coco_eval = MYeval_righthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0

def test_wholebody(coco,coco_dt):
    print('\n wholebody mAP ----------------------------------')
    coco_eval = MYeval_wholebody(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def main():
    args = parse_args()
    coco = COCO(args.gt_file)

    coco_dt = coco.loadRes(args.res_file)
    print('Testing: {}'.format(args.res_file), flush=True)

    test_body(coco,coco_dt)
    test_foot(coco, coco_dt)
    test_face(coco, coco_dt)
    test_lefthand(coco, coco_dt)
    test_righthand(coco, coco_dt)
    test_wholebody(coco, coco_dt)

if  __name__ == '__main__':
   main()
